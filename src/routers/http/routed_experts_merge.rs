//! Routed experts merging utilities for vLLM P/D disaggregation.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::Value;

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
const NPY_V1_HEADER_PREFIX_LEN: usize = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoutedExpertsTensor {
    seq_len: usize,
    layers: usize,
    topk: usize,
    data: Vec<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoutedExpertsDtype {
    Uint8,
    Int16,
    Int32,
}

impl RoutedExpertsDtype {
    fn item_size(self) -> usize {
        match self {
            Self::Uint8 => 1,
            Self::Int16 => 2,
            Self::Int32 => 4,
        }
    }

    fn descr(self) -> &'static str {
        match self {
            Self::Uint8 => "|u1",
            Self::Int16 => "<i2",
            Self::Int32 => "<i4",
        }
    }
}

impl RoutedExpertsTensor {
    fn row_width(&self) -> usize {
        self.layers * self.topk
    }

    fn row_range(&self, row_idx: usize) -> std::ops::Range<usize> {
        let row_start = row_idx * self.row_width();
        row_start..row_start + self.row_width()
    }

    fn row(&self, row_idx: usize) -> &[i32] {
        &self.data[self.row_range(row_idx)]
    }

    fn row_is_placeholder(&self, row_idx: usize) -> bool {
        self.row(row_idx).iter().all(|value| *value == -1)
    }

    fn has_placeholder_rows(&self) -> bool {
        (0..self.seq_len).any(|row_idx| self.row_is_placeholder(row_idx))
    }

    fn slice_rows(&self, start: usize, end: usize) -> Result<Self, String> {
        if start > end || end > self.seq_len {
            return Err(format!(
                "cannot slice routed_experts rows {start}..{end} from {} rows",
                self.seq_len
            ));
        }
        let row_width = self.row_width();
        Ok(Self {
            seq_len: end - start,
            layers: self.layers,
            topk: self.topk,
            data: self.data[start * row_width..end * row_width].to_vec(),
        })
    }

    fn empty_like(&self) -> Self {
        Self {
            seq_len: 0,
            layers: self.layers,
            topk: self.topk,
            data: Vec::new(),
        }
    }

    fn concat_rows(&self, other: &Self) -> Result<Self, String> {
        if self.layers != other.layers || self.topk != other.topk {
            return Err(format!(
                "cannot concatenate routed_experts with shapes ({}, {}, {}) and ({}, {}, {})",
                self.seq_len, self.layers, self.topk, other.seq_len, other.layers, other.topk
            ));
        }
        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);
        Ok(Self {
            seq_len: self.seq_len + other.seq_len,
            layers: self.layers,
            topk: self.topk,
            data,
        })
    }
}

pub fn prefill_has_routed_experts(prefill_json: &Value) -> bool {
    prefill_choice_routed_experts(prefill_json).is_some()
}

pub fn merge_routed_experts_in_json(
    prefill_json: &Value,
    decode_json: &mut Value,
) -> Result<bool, String> {
    let prefill_routed = prefill_choice_routed_experts(prefill_json);
    let decode_has_routing = decode_has_routed_experts(decode_json);

    if prefill_routed.is_none() && !decode_has_routing {
        return Ok(false);
    }

    let prefill_routed = prefill_routed.ok_or_else(|| {
        "decode response contained routed_experts, but prefill response did not".to_string()
    })?;

    if !all_required_decode_choices_have_routed_experts(decode_json) {
        return Err(
            "prefill response contained routed_experts, but decode response did not contain choice routed_experts"
            .to_string(),
        );
    }
    let prompt_len = prompt_token_ids_len_from_prefill_response(prefill_json)?;
    let prompt_tensor = decode_routed_experts_value(prefill_routed, "prefill routed_experts")?;
    if prompt_tensor.seq_len > prompt_len {
        return Err(format!(
            "prefill routed_experts has {} rows, but prompt_token_ids has {prompt_len} tokens",
            prompt_tensor.seq_len
        ));
    }
    if prompt_tensor.has_placeholder_rows() {
        return Err(
            "prefill routed_experts contained placeholder rows; external KV routed-experts recovery is not supported"
                .to_string(),
        );
    }
    merge_prompt_into_decode_choices(decode_json, prompt_len, &prompt_tensor)?;

    Ok(true)
}

fn prefill_choice_routed_experts(prefill_json: &Value) -> Option<&Value> {
    prefill_json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("routed_experts"))
        .filter(|value| !value.is_null())
}

fn decode_has_routed_experts(decode_json: &Value) -> bool {
    decode_json
        .get("choices")
        .and_then(Value::as_array)
        .map(|choices| {
            choices.iter().any(|choice| {
                choice
                    .get("routed_experts")
                    .filter(|value| !value.is_null())
                    .is_some()
            })
        })
        .unwrap_or(false)
}

fn all_required_decode_choices_have_routed_experts(decode_json: &Value) -> bool {
    decode_json
        .get("choices")
        .and_then(Value::as_array)
        .map(|choices| {
            !choices.is_empty()
                && choices.iter().all(|choice| {
                    decode_choice_has_routed_experts(choice)
                        || !decode_choice_requires_routed_experts(choice)
                })
        })
        .unwrap_or(false)
}

fn merge_prompt_into_decode_choices(
    decode_json: &mut Value,
    prompt_len: usize,
    prompt_tensor: &RoutedExpertsTensor,
) -> Result<(), String> {
    let choices = decode_json
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "decode response choices must be an array".to_string())?;
    for choice in choices {
        let expected_completion_len = choice_token_ids_len(choice)
            .ok_or_else(|| "decode choice must contain token_ids".to_string())?
            .saturating_sub(1);
        let completion_tensor = if let Some(routed_experts) = choice
            .get("routed_experts")
            .filter(|value| !value.is_null())
        {
            completion_from_decode_routed_experts(
                routed_experts,
                prompt_len,
                prompt_tensor.seq_len,
                expected_completion_len,
            )?
        } else if expected_completion_len == 0 {
            prompt_tensor.empty_like()
        } else {
            return Err("decode choice routed_experts is missing".to_string());
        };
        if completion_tensor.has_placeholder_rows() {
            return Err(
                "decode choice completion routed_experts contained placeholder rows".to_string(),
            );
        }
        let merged = prompt_tensor.concat_rows(&completion_tensor)?;
        if let Some(choice_obj) = choice.as_object_mut() {
            choice_obj.insert(
                "routed_experts".to_string(),
                Value::String(encode_routed_experts_payload(&merged)?),
            );
        } else {
            return Err("decode response choice must be a JSON object".to_string());
        };
    }
    Ok(())
}

fn completion_from_decode_routed_experts(
    routed_experts: &Value,
    prompt_len: usize,
    prompt_routed_len: usize,
    expected_completion_len: usize,
) -> Result<RoutedExpertsTensor, String> {
    let tensor = decode_routed_experts_value(routed_experts, "decode choice routed_experts")?;
    match tensor.seq_len {
        len if len == expected_completion_len => Ok(tensor),
        len if len == prompt_routed_len + expected_completion_len
            || len == prompt_len + expected_completion_len =>
        {
            tensor.slice_rows(len - expected_completion_len, len)
        }
        len => Err(format!(
            "decode choice routed_experts has {len} rows, expected {expected_completion_len} completion rows or {} full-sequence rows",
            prompt_len + expected_completion_len
        )),
    }
}

fn choice_token_ids_len(choice: &Value) -> Option<usize> {
    choice
        .get("token_ids")
        .and_then(Value::as_array)
        .map(Vec::len)
}

fn decode_choice_has_routed_experts(choice: &Value) -> bool {
    choice
        .get("routed_experts")
        .filter(|value| !value.is_null())
        .is_some()
}

fn decode_choice_requires_routed_experts(choice: &Value) -> bool {
    choice
        .get("token_ids")
        .and_then(Value::as_array)
        .map(|token_ids| token_ids.len() > 1)
        .unwrap_or(true)
}

fn prompt_token_ids_len_from_prefill_response(prefill_json: &Value) -> Result<usize, String> {
    let tokens_value = prefill_json
        .get("prompt_token_ids")
        .ok_or_else(|| "prefill routed_experts response requires prompt_token_ids".to_string())?;
    Ok(tokens_value
        .as_array()
        .map(Vec::len)
        .ok_or_else(|| "prefill response prompt_token_ids must be an array".to_string())?)
}

fn decode_routed_experts_value(value: &Value, name: &str) -> Result<RoutedExpertsTensor, String> {
    let data = value
        .as_str()
        .ok_or_else(|| format!("{name} must be a base64 .npy string"))?;
    if data.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    decode_routed_experts_payload(data, name)
}

fn decode_routed_experts_payload(payload: &str, name: &str) -> Result<RoutedExpertsTensor, String> {
    let bytes = STANDARD
        .decode(payload)
        .map_err(|error| format!("{name} must be valid base64: {error}"))?;
    parse_npy_i32(&bytes, name)
}

fn parse_npy_i32(bytes: &[u8], name: &str) -> Result<RoutedExpertsTensor, String> {
    if bytes.len() < NPY_V1_HEADER_PREFIX_LEN || &bytes[..6] != NPY_MAGIC {
        return Err(format!("{name} must be a NumPy .npy payload"));
    }

    let major = bytes[6];
    let (header_len, data_start) = match major {
        1 => {
            let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (header_len, NPY_V1_HEADER_PREFIX_LEN)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(format!("{name} has a truncated NumPy header"));
            }
            let header_len =
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (header_len, 12)
        }
        version => {
            return Err(format!(
                "{name} has unsupported NumPy .npy major version {version}"
            ))
        }
    };

    let header_end = data_start + header_len;
    if bytes.len() < header_end {
        return Err(format!("{name} has a truncated NumPy header"));
    }

    let header = std::str::from_utf8(&bytes[data_start..header_end])
        .map_err(|error| format!("{name} NumPy header is not UTF-8: {error}"))?;
    let dtype = parse_npy_dtype(header, name)?;
    validate_npy_header(header, name)?;
    let (seq_len, layers, topk) = parse_shape(header, name)?;

    let data_bytes = &bytes[header_end..];
    let expected_len = seq_len
        .checked_mul(layers)
        .and_then(|value| value.checked_mul(topk))
        .and_then(|value| value.checked_mul(dtype.item_size()))
        .ok_or_else(|| format!("{name} shape overflows routed_experts size"))?;
    if data_bytes.len() != expected_len {
        return Err(format!(
            "{name} data length {} does not match shape ({seq_len}, {layers}, {topk})",
            data_bytes.len()
        ));
    }

    let data = match dtype {
        RoutedExpertsDtype::Uint8 => data_bytes.iter().map(|value| i32::from(*value)).collect(),
        RoutedExpertsDtype::Int16 => data_bytes
            .chunks_exact(2)
            .map(|chunk| i32::from(i16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        RoutedExpertsDtype::Int32 => data_bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    };

    Ok(RoutedExpertsTensor {
        seq_len,
        layers,
        topk,
        data,
    })
}

fn parse_npy_dtype(header: &str, name: &str) -> Result<RoutedExpertsDtype, String> {
    if header.contains("'<i4'") || header.contains("\"<i4\"") {
        return Ok(RoutedExpertsDtype::Int32);
    }
    if header.contains("'|u1'") || header.contains("\"|u1\"") {
        return Ok(RoutedExpertsDtype::Uint8);
    }
    if header.contains("'<i2'") || header.contains("\"<i2\"") {
        return Ok(RoutedExpertsDtype::Int16);
    }
    Err(format!(
        "{name} must be uint8, little-endian int16, or little-endian int32 .npy data"
    ))
}

fn validate_npy_header(header: &str, name: &str) -> Result<(), String> {
    if !(header.contains("fortran_order") && header.contains("False")) {
        return Err(format!("{name} must be C-order .npy data"));
    }
    Ok(())
}

fn parse_shape(header: &str, name: &str) -> Result<(usize, usize, usize), String> {
    let shape_pos = header
        .find("shape")
        .ok_or_else(|| format!("{name} NumPy header is missing shape"))?;
    let shape_header = &header[shape_pos..];
    let shape_start = shape_header
        .find('(')
        .ok_or_else(|| format!("{name} NumPy shape is missing '('"))?;
    let shape_end = shape_header[shape_start + 1..]
        .find(')')
        .ok_or_else(|| format!("{name} NumPy shape is missing ')'"))?
        + shape_start
        + 1;
    let shape_values = &shape_header[shape_start + 1..shape_end];
    let dims = shape_values
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("{name} NumPy shape contains invalid dimension: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    match dims.as_slice() {
        [seq_len, layers, topk] => Ok((*seq_len, *layers, *topk)),
        _ => Err(format!("{name} must have shape (seq, layers, topk)")),
    }
}

fn encode_routed_experts_payload(tensor: &RoutedExpertsTensor) -> Result<String, String> {
    let dtype = compact_dtype(&tensor.data);
    let header_body = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        dtype.descr(),
        tensor.seq_len,
        tensor.layers,
        tensor.topk
    );
    let mut header = header_body.into_bytes();
    let padding = (16 - ((NPY_V1_HEADER_PREFIX_LEN + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat(b' ').take(padding));
    header.push(b'\n');

    let header_len = u16::try_from(header.len())
        .map_err(|_| "routed_experts NumPy header is too large for v1 .npy".to_string())?;
    let mut bytes = Vec::with_capacity(
        NPY_V1_HEADER_PREFIX_LEN + header.len() + tensor.data.len() * dtype.item_size(),
    );
    bytes.extend_from_slice(NPY_MAGIC);
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header);
    for value in &tensor.data {
        match dtype {
            RoutedExpertsDtype::Uint8 => {
                bytes.push(u8::try_from(*value).map_err(|_| {
                    format!("routed_experts value {value} cannot be encoded as uint8")
                })?)
            }
            RoutedExpertsDtype::Int16 => bytes.extend_from_slice(
                &i16::try_from(*value)
                    .map_err(|_| {
                        format!("routed_experts value {value} cannot be encoded as int16")
                    })?
                    .to_le_bytes(),
            ),
            RoutedExpertsDtype::Int32 => bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }

    Ok(STANDARD.encode(bytes))
}

fn compact_dtype(data: &[i32]) -> RoutedExpertsDtype {
    if data
        .iter()
        .all(|value| (0..=i32::from(u8::MAX)).contains(value))
    {
        return RoutedExpertsDtype::Uint8;
    }
    if data
        .iter()
        .all(|value| (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(value))
    {
        return RoutedExpertsDtype::Int16;
    }
    RoutedExpertsDtype::Int32
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn uint8_payload(seq_len: usize, layers: usize, topk: usize, data: Vec<u8>) -> String {
        let header_body = format!(
            "{{'descr': '|u1', 'fortran_order': False, 'shape': ({seq_len}, {layers}, {topk}), }}"
        );
        let mut header = header_body.into_bytes();
        let padding = (16 - ((NPY_V1_HEADER_PREFIX_LEN + header.len() + 1) % 16)) % 16;
        header.extend(std::iter::repeat(b' ').take(padding));
        header.push(b'\n');

        let mut bytes = Vec::new();
        bytes.extend_from_slice(NPY_MAGIC);
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        STANDARD.encode(bytes)
    }

    #[test]
    fn merge_stores_prefill_choice_routing_by_token_prefix() {
        let prompt_payload = uint8_payload(2, 1, 2, vec![10, 11, 20, 21]);
        let full_decode_payload = uint8_payload(3, 1, 2, vec![10, 11, 20, 21, 30, 31]);
        let prefill = json!({
            "prompt_token_ids": [101, 102],
            "choices": [{"routed_experts": prompt_payload}],
        });
        let mut decode =
            json!({"choices": [{"token_ids": [7, 8], "routed_experts": full_decode_payload}]});

        assert!(merge_routed_experts_in_json(&prefill, &mut decode).unwrap());
        let merged = decode_routed_experts_value(
            decode["choices"][0].get("routed_experts").unwrap(),
            "merged choice routed_experts",
        )
        .unwrap();

        assert_eq!(merged.data, vec![10, 11, 20, 21, 30, 31]);
    }
}
