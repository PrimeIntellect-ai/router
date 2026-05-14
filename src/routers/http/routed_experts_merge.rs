//! Stitch compact routed-experts payloads for vLLM P/D disaggregation.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::Value;

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
const NPY_V1_HEADER_PREFIX_LEN: usize = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoutedExpertsPayload {
    seq_len: usize,
    layers: usize,
    topk: usize,
    descr: String,
    item_size: usize,
    data: Vec<u8>,
}

impl RoutedExpertsPayload {
    fn suffix_rows(&self, row_start: usize) -> Result<Self, String> {
        let row_size = self.layers * self.topk * self.item_size;
        let byte_start = row_start * row_size;
        let data = self
            .data
            .get(byte_start..)
            .ok_or_else(|| {
                format!(
                    "decode routed_experts has {} rows, expected at least {row_start}",
                    self.seq_len
                )
            })?
            .to_vec();

        Ok(Self {
            seq_len: self.seq_len - row_start,
            layers: self.layers,
            topk: self.topk,
            descr: self.descr.clone(),
            item_size: self.item_size,
            data,
        })
    }

    fn concat_rows(&self, other: &Self) -> Result<Self, String> {
        if self.descr != other.descr || self.layers != other.layers || self.topk != other.topk {
            return Err(format!(
                "cannot concatenate routed_experts with shapes/dtypes ({}, {}, {}, {}) and ({}, {}, {}, {})",
                self.seq_len,
                self.layers,
                self.topk,
                self.descr,
                other.seq_len,
                other.layers,
                other.topk,
                other.descr,
            ));
        }

        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);

        Ok(Self {
            seq_len: self.seq_len + other.seq_len,
            layers: self.layers,
            topk: self.topk,
            descr: self.descr.clone(),
            item_size: self.item_size,
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
    if prefill_routed.is_none() && !decode_has_routed_experts(decode_json) {
        return Ok(false);
    }

    let prompt = decode_routed_experts_value(
        prefill_routed.ok_or_else(|| {
            "decode response contained routed_experts, but prefill response did not".to_string()
        })?,
        "prefill routed_experts",
    )?;

    let choices = decode_json["choices"]
        .as_array_mut()
        .ok_or_else(|| "decode response choices must be an array".to_string())?;

    for choice in choices {
        let routed_experts = choice
            .get("routed_experts")
            .filter(|value| !value.is_null())
            .ok_or_else(|| "decode choice routed_experts is missing".to_string())?;
        let decode = decode_routed_experts_value(routed_experts, "decode routed_experts")?;
        let completion = decode.suffix_rows(prompt.seq_len)?;
        let merged = prompt.concat_rows(&completion)?;
        choice["routed_experts"] = Value::String(encode_routed_experts_payload(&merged)?);
    }

    Ok(true)
}

fn prefill_choice_routed_experts(prefill_json: &Value) -> Option<&Value> {
    prefill_json["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("routed_experts"))
        .filter(|value| !value.is_null())
}

fn decode_has_routed_experts(decode_json: &Value) -> bool {
    decode_json["choices"]
        .as_array()
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

fn decode_routed_experts_value(value: &Value, name: &str) -> Result<RoutedExpertsPayload, String> {
    let payload = value
        .as_str()
        .ok_or_else(|| format!("{name} must be a base64 .npy string"))?;
    let bytes = STANDARD
        .decode(payload)
        .map_err(|error| format!("{name} base64 decode failed: {error}"))?;
    parse_npy_payload(&bytes, name)
}

fn parse_npy_payload(bytes: &[u8], name: &str) -> Result<RoutedExpertsPayload, String> {
    if bytes.len() < NPY_V1_HEADER_PREFIX_LEN || &bytes[..6] != NPY_MAGIC {
        return Err(format!("{name} is not a NumPy .npy payload"));
    }

    let (header_len, data_start) = match bytes[6] {
        1 => (
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
            NPY_V1_HEADER_PREFIX_LEN,
        ),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        version => return Err(format!("{name} has unsupported .npy version {version}")),
    };

    let header_end = data_start + header_len;
    let header = std::str::from_utf8(&bytes[data_start..header_end])
        .map_err(|error| format!("{name} header decode failed: {error}"))?;
    let descr = parse_descr(header, name)?;
    let item_size = dtype_item_size(&descr, name)?;
    let (seq_len, layers, topk) = parse_shape(header, name)?;
    let data = bytes[header_end..].to_vec();
    let expected_data_len = seq_len * layers * topk * item_size;
    if data.len() != expected_data_len {
        return Err(format!(
            "{name} has {} data bytes, expected {expected_data_len}",
            data.len()
        ));
    }

    Ok(RoutedExpertsPayload {
        seq_len,
        layers,
        topk,
        descr,
        item_size,
        data,
    })
}

fn parse_descr(header: &str, name: &str) -> Result<String, String> {
    let after_key = header
        .split("'descr':")
        .nth(1)
        .or_else(|| header.split("\"descr\":").nth(1))
        .ok_or_else(|| format!("{name} header is missing descr"))?;
    let quote = after_key
        .find(['\'', '"'])
        .ok_or_else(|| format!("{name} descr is missing opening quote"))?;
    let after_quote = &after_key[quote + 1..];
    let end_quote = after_quote
        .find(['\'', '"'])
        .ok_or_else(|| format!("{name} descr is missing closing quote"))?;
    Ok(after_quote[..end_quote].to_string())
}

fn dtype_item_size(descr: &str, name: &str) -> Result<usize, String> {
    match descr {
        "|u1" => Ok(1),
        "<i2" => Ok(2),
        "<i4" => Ok(4),
        _ => Err(format!("{name} has unsupported dtype {descr}")),
    }
}

fn parse_shape(header: &str, name: &str) -> Result<(usize, usize, usize), String> {
    let shape_header = header
        .split("shape")
        .nth(1)
        .ok_or_else(|| format!("{name} header is missing shape"))?;
    let shape_start = shape_header
        .find('(')
        .ok_or_else(|| format!("{name} shape is missing '('"))?;
    let shape_end = shape_header[shape_start + 1..]
        .find(')')
        .ok_or_else(|| format!("{name} shape is missing ')'"))?
        + shape_start
        + 1;
    let dims = shape_header[shape_start + 1..shape_end]
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("{name} shape parse failed: {error}"))?;

    match dims.as_slice() {
        [seq_len, layers, topk] => Ok((*seq_len, *layers, *topk)),
        _ => Err(format!("{name} must have shape (seq, layers, topk)")),
    }
}

fn encode_routed_experts_payload(payload: &RoutedExpertsPayload) -> Result<String, String> {
    let header_body = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        payload.descr, payload.seq_len, payload.layers, payload.topk
    );
    let mut header = header_body.into_bytes();
    let padding = (16 - ((NPY_V1_HEADER_PREFIX_LEN + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat_n(b' ', padding));
    header.push(b'\n');

    let header_len = u16::try_from(header.len())
        .map_err(|_| "routed_experts NumPy header is too large for v1 .npy".to_string())?;
    let mut bytes =
        Vec::with_capacity(NPY_V1_HEADER_PREFIX_LEN + header.len() + payload.data.len());
    bytes.extend_from_slice(NPY_MAGIC);
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload.data);
    Ok(STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn uint8_payload(seq_len: usize, layers: usize, topk: usize, data: &[u8]) -> String {
        let payload = RoutedExpertsPayload {
            seq_len,
            layers,
            topk,
            descr: "|u1".to_string(),
            item_size: 1,
            data: data.to_vec(),
        };
        encode_routed_experts_payload(&payload).unwrap()
    }

    #[test]
    fn merge_replaces_decode_prompt_routing_with_prefill_routing() {
        let prompt_payload = uint8_payload(2, 1, 2, &[10, 11, 20, 21]);
        let decode_payload = uint8_payload(3, 1, 2, &[0, 0, 1, 1, 30, 31]);
        let prefill = json!({
            "choices": [{"routed_experts": prompt_payload}],
        });
        let mut decode = json!({"choices": [{"routed_experts": decode_payload}]});

        assert!(merge_routed_experts_in_json(&prefill, &mut decode).unwrap());
        let merged = decode_routed_experts_value(
            decode["choices"][0].get("routed_experts").unwrap(),
            "merged routed_experts",
        )
        .unwrap();

        assert_eq!(merged.seq_len, 3);
        assert_eq!(merged.data, vec![10, 11, 20, 21, 30, 31]);
    }
}
