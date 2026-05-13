//! Routed experts merging utilities for vLLM P/D disaggregation.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashMap;

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
const NPY_V1_HEADER_PREFIX_LEN: usize = 10;
const ROUTED_EXPERTS_PROMPT_START: &str = "routed_experts_prompt_start";

#[derive(Debug, Default)]
pub struct RoutedExpertsPrefixCache {
    root: Mutex<RoutedExpertsTrieNode>,
}

impl RoutedExpertsPrefixCache {
    fn recover_and_store_prompt(
        &self,
        request_json: &Value,
        payload: &mut Value,
    ) -> Result<(), String> {
        let tokens = prompt_tokens_from_request(request_json)?;
        let prompt_start = routed_experts_prompt_start(request_json)?;
        let mut tensor = decode_routed_experts_value(payload, "prefill routed_experts")?;

        if prompt_start + tensor.seq_len > tokens.len() {
            return Err(format!(
                "prefill routed_experts length {} with prompt start {} exceeds request token length {}",
                tensor.seq_len,
                prompt_start,
                tokens.len()
            ));
        }

        let mut root = self.root.lock();
        let recovered = fill_placeholders_from_trie(&root, &tokens, prompt_start, &mut tensor)?;
        store_tensor_in_trie(&mut root, &tokens, prompt_start, &tensor);
        drop(root);

        if recovered > 0 {
            *payload = Value::String(encode_routed_experts_payload(&tensor)?);
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
struct RoutedExpertsTrieNode {
    row: Option<Vec<i32>>,
    children: HashMap<i64, RoutedExpertsTrieNode>,
}

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
    Int32,
}

impl RoutedExpertsDtype {
    fn item_size(self) -> usize {
        match self {
            Self::Uint8 => 1,
            Self::Int32 => 4,
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

    fn row_mut(&mut self, row_idx: usize) -> &mut [i32] {
        let range = self.row_range(row_idx);
        &mut self.data[range]
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
}

pub fn prefill_has_routed_experts(prefill_json: &Value) -> bool {
    prompt_routed_experts(prefill_json).is_some()
}

pub fn merge_routed_experts_in_json(
    prefill_json: &Value,
    decode_json: &mut Value,
    request_json: &Value,
    prefix_cache: &RoutedExpertsPrefixCache,
) -> Result<bool, String> {
    let prompt_tokens = prompt_tokens_from_request(request_json)?;
    let prompt_len = prompt_tokens.len();
    let prompt_routed = prompt_routed_experts(prefill_json);
    let decode_has_routing = decode_has_routed_experts(decode_json);

    if prompt_routed.is_none() && !decode_has_routing {
        return Ok(false);
    }

    let prompt_routed = prompt_routed.ok_or_else(|| {
        "decode response contained routed_experts, but prefill response did not".to_string()
    })?;

    if !all_required_decode_choices_have_routed_experts(decode_json) {
        return Err(
            "prefill response contained routed_experts, but decode response did not contain choice routed_experts"
                .to_string(),
        );
    }
    normalize_decode_routed_experts(decode_json, prompt_len)?;

    let decode_obj = decode_json
        .as_object_mut()
        .ok_or_else(|| "decode response is not a JSON object".to_string())?;
    decode_obj.insert("prompt_routed_experts".to_string(), prompt_routed.clone());

    let prompt_payload = decode_obj
        .get_mut("prompt_routed_experts")
        .ok_or_else(|| "decode response prompt_routed_experts was not inserted".to_string())?;
    prefix_cache.recover_and_store_prompt(request_json, prompt_payload)?;

    Ok(true)
}

fn prompt_routed_experts(prefill_json: &Value) -> Option<&Value> {
    prefill_json
        .get("prompt_routed_experts")
        .filter(|value| !value.is_null())
        .or_else(|| {
            prefill_json
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("routed_experts"))
                .filter(|value| !value.is_null())
        })
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

fn normalize_decode_routed_experts(
    decode_json: &mut Value,
    prompt_len: usize,
) -> Result<(), String> {
    let choices = decode_json
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "decode response choices must be an array".to_string())?;
    for choice in choices {
        let token_ids_len = choice_token_ids_len(choice);
        if let Some(routed_experts) = choice
            .get_mut("routed_experts")
            .filter(|value| !value.is_null())
        {
            let mut tensor =
                decode_routed_experts_value(routed_experts, "decode choice routed_experts")?;
            if tensor.has_placeholder_rows() {
                return Err("decode choice routed_experts contained placeholder rows".to_string());
            }
            let expected_len = token_ids_len
                .ok_or_else(|| {
                    "decode choice with routed_experts must contain token_ids".to_string()
                })?
                .saturating_sub(1);
            match tensor.seq_len {
                len if len == expected_len => {}
                len if len == prompt_len + expected_len => {
                    tensor = tensor.slice_rows(prompt_len, prompt_len + expected_len)?;
                    *routed_experts = Value::String(encode_routed_experts_payload(&tensor)?);
                }
                len => {
                    return Err(format!(
                        "decode choice routed_experts has {len} rows, expected {expected_len} completion rows or {} full-sequence rows",
                        prompt_len + expected_len
                    ));
                }
            }
        } else if decode_choice_requires_routed_experts(choice) {
            return Err("decode choice routed_experts is missing".to_string());
        }
    }
    Ok(())
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

fn prompt_tokens_from_request(request_json: &Value) -> Result<Vec<i64>, String> {
    let tokens_value = request_json
        .get("prompt_token_ids")
        .or_else(|| request_json.get("tokens"))
        .ok_or_else(|| {
            "routed_experts responses require tokenized request field prompt_token_ids or tokens"
                .to_string()
        })?;
    let tokens = tokens_value
        .as_array()
        .ok_or_else(|| "request prompt_token_ids/tokens must be an array".to_string())?;

    tokens
        .iter()
        .map(|token| {
            token.as_i64().ok_or_else(|| {
                "request prompt_token_ids/tokens must contain integer token ids".to_string()
            })
        })
        .collect()
}

fn routed_experts_prompt_start(request_json: &Value) -> Result<usize, String> {
    request_json
        .get(ROUTED_EXPERTS_PROMPT_START)
        .map(|value| {
            value
                .as_u64()
                .map(|start| start as usize)
                .ok_or_else(|| format!("{ROUTED_EXPERTS_PROMPT_START} must be an unsigned integer"))
        })
        .unwrap_or(Ok(0))
}

fn fill_placeholders_from_trie(
    root: &RoutedExpertsTrieNode,
    tokens: &[i64],
    prompt_start: usize,
    tensor: &mut RoutedExpertsTensor,
) -> Result<usize, String> {
    let mut node = Some(root);
    for token in tokens.iter().take(prompt_start) {
        node = node.and_then(|current| current.children.get(token));
    }

    let mut recovered = 0;
    for row_idx in 0..tensor.seq_len {
        let token_idx = prompt_start + row_idx;
        node = node.and_then(|current| current.children.get(&tokens[token_idx]));

        if tensor.row_is_placeholder(row_idx) {
            let current = node.ok_or_else(|| {
                format!("missing cached routed_experts for placeholder prompt token row {row_idx}")
            })?;
            let cached_row = current.row.as_ref().ok_or_else(|| {
                format!(
                    "missing cached routed_experts row for placeholder prompt token row {row_idx}"
                )
            })?;
            if cached_row.len() != tensor.row_width() {
                return Err(format!(
                    "cached routed_experts row width {} does not match response row width {}",
                    cached_row.len(),
                    tensor.row_width()
                ));
            }
            tensor.row_mut(row_idx).copy_from_slice(cached_row);
            recovered += 1;
        }
    }

    Ok(recovered)
}

fn store_tensor_in_trie(
    root: &mut RoutedExpertsTrieNode,
    tokens: &[i64],
    prompt_start: usize,
    tensor: &RoutedExpertsTensor,
) {
    let mut node = root;
    for token in tokens.iter().take(prompt_start) {
        node = node.children.entry(*token).or_default();
    }
    for row_idx in 0..tensor.seq_len {
        let token_idx = prompt_start + row_idx;
        node = node.children.entry(tokens[token_idx]).or_default();
        if !tensor.row_is_placeholder(row_idx) {
            node.row = Some(tensor.row(row_idx).to_vec());
        }
    }
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
    Err(format!(
        "{name} must be uint8 or little-endian int32 .npy data"
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
    let header_body = format!(
        "{{'descr': '<i4', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        tensor.seq_len, tensor.layers, tensor.topk
    );
    let mut header = header_body.into_bytes();
    let padding = (16 - ((NPY_V1_HEADER_PREFIX_LEN + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat(b' ').take(padding));
    header.push(b'\n');

    let header_len = u16::try_from(header.len())
        .map_err(|_| "routed_experts NumPy header is too large for v1 .npy".to_string())?;
    let mut bytes =
        Vec::with_capacity(NPY_V1_HEADER_PREFIX_LEN + header.len() + tensor.data.len() * 4);
    bytes.extend_from_slice(NPY_MAGIC);
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header);
    for value in &tensor.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    Ok(STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(seq_len: usize, layers: usize, topk: usize, data: Vec<i32>) -> String {
        encode_routed_experts_payload(&RoutedExpertsTensor {
            seq_len,
            layers,
            topk,
            data,
        })
        .unwrap()
    }

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
    fn merge_stores_prompt_routed_experts_by_token_prefix() {
        let cache = RoutedExpertsPrefixCache::default();
        let prompt_payload = uint8_payload(2, 1, 2, vec![10, 11, 20, 21]);
        let full_decode_payload = uint8_payload(3, 1, 2, vec![10, 11, 20, 21, 30, 31]);
        let prefill = json!({"choices": [{"routed_experts": prompt_payload}]});
        let mut decode =
            json!({"choices": [{"token_ids": [7, 8], "routed_experts": full_decode_payload}]});
        let request = json!({"prompt_token_ids": [101, 102]});

        assert!(merge_routed_experts_in_json(&prefill, &mut decode, &request, &cache).unwrap());
        let prompt = decode_routed_experts_value(
            decode.get("prompt_routed_experts").unwrap(),
            "merged prompt routed_experts",
        )
        .unwrap();
        let completion = decode_routed_experts_value(
            decode["choices"][0].get("routed_experts").unwrap(),
            "merged decode routed_experts",
        )
        .unwrap();

        assert_eq!(prompt.data, vec![10, 11, 20, 21]);
        assert_eq!(completion.data, vec![30, 31]);
    }

    #[test]
    fn merge_recovers_external_kv_placeholders_from_prefix_cache() {
        let cache = RoutedExpertsPrefixCache::default();
        let initial_prompt = uint8_payload(3, 1, 2, vec![10, 11, 20, 21, 30, 31]);
        let completion = uint8_payload(1, 1, 2, vec![40, 41]);
        let request = json!({"prompt_token_ids": [101, 102, 103]});

        let prefill = json!({"choices": [{"routed_experts": initial_prompt}]});
        let mut decode =
            json!({"choices": [{"token_ids": [7, 8], "routed_experts": completion.clone()}]});
        merge_routed_experts_in_json(&prefill, &mut decode, &request, &cache).unwrap();

        let placeholder_prompt = payload(3, 1, 2, vec![-1, -1, -1, -1, 30, 31]);
        let prefill = json!({"choices": [{"routed_experts": placeholder_prompt}]});
        let mut decode = json!({"choices": [{"token_ids": [7, 8], "routed_experts": completion}]});

        merge_routed_experts_in_json(&prefill, &mut decode, &request, &cache).unwrap();
        let prompt = decode_routed_experts_value(
            decode.get("prompt_routed_experts").unwrap(),
            "merged prompt routed_experts",
        )
        .unwrap();

        assert_eq!(prompt.data, vec![10, 11, 20, 21, 30, 31]);
    }
}
