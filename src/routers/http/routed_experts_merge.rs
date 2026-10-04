//! Stitch compact routed-experts payloads for vLLM P/D disaggregation.

use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoutedExpertsPayload {
    start: usize,
    seq_len: usize,
    layers: usize,
    topk: usize,
    dtype: String,
    data: Vec<u8>,
}

impl RoutedExpertsPayload {
    fn row_bytes(&self) -> usize {
        self.layers * self.topk * dtype_itemsize(&self.dtype).expect("dtype validated on decode")
    }

    fn suffix_rows(&self, row_count: usize) -> Result<Self, String> {
        if row_count > self.seq_len {
            return Err(format!(
                "decode routed_experts has {} rows, expected at least {row_count}",
                self.seq_len
            ));
        }
        Ok(Self {
            start: self.start + row_count,
            seq_len: self.seq_len - row_count,
            data: self.data[row_count * self.row_bytes()..].to_vec(),
            ..self.clone()
        })
    }

    /// prime-rl picks uint16 only when an expert id exceeds 255, so prefill can be
    /// uint8 while decode is uint16; zero-extend the little-endian uint8 elements.
    fn widen_to_uint16(&self) -> Self {
        Self {
            dtype: "uint16".to_string(),
            data: self.data.iter().flat_map(|&byte| [byte, 0]).collect(),
            ..self.clone()
        }
    }

    fn concat_rows(&self, other: &Self) -> Result<Self, String> {
        if self.layers != other.layers || self.topk != other.topk {
            return Err(format!(
                "cannot concatenate routed_experts with shapes ({}, {}, {}) and ({}, {}, {})",
                self.seq_len, self.layers, self.topk, other.seq_len, other.layers, other.topk,
            ));
        }
        let (head, tail) = match (self.dtype.as_str(), other.dtype.as_str()) {
            (a, b) if a == b => (self.clone(), other.clone()),
            ("uint8", "uint16") => (self.widen_to_uint16(), other.clone()),
            ("uint16", "uint8") => (self.clone(), other.widen_to_uint16()),
            (a, b) => {
                return Err(format!(
                    "cannot concatenate routed_experts with dtypes {a} and {b}"
                ))
            }
        };
        let mut data = head.data;
        data.extend_from_slice(&tail.data);

        Ok(Self {
            seq_len: self.seq_len + other.seq_len,
            dtype: head.dtype,
            data,
            ..self.clone()
        })
    }
}

fn dtype_itemsize(dtype: &str) -> Option<usize> {
    match dtype {
        "uint8" => Some(1),
        "uint16" => Some(2),
        "int32" | "float32" => Some(4),
        _ => None,
    }
}

pub fn prefill_has_routed_experts(prefill_json: &Value) -> bool {
    prefill_choice_routed_experts(prefill_json).is_some()
        || !routed_payload_segments(&prefill_json["choices"][0]).is_empty()
}

/// Payload fields the forward pass produces for every position, prompt included, so the
/// prefill worker holds the prompt's rows and the decode worker the rest. Other fields
/// (sampling masks and their logprobs) cover sampled tokens only and come from decode.
const STITCHED_FIELDS: [&str; 2] = ["routed_experts", "routed_expert_weights"];

/// By-handle twin of [`merge_routed_experts_in_json`]: choices carry `payload`
/// segments (`field`, `file`, `offset`, `pos`, `rows`, `dtype`, `shape`) instead
/// of inline `routed_experts`. A segment covers the absolute token positions
/// `[pos, pos + rows)`. Under P/D the decode instance's rows start at its first
/// forward (`prompt_len - 1`), so they overlap the end of prefill's. Each decode
/// choice gets the prefill choice's segments of every [`STITCHED_FIELDS`] field,
/// then its own segments with those fields' rows before the field's prefill
/// coverage end dropped. Other fields pass through.
pub fn merge_routed_payload_in_json(
    prefill_json: &Value,
    decode_json: &mut Value,
) -> Result<bool, String> {
    let prefill_segments = routed_payload_segments(&prefill_json["choices"][0]);
    let decode_has = decode_json["choices"].as_array().is_some_and(|choices| {
        choices
            .iter()
            .any(|c| !routed_payload_segments(c).is_empty())
    });
    if prefill_segments.is_empty() && !decode_has {
        return Ok(false);
    }
    let mut cuts: HashMap<&str, u64> = HashMap::new();
    for segment in &prefill_segments {
        let end = segment_u64(segment, "pos")? + segment_u64(segment, "rows")?;
        let cut = cuts
            .entry(segment["field"].as_str().unwrap_or_default())
            .or_default();
        *cut = (*cut).max(end);
    }

    let choices = decode_json["choices"]
        .as_array_mut()
        .ok_or_else(|| "decode response choices must be an array".to_string())?;
    for choice in choices {
        let mut merged: Vec<Value> = prefill_segments.iter().map(|s| (*s).clone()).collect();
        let decode_segments = choice["payload"].as_array().cloned().unwrap_or_default();
        for mut segment in decode_segments {
            if is_stitched(&segment) {
                let field = segment["field"].as_str().unwrap_or_default();
                let cut = *cuts.get(field).ok_or_else(|| {
                    format!("decode payload contained {field}, but prefill payload did not")
                })?;
                let (pos, rows) = (
                    segment_u64(&segment, "pos")?,
                    segment_u64(&segment, "rows")?,
                );
                if pos + rows <= cut {
                    continue;
                }
                if pos < cut {
                    let offset = segment_u64(&segment, "offset")?;
                    segment["offset"] = json!(offset + (cut - pos) * segment_row_bytes(&segment)?);
                    segment["pos"] = json!(cut);
                    segment["rows"] = json!(pos + rows - cut);
                }
            }
            merged.push(segment);
        }
        choice["payload"] = Value::Array(merged);
    }
    Ok(true)
}

fn is_stitched(segment: &Value) -> bool {
    segment["field"]
        .as_str()
        .is_some_and(|field| STITCHED_FIELDS.contains(&field))
}

fn routed_payload_segments(choice: &Value) -> Vec<&Value> {
    choice["payload"]
        .as_array()
        .map(|segments| segments.iter().filter(|s| is_stitched(s)).collect())
        .unwrap_or_default()
}

fn segment_u64(segment: &Value, key: &str) -> Result<u64, String> {
    segment[key]
        .as_u64()
        .ok_or_else(|| format!("payload segment {key} must be a non-negative integer"))
}

fn segment_row_bytes(segment: &Value) -> Result<u64, String> {
    let dtype = segment["dtype"].as_str().unwrap_or_default();
    let itemsize = dtype_itemsize(dtype)
        .ok_or_else(|| format!("unsupported payload segment dtype {dtype:?}"))?;
    let shape = segment["shape"]
        .as_array()
        .ok_or_else(|| "payload segment shape must be an array".to_string())?;
    shape.iter().try_fold(itemsize as u64, |bytes, dim| {
        dim.as_u64()
            .map(|dim| bytes * dim)
            .ok_or_else(|| "payload segment shape dimension must be a non-negative integer".into())
    })
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
        choice["routed_experts"] = encode_routed_experts_payload(&merged);
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
        .as_object()
        .ok_or_else(|| format!("{name} must be an object with base64 data and shape"))?;
    let data_payload = payload
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{name} data must be a base64 string"))?;
    let start = payload
        .get("start")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{name} start must be a non-negative integer"))?;
    let start =
        usize::try_from(start).map_err(|error| format!("{name} start parse failed: {error}"))?;
    let (seq_len, layers, topk) = parse_shape(payload.get("shape"), name)?;
    let dtype = payload
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{name} dtype must be a string"))?
        .to_string();
    let itemsize =
        dtype_itemsize(&dtype).ok_or_else(|| format!("{name} has unsupported dtype {dtype:?}"))?;
    let bytes = STANDARD
        .decode(data_payload)
        .map_err(|error| format!("{name} base64 decode failed: {error}"))?;
    let expected_data_len = seq_len
        .checked_mul(layers)
        .and_then(|size| size.checked_mul(topk))
        .and_then(|size| size.checked_mul(itemsize))
        .ok_or_else(|| format!("{name} shape is too large"))?;
    if bytes.len() != expected_data_len {
        return Err(format!(
            "{name} has {} data bytes, expected {expected_data_len}",
            bytes.len()
        ));
    }

    Ok(RoutedExpertsPayload {
        start,
        seq_len,
        layers,
        topk,
        dtype,
        data: bytes,
    })
}

fn parse_shape(value: Option<&Value>, name: &str) -> Result<(usize, usize, usize), String> {
    let shape = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name} shape must be an array"))?;
    let dims = shape
        .iter()
        .map(|value| {
            let dim = value
                .as_u64()
                .ok_or_else(|| "shape dimension must be a non-negative integer".to_string())?;
            usize::try_from(dim).map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("{name} shape parse failed: {error}"))?;

    match dims.as_slice() {
        [seq_len, layers, topk] => Ok((*seq_len, *layers, *topk)),
        _ => Err(format!("{name} must have shape (seq, layers, topk)")),
    }
}

fn encode_routed_experts_payload(payload: &RoutedExpertsPayload) -> Value {
    json!({
        "data": STANDARD.encode(&payload.data),
        "shape": [payload.seq_len, payload.layers, payload.topk],
        "start": payload.start,
        "dtype": payload.dtype,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(seq_len: usize, layers: usize, topk: usize, dtype: &str, data: &[u8]) -> Value {
        let payload = RoutedExpertsPayload {
            start: 0,
            seq_len,
            layers,
            topk,
            dtype: dtype.to_string(),
            data: data.to_vec(),
        };
        encode_routed_experts_payload(&payload)
    }

    fn merge(prompt_payload: Value, decode_payload: Value) -> RoutedExpertsPayload {
        let prefill = json!({"choices": [{"routed_experts": prompt_payload}]});
        let mut decode = json!({"choices": [{"routed_experts": decode_payload}]});
        assert!(merge_routed_experts_in_json(&prefill, &mut decode).unwrap());
        decode_routed_experts_value(&decode["choices"][0]["routed_experts"], "merged").unwrap()
    }

    #[test]
    fn merge_replaces_decode_prompt_routing_with_prefill_routing() {
        let merged = merge(
            payload(2, 1, 2, "uint8", &[10, 11, 20, 21]),
            payload(3, 1, 2, "uint8", &[0, 0, 1, 1, 30, 31]),
        );

        assert_eq!(merged.seq_len, 3);
        assert_eq!(merged.data, vec![10, 11, 20, 21, 30, 31]);
    }

    #[test]
    fn merge_honors_payload_dtype() {
        // uint8 prompt rows widen to the uint16 decode rows (little-endian).
        let merged = merge(
            payload(1, 1, 2, "uint8", &[10, 11]),
            payload(2, 1, 2, "uint16", &[0, 0, 0, 0, 0, 1, 2, 1]),
        );
        assert_eq!(merged.dtype, "uint16");
        assert_eq!(merged.data, vec![10, 0, 11, 0, 0, 1, 2, 1]);
    }

    fn segment(field: &str, file: &str, offset: u64, pos: u64, rows: u64, dtype: &str) -> Value {
        json!({"field": field, "file": file, "offset": offset, "pos": pos, "rows": rows,
               "dtype": dtype, "shape": [2, 4]})
    }

    #[test]
    fn merge_payload_prepends_prefill_routing_and_drops_decode_rows_before_cut() {
        // prompt_len 3, 3 completion tokens. Prefill covers [0, 3); decode's
        // first forward is at prompt_len - 1, so it covers [2, 5).
        let prefill = json!({"choices": [{"payload": [
            segment("routed_experts", "p.bin", 0, 0, 3, "uint8"),
            segment("routed_expert_weights", "p.bin", 24, 0, 3, "float32"),
            segment("sampling_mask", "p.bin", 120, 3, 1, "int32"),
        ]}]});
        let mut decode = json!({"choices": [{"payload": [
            segment("routed_experts", "d.bin", 0, 2, 3, "uint16"),
            segment("routed_expert_weights", "d.bin", 48, 2, 3, "float32"),
            segment("sampling_mask", "d.bin", 144, 3, 3, "int32"),
            segment("sampling_mask_logprobs", "d.bin", 240, 3, 3, "float32"),
        ]}]});

        assert!(merge_routed_payload_in_json(&prefill, &mut decode).unwrap());
        assert_eq!(
            decode["choices"][0]["payload"],
            json!([
                segment("routed_experts", "p.bin", 0, 0, 3, "uint8"),
                segment("routed_expert_weights", "p.bin", 24, 0, 3, "float32"),
                segment("routed_experts", "d.bin", 16, 3, 2, "uint16"),
                segment("routed_expert_weights", "d.bin", 80, 3, 2, "float32"),
                segment("sampling_mask", "d.bin", 144, 3, 3, "int32"),
                segment("sampling_mask_logprobs", "d.bin", 240, 3, 3, "float32"),
            ])
        );
    }

    #[test]
    fn merge_payload_is_noop_without_routed_segments() {
        let prefill = json!({"choices": [{"payload": [segment("sampling_mask", "p.bin", 0, 2, 1, "int32")]}]});
        let mut decode = json!({"choices": [{"text": "x"}]});
        assert!(!merge_routed_payload_in_json(&prefill, &mut decode).unwrap());
        assert_eq!(decode, json!({"choices": [{"text": "x"}]}));
    }
}
