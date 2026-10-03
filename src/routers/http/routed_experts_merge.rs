//! Stitch compact routed-experts payloads for vLLM P/D disaggregation.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoutedExpertsPayload {
    start: usize,
    seq_len: usize,
    layers: usize,
    topk: usize,
    data: Vec<u8>,
}

impl RoutedExpertsPayload {
    fn suffix_rows(&self, row_count: usize) -> Result<Self, String> {
        if row_count > self.seq_len {
            return Err(format!(
                "decode routed_experts has {} rows, expected at least {row_count}",
                self.seq_len
            ));
        }
        let row_size = self.layers * self.topk;
        let byte_start = row_count * row_size;
        let data = self
            .data
            .get(byte_start..)
            .ok_or_else(|| {
                format!(
                    "decode routed_experts has {} rows, expected at least {row_count}",
                    self.seq_len
                )
            })?
            .to_vec();

        Ok(Self {
            start: self.start + row_count,
            seq_len: self.seq_len - row_count,
            layers: self.layers,
            topk: self.topk,
            data,
        })
    }

    fn concat_rows(&self, other: &Self) -> Result<Self, String> {
        if self.layers != other.layers || self.topk != other.topk {
            return Err(format!(
                "cannot concatenate routed_experts with shapes ({}, {}, {}) and ({}, {}, {})",
                self.seq_len, self.layers, self.topk, other.seq_len, other.layers, other.topk,
            ));
        }
        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);

        Ok(Self {
            start: self.start,
            seq_len: self.seq_len + other.seq_len,
            layers: self.layers,
            topk: self.topk,
            data,
        })
    }
}

pub fn prefill_has_routed_experts(prefill_json: &Value) -> bool {
    prefill_choice_routed_experts(prefill_json).is_some()
        || !routed_payload_segments(&prefill_json["choices"][0]).is_empty()
}

/// By-handle twin of [`merge_routed_experts_in_json`]: choices carry `payload`
/// segments (`field`, `file`, `offset`, `pos`, `rows`, `dtype`, `shape`) instead
/// of inline `routed_experts`. Each decode choice gets the prefill choice's
/// routed_experts segments, then its own segments with routed_experts rows
/// before the prefill coverage end clipped away. Other fields pass through.
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
    if prefill_segments.is_empty() {
        return Err("decode payload contained routed_experts, but prefill payload did not".into());
    }
    let mut cut = 0;
    for segment in &prefill_segments {
        cut = cut.max(segment_u64(segment, "pos")? + segment_u64(segment, "rows")?);
    }

    let choices = decode_json["choices"]
        .as_array_mut()
        .ok_or_else(|| "decode response choices must be an array".to_string())?;
    for choice in choices {
        let mut merged: Vec<Value> = prefill_segments.iter().map(|s| (*s).clone()).collect();
        let decode_segments = choice["payload"].as_array().cloned().unwrap_or_default();
        for mut segment in decode_segments {
            if segment["field"] == "routed_experts" {
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

fn routed_payload_segments(choice: &Value) -> Vec<&Value> {
    choice["payload"]
        .as_array()
        .map(|segments| {
            segments
                .iter()
                .filter(|segment| segment["field"] == "routed_experts")
                .collect()
        })
        .unwrap_or_default()
}

fn segment_u64(segment: &Value, key: &str) -> Result<u64, String> {
    segment[key]
        .as_u64()
        .ok_or_else(|| format!("payload segment {key} must be a non-negative integer"))
}

fn segment_row_bytes(segment: &Value) -> Result<u64, String> {
    let itemsize = match segment["dtype"].as_str().unwrap_or_default() {
        "uint8" | "int8" | "bool" => 1,
        "uint16" | "int16" | "float16" | "bfloat16" => 2,
        "uint32" | "int32" | "float32" => 4,
        "uint64" | "int64" | "float64" => 8,
        dtype => return Err(format!("unsupported payload segment dtype {dtype:?}")),
    };
    let shape = segment["shape"]
        .as_array()
        .ok_or_else(|| "payload segment shape must be an array".to_string())?;
    shape.iter().try_fold(itemsize, |bytes, dim| {
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
    let bytes = STANDARD
        .decode(data_payload)
        .map_err(|error| format!("{name} base64 decode failed: {error}"))?;
    let expected_data_len = seq_len
        .checked_mul(layers)
        .and_then(|size| size.checked_mul(topk))
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn uint8_payload(seq_len: usize, layers: usize, topk: usize, data: &[u8]) -> Value {
        let payload = RoutedExpertsPayload {
            start: 0,
            seq_len,
            layers,
            topk,
            data: data.to_vec(),
        };
        encode_routed_experts_payload(&payload)
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

    fn segment(field: &str, file: &str, offset: u64, pos: u64, rows: u64, dtype: &str) -> Value {
        json!({"field": field, "file": file, "offset": offset, "pos": pos, "rows": rows,
               "dtype": dtype, "shape": [2, 4]})
    }

    #[test]
    fn merge_payload_prepends_prefill_routing_and_clips_decode_prompt_rows() {
        let prefill = json!({"choices": [{"payload": [
            segment("routed_experts", "p.bin", 0, 0, 2, "uint8"),
            segment("sampling_mask", "p.bin", 16, 2, 1, "int32"),
        ]}]});
        let mut decode = json!({"choices": [{"payload": [
            segment("routed_experts", "d.bin", 0, 0, 3, "uint16"),
            segment("sampling_mask", "d.bin", 48, 2, 1, "int32"),
        ]}]});

        assert!(merge_routed_payload_in_json(&prefill, &mut decode).unwrap());
        assert_eq!(
            decode["choices"][0]["payload"],
            json!([
                segment("routed_experts", "p.bin", 0, 0, 2, "uint8"),
                segment("routed_experts", "d.bin", 32, 2, 1, "uint16"),
                segment("sampling_mask", "d.bin", 48, 2, 1, "int32"),
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
