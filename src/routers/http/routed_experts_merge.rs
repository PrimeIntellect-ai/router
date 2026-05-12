//! Routed experts merging utilities for vLLM P/D disaggregation.

use serde_json::Value;

pub fn prefill_has_routed_experts(prefill_json: &Value) -> bool {
    prefill_json
        .get("prompt_routed_experts")
        .filter(|value| !value.is_null())
        .is_some()
}

pub fn merge_routed_experts_in_json(
    prefill_json: &Value,
    decode_json: &mut Value,
) -> Result<bool, String> {
    let prefill_prompt_routed = prefill_json
        .get("prompt_routed_experts")
        .filter(|value| !value.is_null());
    let decode_has_completion_routed = has_completion_routed_experts(decode_json);

    if prefill_prompt_routed.is_none() && !decode_has_completion_routed {
        return Ok(false);
    }

    let prefill_prompt_routed = prefill_prompt_routed.ok_or_else(|| {
        "decode response contained routed_experts, but prefill response did not contain prompt_routed_experts"
            .to_string()
    })?;

    validate_routed_experts_payload(prefill_prompt_routed, "prefill prompt_routed_experts")?;
    if !routed_experts_sum_nonzero(prefill_prompt_routed) {
        return Err("prefill prompt_routed_experts sum to zero".to_string());
    }

    if !all_choices_have_completion_routed_experts(decode_json) {
        return Err(
            "prefill response contained prompt_routed_experts, but decode response did not contain choice routed_experts"
                .to_string(),
        );
    }
    validate_completion_routed_experts(decode_json)?;

    let decode_obj = decode_json
        .as_object_mut()
        .ok_or_else(|| "decode response is not a JSON object".to_string())?;
    decode_obj.insert(
        "prompt_routed_experts".to_string(),
        prefill_prompt_routed.clone(),
    );

    Ok(true)
}

fn validate_completion_routed_experts(decode_json: &Value) -> Result<(), String> {
    let Some(choices) = decode_json
        .get("choices")
        .and_then(|value| value.as_array())
    else {
        return Ok(());
    };

    for choice in choices {
        let Some(routed_experts) = choice
            .get("routed_experts")
            .filter(|value| !value.is_null())
        else {
            continue;
        };
        validate_routed_experts_payload(routed_experts, "decode choice routed_experts")?;
        if !routed_experts_is_empty(routed_experts) && !routed_experts_sum_nonzero(routed_experts) {
            return Err("decode choice routed_experts sum to zero".to_string());
        }
    }

    Ok(())
}

fn has_completion_routed_experts(decode_json: &Value) -> bool {
    decode_json
        .get("choices")
        .and_then(|value| value.as_array())
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

fn all_choices_have_completion_routed_experts(decode_json: &Value) -> bool {
    decode_json
        .get("choices")
        .and_then(|value| value.as_array())
        .map(|choices| {
            !choices.is_empty()
                && choices.iter().all(|choice| {
                    choice
                        .get("routed_experts")
                        .filter(|value| !value.is_null())
                        .is_some()
                })
        })
        .unwrap_or(false)
}

fn validate_routed_experts_payload(value: &Value, name: &str) -> Result<(), String> {
    match value {
        Value::Object(_) => validate_routed_experts_bytes(value, name),
        _ => Err(format!("{} must be a base64 routed-experts object", name)),
    }
}

fn validate_routed_experts_bytes(value: &Value, name: &str) -> Result<(), String> {
    let Some(object) = value.as_object() else {
        return Err(format!("{} must be a base64 object", name));
    };

    let encoding = object
        .get("encoding")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{} encoding must be a string", name))?;
    if encoding != "base64" {
        return Err(format!("{} encoding must be base64", name));
    }

    let dtype = object
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{} dtype must be a string", name))?;
    if dtype != "int16" {
        return Err(format!("{} dtype must be int16", name));
    }

    let shape = object
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{} shape must be a JSON array", name))?;
    if shape.len() != 3 {
        return Err(format!("{} shape must have 3 dimensions", name));
    }
    let num_values = shape.iter().try_fold(1_u64, |acc, dim| {
        let dim = dim
            .as_u64()
            .ok_or_else(|| format!("{} shape entries must be non-negative integers", name))?;
        acc.checked_mul(dim)
            .ok_or_else(|| format!("{} shape is too large", name))
    })?;

    let data = object
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{} data must be a string", name))?;
    validate_base64_data(data, name)?;

    let decoded_len = base64_decoded_len(data, name)?;
    let expected_len = num_values
        .checked_mul(2)
        .ok_or_else(|| format!("{} byte length is too large", name))?;
    if decoded_len != expected_len {
        return Err(format!(
            "{} data byte length {} does not match shape byte length {}",
            name, decoded_len, expected_len
        ));
    }

    Ok(())
}

fn validate_base64_data(data: &str, name: &str) -> Result<(), String> {
    if data.len() % 4 != 0 {
        return Err(format!("{} data must have a valid base64 length", name));
    }
    if !data
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/' || byte == b'=')
    {
        return Err(format!("{} data must be standard base64", name));
    }

    Ok(())
}

fn base64_decoded_len(data: &str, name: &str) -> Result<u64, String> {
    let padding = data
        .as_bytes()
        .iter()
        .rev()
        .take_while(|&&byte| byte == b'=')
        .count();
    if padding > 2 {
        return Err(format!("{} data has invalid base64 padding", name));
    }

    Ok((data.len() as u64 / 4) * 3 - padding as u64)
}

fn routed_experts_is_empty(value: &Value) -> bool {
    match value {
        Value::Object(object) => object
            .get("shape")
            .and_then(Value::as_array)
            .and_then(|shape| shape.first())
            .and_then(Value::as_u64)
            .map(|tokens| tokens == 0)
            .unwrap_or(false),
        _ => false,
    }
}

fn routed_experts_sum_nonzero(value: &Value) -> bool {
    match value {
        Value::Object(object) => object
            .get("data")
            .and_then(Value::as_str)
            .map(|data| data.bytes().any(|byte| byte != b'A' && byte != b'='))
            .unwrap_or(false),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn merges_prefill_prompt_routing_and_keeps_decode_choice_routing() {
        let prefill = json!({
            "prompt_routed_experts": {
                "encoding": "base64",
                "dtype": "int16",
                "shape": [1, 1, 2],
                "data": "AQACAA=="
            }
        });
        let mut decode = json!({
            "choices": [{
                "routed_experts": {
                    "encoding": "base64",
                    "dtype": "int16",
                    "shape": [1, 1, 2],
                    "data": "AwAEAA=="
                }
            }]
        });

        let merged = merge_routed_experts_in_json(&prefill, &mut decode).unwrap();

        assert!(merged);
        assert_eq!(
            decode["prompt_routed_experts"],
            prefill["prompt_routed_experts"]
        );
        assert_eq!(decode["choices"][0]["routed_experts"]["data"], "AwAEAA==");
    }

    #[test]
    fn errors_when_decode_has_routing_but_prefill_does_not() {
        let prefill = json!({});
        let mut decode = json!({
            "choices": [{
                "routed_experts": {
                    "encoding": "base64",
                    "dtype": "int16",
                    "shape": [1, 1, 2],
                    "data": "AwAEAA=="
                }
            }]
        });

        let err = merge_routed_experts_in_json(&prefill, &mut decode).unwrap_err();

        assert!(err.contains("prefill response did not contain prompt_routed_experts"));
    }

    #[test]
    fn errors_on_zero_sum_prompt_routing() {
        let prefill = json!({
            "prompt_routed_experts": {
                "encoding": "base64",
                "dtype": "int16",
                "shape": [1, 1, 2],
                "data": "AAAAAA=="
            }
        });
        let mut decode = json!({
            "choices": [{
                "routed_experts": {
                    "encoding": "base64",
                    "dtype": "int16",
                    "shape": [1, 1, 2],
                    "data": "AwAEAA=="
                }
            }]
        });

        let err = merge_routed_experts_in_json(&prefill, &mut decode).unwrap_err();

        assert!(err.contains("sum to zero"));
    }

    #[test]
    fn errors_when_prefill_has_routing_but_decode_choice_does_not() {
        let prefill = json!({
            "prompt_routed_experts": {
                "encoding": "base64",
                "dtype": "int16",
                "shape": [1, 1, 2],
                "data": "AQACAA=="
            }
        });
        let mut decode = json!({"choices": [{"text": "ok"}]});

        let err = merge_routed_experts_in_json(&prefill, &mut decode).unwrap_err();

        assert!(err.contains("decode response did not contain choice routed_experts"));
    }

    #[test]
    fn does_nothing_for_non_routed_response() {
        let prefill = json!({});
        let mut decode = json!({"choices": [{"text": "ok"}]});

        let merged = merge_routed_experts_in_json(&prefill, &mut decode).unwrap();

        assert!(!merged);
        assert!(decode.get("prompt_routed_experts").is_none());
    }

    #[test]
    fn errors_on_zero_sum_base64_prompt_routing() {
        let prefill = json!({
            "prompt_routed_experts": {
                "encoding": "base64",
                "dtype": "int16",
                "shape": [1, 1, 2],
                "data": "AAAAAA=="
            }
        });
        let mut decode = json!({
            "choices": [{
                "routed_experts": {
                    "encoding": "base64",
                    "dtype": "int16",
                    "shape": [1, 1, 2],
                    "data": "AwAEAA=="
                }
            }]
        });

        let err = merge_routed_experts_in_json(&prefill, &mut decode).unwrap_err();

        assert!(err.contains("sum to zero"));
    }
}
