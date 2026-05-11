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

    validate_routed_experts_array(prefill_prompt_routed, "prefill prompt_routed_experts")?;
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
        validate_routed_experts_array(routed_experts, "decode choice routed_experts")?;
        if let Some(entries) = routed_experts.as_array() {
            if !entries.is_empty() && !routed_experts_sum_nonzero(routed_experts) {
                return Err("decode choice routed_experts sum to zero".to_string());
            }
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

fn validate_routed_experts_array(value: &Value, name: &str) -> Result<(), String> {
    let Some(tokens) = value.as_array() else {
        return Err(format!("{} must be a JSON array", name));
    };

    for token in tokens {
        let Some(layers) = token.as_array() else {
            return Err(format!("{} token entry must be a JSON array", name));
        };
        for layer in layers {
            let Some(experts) = layer.as_array() else {
                return Err(format!("{} layer entry must be a JSON array", name));
            };
            for expert in experts {
                if expert.as_i64().is_none() {
                    return Err(format!("{} expert id must be an integer", name));
                }
            }
        }
    }

    Ok(())
}

fn routed_experts_sum_nonzero(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.as_i64().unwrap_or(0) != 0,
        Value::Array(array) => array.iter().any(routed_experts_sum_nonzero),
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
            "prompt_routed_experts": [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ],
            "choices": [{"routed_experts": [[[99]]]}]
        });
        let mut decode = json!({
            "choices": [{
                "routed_experts": [
                    [[11, 12], [13, 14]]
                ]
            }]
        });

        let merged = merge_routed_experts_in_json(&prefill, &mut decode).unwrap();

        assert!(merged);
        assert_eq!(
            decode["prompt_routed_experts"],
            prefill["prompt_routed_experts"]
        );
        assert_eq!(decode["choices"][0]["routed_experts"][0][0][0], 11);
    }

    #[test]
    fn errors_when_decode_has_routing_but_prefill_does_not() {
        let prefill = json!({});
        let mut decode = json!({"choices": [{"routed_experts": [[[3]]]}]});

        let err = merge_routed_experts_in_json(&prefill, &mut decode).unwrap_err();

        assert!(err.contains("prefill response did not contain prompt_routed_experts"));
    }

    #[test]
    fn errors_on_zero_sum_prompt_routing() {
        let prefill = json!({
            "prompt_routed_experts": [
                [[0]],
                [[0]]
            ]
        });
        let mut decode = json!({"choices": [{"routed_experts": [[[3]]]}]});

        let err = merge_routed_experts_in_json(&prefill, &mut decode).unwrap_err();

        assert!(err.contains("sum to zero"));
    }

    #[test]
    fn errors_when_prefill_has_routing_but_decode_choice_does_not() {
        let prefill = json!({
            "prompt_routed_experts": [
                [[1]],
                [[2]]
            ]
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
}
