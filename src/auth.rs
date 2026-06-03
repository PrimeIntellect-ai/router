use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use serde::Deserialize;

/// Claims extracted from RFT inference JWTs signed by the platform.
#[derive(Debug, Clone, Deserialize)]
pub struct RftClaims {
    /// User ID (subject)
    pub sub: String,
    /// RFT run ID
    pub run_id: String,
    /// Team ID
    #[serde(default)]
    pub team_id: String,
    /// Base model name
    #[serde(default)]
    pub model: Option<String>,
    /// Allowed LoRA adapter name
    #[serde(default)]
    pub lora: Option<String>,
    /// Alternate names that should resolve to the base `model`. Used when
    /// the platform exposes a user-facing model identifier (e.g.
    /// `sprints/Llama-3.2-1B-Instruct`) that differs from the canonical
    /// HF path vLLM serves (`meta-llama/Llama-3.2-1B-Instruct`). A
    /// request hitting an alias is authorized and the request body's
    /// `model` field is rewritten to `self.model` before dispatch.
    #[serde(default)]
    pub model_aliases: Vec<String>,
}

/// Verifier for RS256 JWTs signed by the platform.
pub struct JwtVerifier {
    decoding_key: DecodingKey,
    validation: Validation,
}

impl JwtVerifier {
    /// Create a new verifier from a PEM-encoded RSA public key.
    pub fn new(pem: &str) -> Result<Self, String> {
        let decoding_key = DecodingKey::from_rsa_pem(pem.as_bytes())
            .map_err(|e| format!("Invalid RSA public key: {e}"))?;
        let mut validation = Validation::new(Algorithm::RS256);
        // Allow 60 seconds of clock skew between platform and router
        validation.leeway = 60;
        // We don't validate audience
        validation.validate_aud = false;
        Ok(Self {
            decoding_key,
            validation,
        })
    }

    /// Verify a JWT and return the extracted claims.
    pub fn verify(&self, token: &str) -> Result<RftClaims, jsonwebtoken::errors::Error> {
        let data = decode::<RftClaims>(token, &self.decoding_key, &self.validation)?;
        Ok(data.claims)
    }
}

impl RftClaims {
    /// Check whether this run-scoped JWT is allowed to target `requested`.
    ///
    /// A JWT-authenticated request may target:
    ///   * the base model named in the `model` claim, or
    ///   * the run's own LoRA adapter named in the `lora` claim, or
    ///   * any LoRA whose name is `<lora>-<suffix>` (forward-compat with
    ///     step-versioned adapter names like `rft-<run_id>-step-42`).
    ///
    /// Empty / missing requested name returns `false` — the caller must
    /// resolve the request to a concrete model name (typically the JWT's
    /// `model` claim) before calling. This is intentional: in multi-model
    /// deployments a `None` model would otherwise route to an arbitrary
    /// worker outside the JWT's scope.
    pub fn allows_model(&self, requested: &str) -> bool {
        if requested.is_empty() {
            return false;
        }

        if let Some(base) = self.model.as_deref() {
            if !base.is_empty() && base == requested {
                return true;
            }
        }

        if self.is_model_alias(requested) {
            return true;
        }

        if let Some(lora) = self.lora.as_deref() {
            // Empty lora claim must never authorize anything; an empty
            // string is a prefix of every other string, which would let
            // `requested == "-anything"` slip through the prefix branch
            // below.
            if lora.is_empty() {
                return false;
            }
            if requested == lora {
                return true;
            }
            // Prefix match so that step-versioned aliases like
            // `rft-<run_id>-step-42` remain authorized by the same JWT.
            // The `run_id` portion is the unguessable security boundary.
            if let Some(rest) = requested.strip_prefix(lora) {
                if rest.starts_with('-') {
                    return true;
                }
            }
        }

        false
    }

    /// Whether `requested` is one of the alternate names declared in
    /// `model_aliases`. Empty entries are ignored so a misconfigured
    /// claim never authorizes the empty model.
    fn is_model_alias(&self, requested: &str) -> bool {
        if requested.is_empty() {
            return false;
        }
        self.model_aliases
            .iter()
            .any(|a| !a.is_empty() && a == requested)
    }

    /// If `requested` matched the JWT only via `model_aliases`, return
    /// the canonical model name (`self.model`) so the request body can
    /// be rewritten before forwarding to vLLM. Returns `None` if the
    /// request already targets the base model or a LoRA — those need to
    /// pass through unchanged so vLLM can dispatch the LoRA adapter.
    ///
    /// LoRA-shadowing is the load-bearing case: a pathological JWT could
    /// list the same string in both `lora` (or as a `<lora>-…` step
    /// adapter) and `model_aliases`. Rewriting such a request to the
    /// base model would silently swap a LoRA call for a base-model call,
    /// so we treat LoRA matches as taking precedence over alias matches.
    pub fn canonical_for_alias(&self, requested: &str) -> Option<String> {
        if !self.is_model_alias(requested) {
            return None;
        }
        let base = self.model.as_deref()?;
        if base.is_empty() || base == requested {
            return None;
        }
        if self.matches_lora(requested) {
            return None;
        }
        Some(base.to_string())
    }

    /// Whether `requested` is authorized via the `lora` claim — either
    /// an exact match or a `<lora>-<suffix>` step adapter. Mirrors the
    /// LoRA branch of `allows_model` so `canonical_for_alias` can defer
    /// to LoRA dispatch when both branches would otherwise authorize.
    fn matches_lora(&self, requested: &str) -> bool {
        let Some(lora) = self.lora.as_deref() else {
            return false;
        };
        if lora.is_empty() {
            return false;
        }
        if requested == lora {
            return true;
        }
        if let Some(rest) = requested.strip_prefix(lora) {
            return rest.starts_with('-');
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claims(model: Option<&str>, lora: Option<&str>) -> RftClaims {
        claims_with_aliases(model, lora, &[])
    }

    fn claims_with_aliases(model: Option<&str>, lora: Option<&str>, aliases: &[&str]) -> RftClaims {
        RftClaims {
            sub: "user".into(),
            run_id: "abc".into(),
            team_id: String::new(),
            model: model.map(String::from),
            lora: lora.map(String::from),
            model_aliases: aliases.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    #[test]
    fn allows_base_model() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(c.allows_model("Qwen/Qwen3-4B"));
    }

    #[test]
    fn allows_own_lora() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(c.allows_model("rft-abc"));
    }

    #[test]
    fn allows_step_versioned_lora() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(c.allows_model("rft-abc-step-42"));
    }

    #[test]
    fn rejects_other_run_lora() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(!c.allows_model("rft-xyz"));
        assert!(!c.allows_model("rft-abcd")); // not a `-` boundary
    }

    #[test]
    fn rejects_other_base_model() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(!c.allows_model("meta-llama/Llama-3-8B"));
    }

    #[test]
    fn rejects_empty_requested() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(!c.allows_model(""));
    }

    #[test]
    fn empty_lora_does_not_authorize_anything() {
        // Regression: an empty `lora` claim used to authorize any model
        // beginning with "-" via `strip_prefix("")` returning Some(rest).
        let c = claims(Some("Qwen/Qwen3-4B"), Some(""));
        assert!(!c.allows_model("-malicious"));
        assert!(!c.allows_model("rft-abc"));
        // Base model still works.
        assert!(c.allows_model("Qwen/Qwen3-4B"));
    }

    #[test]
    fn empty_base_model_does_not_authorize_empty_request() {
        let c = claims(Some(""), Some("rft-abc"));
        assert!(!c.allows_model(""));
    }

    #[test]
    fn allows_model_alias() {
        let c = claims_with_aliases(
            Some("meta-llama/Llama-3.2-1B-Instruct"),
            Some("rft-abc"),
            &["sprints/Llama-3.2-1B-Instruct"],
        );
        assert!(c.allows_model("sprints/Llama-3.2-1B-Instruct"));
        assert!(c.allows_model("meta-llama/Llama-3.2-1B-Instruct"));
        assert!(!c.allows_model("other/model"));
    }

    #[test]
    fn canonical_for_alias_rewrites_alias_only() {
        let c = claims_with_aliases(
            Some("meta-llama/Llama-3.2-1B-Instruct"),
            Some("rft-abc"),
            &["sprints/Llama-3.2-1B-Instruct"],
        );
        assert_eq!(
            c.canonical_for_alias("sprints/Llama-3.2-1B-Instruct").as_deref(),
            Some("meta-llama/Llama-3.2-1B-Instruct"),
        );
        // Base model and lora must NOT be rewritten — lora dispatch
        // depends on the original name reaching vLLM.
        assert_eq!(c.canonical_for_alias("meta-llama/Llama-3.2-1B-Instruct"), None);
        assert_eq!(c.canonical_for_alias("rft-abc"), None);
        assert_eq!(c.canonical_for_alias("rft-abc-step-42"), None);
        assert_eq!(c.canonical_for_alias("unrelated"), None);
    }

    #[test]
    fn empty_alias_entry_authorizes_nothing() {
        let c = claims_with_aliases(Some("meta-llama/Llama-3.2-1B-Instruct"), Some("rft-abc"), &[""]);
        assert!(!c.allows_model(""));
        assert_eq!(c.canonical_for_alias(""), None);
    }

    #[test]
    fn alias_matching_base_does_not_rewrite() {
        // Pathological config: alias equals the base. Should still
        // authorize, but not produce a rewrite (would be a no-op anyway).
        let c = claims_with_aliases(Some("meta-llama/Llama-3.2-1B-Instruct"), None, &["meta-llama/Llama-3.2-1B-Instruct"]);
        assert!(c.allows_model("meta-llama/Llama-3.2-1B-Instruct"));
        assert_eq!(c.canonical_for_alias("meta-llama/Llama-3.2-1B-Instruct"), None);
    }

    #[test]
    fn alias_matching_lora_does_not_rewrite() {
        // Pathological config: an alias entry collides with the lora
        // claim. Rewriting would silently swap the LoRA call for a
        // base-model call, so LoRA matches take precedence.
        let c = claims_with_aliases(
            Some("meta-llama/Llama-3.2-1B-Instruct"),
            Some("rft-abc"),
            &["rft-abc"],
        );
        assert!(c.allows_model("rft-abc"));
        assert_eq!(c.canonical_for_alias("rft-abc"), None);
    }

    #[test]
    fn alias_matching_step_versioned_lora_does_not_rewrite() {
        let c = claims_with_aliases(
            Some("meta-llama/Llama-3.2-1B-Instruct"),
            Some("rft-abc"),
            &["rft-abc-step-42"],
        );
        assert!(c.allows_model("rft-abc-step-42"));
        assert_eq!(c.canonical_for_alias("rft-abc-step-42"), None);
    }
}
