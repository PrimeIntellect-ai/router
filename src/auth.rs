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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claims(model: Option<&str>, lora: Option<&str>) -> RftClaims {
        RftClaims {
            sub: "user".into(),
            run_id: "abc".into(),
            team_id: String::new(),
            model: model.map(String::from),
            lora: lora.map(String::from),
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
}
