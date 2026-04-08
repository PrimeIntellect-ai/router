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
    /// Check whether this run-scoped JWT is allowed to target `requested_model`.
    ///
    /// A JWT-authenticated request may target:
    ///   * the base model named in the `model` claim, or
    ///   * the run's own LoRA adapter named in the `lora` claim, or
    ///   * any LoRA whose name is `<lora>-<suffix>` (forward-compat with
    ///     step-versioned adapter names like `rft-<run_id>-step-42`).
    ///
    /// `None` / empty model means "use the worker's default", which we treat
    /// as the base model and therefore allow.
    pub fn allows_model(&self, requested_model: Option<&str>) -> bool {
        let requested = match requested_model {
            Some(m) if !m.is_empty() => m,
            _ => return true, // base model fallthrough
        };

        if self.model.as_deref() == Some(requested) {
            return true;
        }

        if let Some(lora) = self.lora.as_deref() {
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
        assert!(c.allows_model(Some("Qwen/Qwen3-4B")));
    }

    #[test]
    fn allows_own_lora() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(c.allows_model(Some("rft-abc")));
    }

    #[test]
    fn allows_step_versioned_lora() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(c.allows_model(Some("rft-abc-step-42")));
    }

    #[test]
    fn rejects_other_run_lora() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(!c.allows_model(Some("rft-xyz")));
        assert!(!c.allows_model(Some("rft-abcd"))); // not a `-` boundary
    }

    #[test]
    fn rejects_other_base_model() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(!c.allows_model(Some("meta-llama/Llama-3-8B")));
    }

    #[test]
    fn empty_model_allowed_as_base() {
        let c = claims(Some("Qwen/Qwen3-4B"), Some("rft-abc"));
        assert!(c.allows_model(None));
        assert!(c.allows_model(Some("")));
    }
}
