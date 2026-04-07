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
