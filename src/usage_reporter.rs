//! Background usage reporter that periodically flushes accumulated per-run
//! inference token counts to the platform billing API.
//!
//! The reporter accumulates prompt/completion tokens per `run_id` and flushes
//! them every `flush_interval` seconds via `POST /internal/rft/usage`.
//!
//! A global instance is set once at startup and accessed from anywhere via
//! `UsageReporter::global()`, similar to how the metrics system works.

use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::sync::Notify;
use tracing::{error, info, warn};

/// Global singleton for the usage reporter.
static GLOBAL_REPORTER: OnceLock<UsageReporter> = OnceLock::new();

/// Accumulated token counts for a single run.
#[derive(Default)]
struct RunUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// Payload sent to `POST /internal/rft/usage`.
#[derive(Serialize)]
struct UsagePayload {
    run_id: String,
    step: u64,
    usage_type: &'static str,
    tokens: u64,
    input_tokens: u64,
    output_tokens: u64,
}

/// Configuration for the usage reporter.
#[derive(Clone)]
pub struct UsageReporterConfig {
    /// Platform usage endpoint, e.g. `https://api.primeintellect.ai/api/internal/rft/usage`
    pub url: String,
    /// API key sent as `X-Api-Key` header (the internal RFT key).
    pub api_key: String,
    /// How often to flush accumulated usage (default: 30s).
    pub flush_interval: Duration,
}

/// Handle to the background usage reporter.
///
/// Cloning is cheap (inner state is `Arc`).
#[derive(Clone)]
pub struct UsageReporter {
    inner: Arc<Inner>,
}

struct Inner {
    /// Per-run accumulated tokens since last flush.
    pending: Mutex<HashMap<String, RunUsage>>,
    /// Monotonically increasing batch counter used as the `step` field so
    /// each report is unique for the platform's idempotency key.
    batch_seq: AtomicU64,
    /// Notify the flush task to wake up (used for graceful shutdown).
    notify: Notify,
    config: UsageReporterConfig,
    client: Client,
}

impl UsageReporter {
    /// Set the global usage reporter instance. Called once at startup.
    pub fn set_global(reporter: UsageReporter) {
        let _ = GLOBAL_REPORTER.set(reporter);
    }

    /// Get the global reporter, if configured.
    pub fn global() -> Option<&'static UsageReporter> {
        GLOBAL_REPORTER.get()
    }

    /// Record usage on the global reporter (no-op if not configured).
    pub fn record_global(run_id: &str, prompt_tokens: u64, completion_tokens: u64) {
        if let Some(reporter) = Self::global() {
            reporter.record(run_id, prompt_tokens, completion_tokens);
        }
    }

    /// Create a new reporter and spawn the background flush task.
    pub fn new(config: UsageReporterConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create usage reporter HTTP client");

        let reporter = Self {
            inner: Arc::new(Inner {
                pending: Mutex::new(HashMap::new()),
                batch_seq: AtomicU64::new(0),
                notify: Notify::new(),
                config: config.clone(),
                client,
            }),
        };

        // Spawn background flush loop
        let r = reporter.clone();
        tokio::spawn(async move {
            r.flush_loop().await;
        });

        info!(
            "Usage reporter started (flush interval: {:?}, url: {})",
            config.flush_interval, config.url
        );

        reporter
    }

    /// Record inference token usage for a run. This is lock-free on the hot
    /// path (just increments counters in a mutex-guarded hashmap).
    pub fn record(&self, run_id: &str, prompt_tokens: u64, completion_tokens: u64) {
        let mut pending = self.inner.pending.lock().unwrap();
        let entry = pending.entry(run_id.to_string()).or_default();
        entry.prompt_tokens += prompt_tokens;
        entry.completion_tokens += completion_tokens;
    }

    /// Background loop that periodically flushes accumulated usage.
    async fn flush_loop(&self) {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(self.inner.config.flush_interval) => {},
                _ = self.inner.notify.notified() => {
                    // Shutdown signal — do one final flush then exit
                    self.flush().await;
                    return;
                }
            }
            self.flush().await;
        }
    }

    /// Drain the pending map and POST each run's usage to the platform.
    async fn flush(&self) {
        let entries: HashMap<String, RunUsage> = {
            let mut pending = self.inner.pending.lock().unwrap();
            std::mem::take(&mut *pending)
        };

        if entries.is_empty() {
            return;
        }

        for (run_id, usage) in entries {
            let total = usage.prompt_tokens + usage.completion_tokens;
            if total == 0 {
                continue;
            }

            let step = self.inner.batch_seq.fetch_add(1, Ordering::Relaxed);

            let payload = UsagePayload {
                run_id: run_id.clone(),
                step,
                usage_type: "INFERENCE",
                tokens: total,
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            };

            let result = self
                .inner
                .client
                .post(&self.inner.config.url)
                .header("X-Api-Key", &self.inner.config.api_key)
                .json(&payload)
                .send()
                .await;

            match result {
                Ok(resp) if resp.status().is_success() => {
                    info!(
                        "Reported usage for run {}: {} prompt + {} completion tokens (step={})",
                        run_id, usage.prompt_tokens, usage.completion_tokens, step
                    );
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    warn!(
                        "Usage report failed for run {} (status {}): {}",
                        run_id, status, body
                    );
                }
                Err(e) => {
                    error!("Usage report request failed for run {}: {}", run_id, e);
                }
            }
        }
    }

    /// Signal the background task to do a final flush and stop.
    pub fn shutdown(&self) {
        self.inner.notify.notify_one();
    }
}
