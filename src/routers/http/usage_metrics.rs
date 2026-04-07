//! Helpers for extracting token usage from upstream responses and recording
//! per-run billing metrics. Shared between the standard HTTP router and the
//! PD router so that all routing paths emit the same metrics.

use crate::metrics::RouterMetrics;

#[derive(serde::Deserialize)]
struct UsageOnly {
    usage: Option<Usage>,
}

#[derive(serde::Deserialize)]
struct Usage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

/// Parse the `usage` field from a JSON response body and record per-run
/// token metrics. Works for both non-streaming (full body) and streaming
/// (SSE chunk) responses. Does *not* increment the request counter — call
/// [`record_run_request`] separately, exactly once per successful request.
pub(crate) fn extract_and_record_usage(run_id: &str, body: &[u8]) {
    if let Ok(parsed) = serde_json::from_slice::<UsageOnly>(body) {
        if let Some(usage) = parsed.usage {
            RouterMetrics::record_run_usage(
                run_id,
                usage.prompt_tokens.unwrap_or(0),
                usage.completion_tokens.unwrap_or(0),
            );
        }
    }
}

/// Scan an SSE chunk for usage data and record per-run metrics.
/// SSE chunks contain lines like `data: {...}`. We look for lines that
/// contain a `"usage"` field with non-null token counts.
pub(crate) fn extract_usage_from_sse_chunk(run_id: &str, bytes: &[u8]) {
    // Quick check: skip chunks that don't contain usage data
    if !bytes.windows(7).any(|w| w == b"\"usage\"") {
        return;
    }
    // Parse each SSE data line
    for line in bytes.split(|&b| b == b'\n') {
        if let Some(json_data) = line.strip_prefix(b"data: ") {
            if json_data == b"[DONE]" {
                continue;
            }
            extract_and_record_usage(run_id, json_data);
        }
    }
}

/// Record that a successful request was processed for the given run.
/// Call this exactly once per successful request, regardless of whether
/// the response body contained a usage block.
pub(crate) fn record_run_request(run_id: &str) {
    RouterMetrics::record_run_request(run_id);
}
