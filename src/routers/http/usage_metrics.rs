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
    // vLLM reports the cached-prefix portion of `prompt_tokens` here when
    // prefix caching is enabled. Older engines and non-vLLM upstreams omit
    // the field entirely, so it must remain optional and default to zero.
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(serde::Deserialize)]
struct PromptTokensDetails {
    cached_tokens: Option<u64>,
}

/// Parse the `usage` field from a JSON response body and record per-run
/// token metrics. Works for both non-streaming (full body) and streaming
/// (SSE chunk) responses. Does *not* increment the request counter — call
/// [`record_run_request`] separately, exactly once per successful request.
pub(crate) fn extract_and_record_usage(run_id: &str, body: &[u8]) {
    if let Ok(parsed) = serde_json::from_slice::<UsageOnly>(body) {
        if let Some(usage) = parsed.usage {
            let cached = usage
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0);
            RouterMetrics::record_run_usage(
                run_id,
                usage.prompt_tokens.unwrap_or(0),
                usage.completion_tokens.unwrap_or(0),
                cached,
            );
        }
    }
}

/// Stateful per-stream SSE usage extractor.
///
/// `reqwest::bytes_stream()` yields chunks at arbitrary TCP segment
/// boundaries, so a single SSE `data: {...}\n` line can be split across
/// two chunks. A stateless per-chunk parser silently drops the usage
/// object in that case, under-counting billing metrics. This extractor
/// buffers partial lines across chunks and only parses complete lines.
pub(crate) struct SseUsageExtractor {
    run_id: String,
    buf: Vec<u8>,
}

/// Cap on the unparsed line buffer. SSE events are normally <<1 KiB; this
/// only kicks in for malformed/newline-less streams to prevent unbounded
/// memory growth. A single oversized event will be dropped (along with any
/// usage it contains), which is the same outcome as the previous stateless
/// parser would have produced for that event.
const MAX_LINE_BUFFER: usize = 1024 * 1024;

impl SseUsageExtractor {
    pub(crate) fn new(run_id: String) -> Self {
        Self {
            run_id,
            buf: Vec::new(),
        }
    }

    /// Feed the next chunk of an SSE response into the extractor.
    /// Complete `data:` lines that contain a `"usage"` field will be
    /// parsed and recorded as per-run token metrics. Partial trailing
    /// lines are retained until the rest of the line arrives.
    pub(crate) fn push_chunk(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);

        // Process every complete line (everything up to the last '\n').
        // Anything after the last '\n' is a partial line and stays in the
        // buffer until the next chunk arrives.
        let last_nl = match self.buf.iter().rposition(|&b| b == b'\n') {
            Some(pos) => pos,
            None => {
                // No complete line yet. Bound buffer growth and bail.
                if self.buf.len() > MAX_LINE_BUFFER {
                    self.buf.clear();
                }
                return;
            }
        };

        // Cheap pre-filter: if the complete prefix doesn't even mention
        // "usage", skip the per-line parse work entirely.
        let complete = &self.buf[..=last_nl];
        if complete.windows(7).any(|w| w == b"\"usage\"") {
            for line in complete.split(|&b| b == b'\n') {
                // SSE lines may be CRLF-terminated; strip any trailing '\r'.
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                if let Some(json_data) = line.strip_prefix(b"data: ") {
                    if json_data == b"[DONE]" {
                        continue;
                    }
                    extract_and_record_usage(&self.run_id, json_data);
                }
            }
        }

        // Drop the bytes we just processed; retain the partial trailing line.
        self.buf.drain(..=last_nl);

        if self.buf.len() > MAX_LINE_BUFFER {
            self.buf.clear();
        }
    }
}

/// Record that a successful request was processed for the given run.
/// Call this exactly once per successful request, regardless of whether
/// the response body contained a usage block.
pub(crate) fn record_run_request(run_id: &str) {
    RouterMetrics::record_run_request(run_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the extractor with arbitrary chunk slicing and return the
    /// concatenated buffer state for inspection. Used by the chunk-split
    /// regression test below.
    fn feed(extractor: &mut SseUsageExtractor, chunks: &[&[u8]]) {
        for c in chunks {
            extractor.push_chunk(c);
        }
    }

    #[test]
    fn extracts_usage_when_line_is_split_across_chunks() {
        // A real `data: {...}` SSE line containing a usage block, sliced
        // mid-JSON across two TCP chunks. The previous stateless parser
        // dropped this entirely; the buffered extractor must accept it.
        let line =
            b"data: {\"id\":\"1\",\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n";
        let split = 25; // arbitrary mid-line split
        let (a, b) = line.split_at(split);

        let mut extractor = SseUsageExtractor::new("run-split".to_string());
        feed(&mut extractor, &[a, b]);

        // After the second chunk, the line is complete and the buffer
        // should be drained back to empty.
        assert!(extractor.buf.is_empty());
    }

    #[test]
    fn handles_crlf_line_endings() {
        let line = b"data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\r\n";
        let mut extractor = SseUsageExtractor::new("run-crlf".to_string());
        extractor.push_chunk(line);
        assert!(extractor.buf.is_empty());
    }

    #[test]
    fn ignores_done_marker() {
        let mut extractor = SseUsageExtractor::new("run-done".to_string());
        extractor.push_chunk(b"data: [DONE]\n");
        assert!(extractor.buf.is_empty());
    }

    #[test]
    fn buffers_partial_trailing_line() {
        let mut extractor = SseUsageExtractor::new("run-partial".to_string());
        extractor.push_chunk(b"data: {\"usage\":{\"prompt_tokens\":1");
        // No newline yet — the partial line must be retained.
        assert!(!extractor.buf.is_empty());
        extractor.push_chunk(b",\"completion_tokens\":2}}\n");
        assert!(extractor.buf.is_empty());
    }

    #[test]
    fn bounds_unbounded_buffer_growth() {
        let mut extractor = SseUsageExtractor::new("run-overflow".to_string());
        // Feed >MAX_LINE_BUFFER bytes with no newline. The buffer must
        // not grow without bound.
        let big = vec![b'x'; MAX_LINE_BUFFER + 16];
        extractor.push_chunk(&big);
        assert!(extractor.buf.len() <= MAX_LINE_BUFFER);
    }

    #[test]
    fn parses_cached_prompt_tokens_when_present() {
        let body = br#"{"usage":{"prompt_tokens":100,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":40}}}"#;
        let parsed: UsageOnly = serde_json::from_slice(body).expect("parse usage block");
        let usage = parsed.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(50));
        let cached = usage
            .prompt_tokens_details
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0);
        assert_eq!(cached, 40);
    }

    #[test]
    fn cached_prompt_tokens_defaults_to_zero_when_absent() {
        // Older vLLM versions and non-vLLM upstreams omit
        // prompt_tokens_details entirely. Must parse cleanly and yield 0.
        let body = br#"{"usage":{"prompt_tokens":12,"completion_tokens":3}}"#;
        let parsed: UsageOnly = serde_json::from_slice(body).expect("parse usage block");
        let usage = parsed.usage.expect("usage present");
        let cached = usage
            .prompt_tokens_details
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0);
        assert_eq!(cached, 0);
    }

    #[test]
    fn cached_prompt_tokens_field_can_be_missing_within_details() {
        // Some engines emit prompt_tokens_details for reasoning metadata
        // without populating cached_tokens. Treat the missing field as 0.
        let body =
            br#"{"usage":{"prompt_tokens":5,"completion_tokens":1,"prompt_tokens_details":{}}}"#;
        let parsed: UsageOnly = serde_json::from_slice(body).expect("parse usage block");
        let usage = parsed.usage.expect("usage present");
        let cached = usage
            .prompt_tokens_details
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0);
        assert_eq!(cached, 0);
    }
}
