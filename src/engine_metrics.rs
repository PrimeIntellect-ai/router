//! Engine load snapshot for external controllers (`GET /engine_metrics`).
//!
//! Scrapes every worker's vLLM Prometheus `/metrics` (and `max_model_len` from
//! `/v1/models`) on request and returns them as JSON next to the router's own
//! per-worker in-flight counts, the worker roles, and the depth of the router's
//! request queue. Values are raw: counters stay cumulative, so consumers derive
//! rates from consecutive snapshots.

use crate::core::{Worker, WorkerType};
use futures::future::join_all;
use reqwest::Client;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

const METRIC_PREFIX: &str = "vllm:";
const CACHE_CONFIG_FAMILY: &str = "vllm:cache_config_info";
const WAITING_BY_REASON: &str = "num_requests_waiting_by_reason";
const FETCH_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Default, PartialEq, Serialize)]
pub struct HistogramSnapshot {
    pub sum: f64,
    pub count: f64,
    /// Upper bound (`le` label, as exposed) -> cumulative count.
    pub buckets: BTreeMap<String, f64>,
}

/// All `vllm:*` samples of one engine (one `engine` label), keyed by sample
/// name without the `vllm:` prefix and summed over all other labels.
#[derive(Debug, Default, PartialEq, Serialize)]
pub struct EngineSnapshot {
    pub gauges: BTreeMap<String, f64>,
    pub counters: BTreeMap<String, f64>,
    pub histograms: BTreeMap<String, HistogramSnapshot>,
    /// Labels of `vllm:cache_config_info` (e.g. `num_gpu_blocks`, `block_size`).
    pub cache_config: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct WorkerMetrics {
    pub url: String,
    pub role: &'static str,
    pub healthy: bool,
    /// Requests the router has in flight to this worker (summed over DP ranks).
    pub router_inflight: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<u64>,
    /// Engine label (`engine`, one per DP rank behind the URL) -> samples.
    pub engines: BTreeMap<String, EngineSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EngineMetrics {
    /// Requests waiting in the router's concurrency queue, not yet sent to any worker.
    pub router_queued: usize,
    pub workers: Vec<WorkerMetrics>,
}

pub fn role(worker_type: &WorkerType) -> &'static str {
    match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    }
}

/// Parse a Prometheus text exposition into one snapshot per `engine` label.
pub fn parse_prometheus_text(text: &str) -> BTreeMap<String, EngineSnapshot> {
    let mut types: HashMap<&str, &str> = HashMap::new();
    let mut engines: BTreeMap<String, EngineSnapshot> = BTreeMap::new();
    for line in text.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let mut parts = rest.split_whitespace();
            if let (Some(name), Some(kind)) = (parts.next(), parts.next()) {
                types.insert(name, kind);
            }
            continue;
        }
        if line.is_empty() || line.starts_with('#') || !line.starts_with(METRIC_PREFIX) {
            continue;
        }
        let Some((name, labels, value)) = parse_sample(line) else {
            continue;
        };
        if name.ends_with("_created") || !value.is_finite() {
            continue;
        }
        let Some((family, kind)) = family_of(&types, name) else {
            continue;
        };
        let engine_label = labels
            .iter()
            .find(|(key, _)| key == "engine")
            .map_or_else(|| "0".to_string(), |(_, value)| value.clone());
        let engine = engines.entry(engine_label).or_default();
        if family == CACHE_CONFIG_FAMILY {
            engine.cache_config.extend(labels);
            continue;
        }
        let mut short = name.trim_start_matches(METRIC_PREFIX).to_string();
        // Keep the queue-reason breakdown instead of summing it away
        if short == WAITING_BY_REASON {
            let reason = labels
                .iter()
                .find(|(key, _)| key == "reason")
                .map_or("unknown", |(_, value)| value.as_str());
            short = format!("num_requests_waiting_reason_{reason}");
        }
        match kind {
            "gauge" => *engine.gauges.entry(short).or_default() += value,
            "counter" => *engine.counters.entry(short).or_default() += value,
            "histogram" => {
                let histogram = engine
                    .histograms
                    .entry(family.trim_start_matches(METRIC_PREFIX).to_string())
                    .or_default();
                if name.ends_with("_sum") {
                    histogram.sum += value;
                } else if name.ends_with("_count") {
                    histogram.count += value;
                } else if name.ends_with("_bucket") {
                    if let Some((_, le)) = labels.iter().find(|(key, _)| key == "le") {
                        *histogram.buckets.entry(le.clone()).or_default() += value;
                    }
                }
            }
            _ => {}
        }
    }
    engines
}

/// The `# TYPE` family a sample belongs to (counters expose `<family>_total`,
/// histograms `<family>_bucket|_sum|_count`); only counters, gauges, and
/// histograms are kept.
fn family_of<'a>(types: &HashMap<&'a str, &'a str>, name: &str) -> Option<(&'a str, &'a str)> {
    let candidates = std::iter::once(name).chain(
        ["_total", "_bucket", "_sum", "_count"]
            .iter()
            .filter_map(|suffix| name.strip_suffix(suffix)),
    );
    for candidate in candidates {
        if let Some((&family, &kind)) = types.get_key_value(candidate) {
            return matches!(kind, "counter" | "gauge" | "histogram").then_some((family, kind));
        }
    }
    None
}

/// Split `name{k="v",...} value [timestamp]` into its parts.
type Labels = Vec<(String, String)>;

fn parse_sample(line: &str) -> Option<(&str, Labels, f64)> {
    let name_end = line.find(|c: char| c == '{' || c.is_whitespace())?;
    let name = &line[..name_end];
    let mut rest = &line[name_end..];
    let mut labels = Vec::new();
    if let Some(body) = rest.strip_prefix('{') {
        let (parsed, after) = parse_labels(body)?;
        labels = parsed;
        rest = after;
    }
    let value = parse_value(rest.split_whitespace().next()?)?;
    Some((name, labels, value))
}

/// Parse `k="v",...}` and return the labels plus the text after `}`.
fn parse_labels(body: &str) -> Option<(Labels, &str)> {
    let mut labels = Vec::new();
    let mut rest = body;
    loop {
        rest = rest.trim_start_matches([' ', ',']);
        if let Some(after) = rest.strip_prefix('}') {
            return Some((labels, after));
        }
        let eq = rest.find('=')?;
        let key = rest[..eq].trim().to_string();
        rest = rest[eq + 1..].trim_start().strip_prefix('"')?;
        let mut value = String::new();
        let mut chars = rest.char_indices();
        let end = loop {
            match chars.next()? {
                (i, '"') => break i,
                (_, '\\') => match chars.next()?.1 {
                    'n' => value.push('\n'),
                    other => value.push(other),
                },
                (_, c) => value.push(c),
            }
        };
        labels.push((key, value));
        rest = &rest[end + 1..];
    }
}

fn parse_value(token: &str) -> Option<f64> {
    match token {
        "+Inf" | "Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        "NaN" => Some(f64::NAN),
        _ => token.parse().ok(),
    }
}

async fn fetch_text(
    client: &Client,
    url: &str,
    authorization: Option<&http::HeaderValue>,
) -> Result<String, String> {
    let mut request = client.get(url).timeout(FETCH_TIMEOUT);
    if let Some(value) = authorization {
        request = request.header(http::header::AUTHORIZATION, value);
    }
    let response = request.send().await.map_err(|e| e.to_string())?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("{url} returned {status}"));
    }
    response.text().await.map_err(|e| e.to_string())
}

fn max_model_len(models: &str) -> Option<u64> {
    let models: serde_json::Value = serde_json::from_str(models).ok()?;
    models
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|card| card.get("max_model_len")?.as_u64())
        .max()
}

async fn collect_worker(
    client: &Client,
    group: &[Arc<dyn Worker>],
    authorization: Option<&http::HeaderValue>,
) -> WorkerMetrics {
    let first = &group[0];
    let base_url = first.base_url().trim_end_matches('/');
    let metrics_url = format!("{base_url}/metrics");
    let models_url = format!("{base_url}/v1/models");
    let (metrics, models) = tokio::join!(
        fetch_text(client, &metrics_url, authorization),
        fetch_text(client, &models_url, authorization),
    );
    let (engines, error) = match metrics {
        Ok(text) => (parse_prometheus_text(&text), None),
        Err(e) => (BTreeMap::new(), Some(e)),
    };
    WorkerMetrics {
        url: base_url.to_string(),
        role: role(&first.worker_type()),
        healthy: group.iter().all(|worker| worker.is_healthy()),
        router_inflight: group.iter().map(|worker| worker.load()).sum(),
        max_model_len: models.ok().as_deref().and_then(max_model_len),
        engines,
        error,
    }
}

/// Scrape all workers concurrently; DP-aware workers sharing a base URL are one engine process and are fetched once.
pub async fn collect(
    client: &Client,
    workers: Vec<Arc<dyn Worker>>,
    authorization: Option<&http::HeaderValue>,
    router_queued: usize,
) -> EngineMetrics {
    let mut groups: BTreeMap<(String, &'static str), Vec<Arc<dyn Worker>>> = BTreeMap::new();
    for worker in workers {
        let key = (worker.base_url().to_string(), role(&worker.worker_type()));
        groups.entry(key).or_default().push(worker);
    }
    let workers = join_all(
        groups
            .values()
            .map(|group| collect_worker(client, group, authorization)),
    )
    .await;
    EngineMetrics {
        router_queued,
        workers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPOSITION: &str = r#"# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="1",model_name="m"} 5.0
# TYPE vllm:num_requests_waiting_by_reason gauge
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 2.0
vllm:num_requests_waiting_by_reason{engine="0",reason="deferred"} 1.0
# TYPE vllm:request_success counter
vllm:request_success_total{engine="0",finished_reason="stop"} 10.0
vllm:request_success_total{engine="0",finished_reason="length"} 4.0
vllm:request_success_created{engine="0",finished_reason="stop"} 1.7e9
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0"} 3.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf"} 14.0
vllm:e2e_request_latency_seconds_sum{engine="0"} 42.5
vllm:e2e_request_latency_seconds_count{engine="0"} 14.0
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="1000"} 1.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="a \"quoted\", label"} NaN
# TYPE python_gc_objects_collected counter
python_gc_objects_collected_total{generation="0"} 7.0
"#;

    #[test]
    fn parses_and_sums_per_engine() {
        let engines = parse_prometheus_text(EXPOSITION);
        assert_eq!(engines.len(), 2);

        let engine = &engines["0"];
        assert_eq!(engine.gauges["num_requests_running"], 3.0);
        assert_eq!(engine.gauges["num_requests_waiting_reason_capacity"], 2.0);
        assert_eq!(engine.gauges["num_requests_waiting_reason_deferred"], 1.0);
        assert!(!engine.gauges.contains_key("kv_cache_usage_perc"));
        assert_eq!(engine.counters["request_success_total"], 14.0);
        assert_eq!(engine.counters.len(), 1);

        let histogram = &engine.histograms["e2e_request_latency_seconds"];
        assert_eq!(histogram.sum, 42.5);
        assert_eq!(histogram.count, 14.0);
        assert_eq!(histogram.buckets["1.0"], 3.0);
        assert_eq!(histogram.buckets["+Inf"], 14.0);

        assert_eq!(engine.cache_config["num_gpu_blocks"], "1000");
        assert_eq!(engine.cache_config["block_size"], "16");

        assert_eq!(engines["1"].gauges["num_requests_running"], 5.0);
    }

    #[test]
    fn missing_engine_label_defaults_to_zero() {
        let text = "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 4\n";
        assert_eq!(
            parse_prometheus_text(text)["0"].gauges["num_requests_waiting"],
            4.0
        );
    }

    #[test]
    fn parses_escaped_label_values() {
        let (name, labels, value) =
            parse_sample(r#"vllm:x{a="q\"uo,te}",b="back\\slash"} 1.5 1700000000"#).unwrap();
        assert_eq!(name, "vllm:x");
        assert_eq!(labels[0], ("a".to_string(), "q\"uo,te}".to_string()));
        assert_eq!(labels[1], ("b".to_string(), "back\\slash".to_string()));
        assert_eq!(value, 1.5);
    }

    #[test]
    fn reads_max_model_len() {
        let models = r#"{"data":[{"id":"m","max_model_len":32768},{"id":"lora"}]}"#;
        assert_eq!(max_model_len(models), Some(32768));
        assert_eq!(max_model_len("not json"), None);
    }
}
