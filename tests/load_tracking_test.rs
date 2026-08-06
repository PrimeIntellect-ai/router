//! Tests for phantom request accumulation in cache_aware load tracking.
//!
//! The cache_aware policy increments a per-worker load counter before each
//! request and decrements it when the request completes. Several code paths
//! can cause the counter to drift:
//!
//! 1. Non-streaming retryable failures (5xx): send_typed_request unconditionally
//!    decrements at line 931, AND the retry cleanup at line 624 also decrements
//!    → double decrement (counter goes too low)
//!
//! 2. Streaming retryable failures: retry cleanup decrements immediately, then
//!    the spawned streaming task also decrements when the response is dropped
//!    → double decrement (race condition)
//!
//! 3. Streaming requests where the worker is removed from the registry mid-stream:
//!    registry.get_by_url() returns None, decrement is silently skipped
//!    → phantom request (counter never decremented)

mod common;

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use common::test_app::create_test_app;
use axum::body::Body;
use http_body_util::BodyExt;
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::core::{BasicWorker, Worker, WorkerType as CoreWorkerType};
use vllm_router_rs::policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy};
use vllm_router_rs::routers::{RouterFactory, RouterTrait};

// ---------------------------------------------------------------------------
// Unit tests: verify load counter invariants directly on Worker objects
// ---------------------------------------------------------------------------

#[test]
fn test_load_counter_underflow_protection() {
    let worker = BasicWorker::new("http://w1:8000".to_string(), CoreWorkerType::Regular);
    assert_eq!(worker.load(), 0);

    worker.decrement_load();
    assert_eq!(worker.load(), 0, "load should not underflow below 0");

    worker.increment_load();
    assert_eq!(worker.load(), 1);
    worker.decrement_load();
    worker.decrement_load(); // extra decrement
    assert_eq!(worker.load(), 0, "double decrement should clamp to 0");
}

#[test]
fn test_double_decrement_causes_counter_drift() {
    // Demonstrates the double-decrement bug: if two requests are in flight
    // and one gets double-decremented, the counter drifts below the true count.
    let worker = Arc::new(BasicWorker::new(
        "http://w1:8000".to_string(),
        CoreWorkerType::Regular,
    ));

    // Simulate 3 concurrent requests
    worker.increment_load(); // request A
    worker.increment_load(); // request B
    worker.increment_load(); // request C
    assert_eq!(worker.load(), 3);

    // Request A completes normally
    worker.decrement_load();
    assert_eq!(worker.load(), 2);

    // Request B fails with retryable 500 on non-streaming path:
    // send_typed_request decrements unconditionally (line 931)
    worker.decrement_load();
    // retry cleanup ALSO decrements (line 624) - THIS IS THE BUG
    worker.decrement_load();
    assert_eq!(worker.load(), 0);

    // Request C completes normally
    worker.decrement_load();
    // Counter is already at 0, can't go lower - but the decrement for C
    // was "eaten" by the double-decrement of B
    assert_eq!(worker.load(), 0);

    // Now if we check: we had 3 requests, all completed, counter is 0.
    // Looks correct! But the intermediate state was wrong: after B's
    // double-decrement, counter was 0 with C still in-flight.
    // This means C appeared to have 0 load when it should have been 1.
}

#[test]
fn test_cache_aware_deprioritizes_high_load_worker() {
    // Verify that phantom load causes cache_aware to route AWAY from a worker.
    let config = CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 2,
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 0,
        max_tree_size: 100,
    };
    let policy = CacheAwarePolicy::with_config(config);

    let mut labels = HashMap::new();
    labels.insert("model_id".to_string(), "test-model".to_string());

    let worker1 = Arc::new(
        BasicWorker::new("http://w1:8000".to_string(), CoreWorkerType::Regular)
            .with_labels(labels.clone()),
    );
    let worker2 = Arc::new(
        BasicWorker::new("http://w2:8000".to_string(), CoreWorkerType::Regular)
            .with_labels(labels.clone()),
    );

    policy.add_worker(worker1.as_ref());
    policy.add_worker(worker2.as_ref());

    let workers: Vec<Arc<dyn Worker>> = vec![worker1.clone(), worker2.clone()];

    // Simulate phantom load accumulation on worker1 (like 250 stuck requests)
    for _ in 0..250 {
        worker1.increment_load();
    }
    assert_eq!(worker1.load(), 250);
    assert_eq!(worker2.load(), 0);

    // With massive load imbalance, cache_aware should always pick worker2
    let mut worker1_selected = 0;

    for i in 0..100 {
        let prompt = format!("test prompt number {}", i);
        match policy.select_worker(&workers, Some(&prompt)) {
            Some(0) => worker1_selected += 1,
            Some(1) => {}
            _ => panic!("unexpected worker index"),
        }
    }

    assert_eq!(
        worker1_selected, 0,
        "worker with phantom load 250 should never be selected (was selected {} times)",
        worker1_selected
    );
}

#[test]
fn test_phantom_load_concentrates_traffic_on_least_loaded() {
    // When phantom requests accumulate unevenly, traffic concentrates
    // on the worker with least phantom load.
    let config = CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 2,
        balance_rel_threshold: 1.5,
        eviction_interval_secs: 0,
        max_tree_size: 100,
    };
    let policy = CacheAwarePolicy::with_config(config);

    let mut labels = HashMap::new();
    labels.insert("model_id".to_string(), "test-model".to_string());

    let worker1 = Arc::new(
        BasicWorker::new("http://w1:8000".to_string(), CoreWorkerType::Regular)
            .with_labels(labels.clone()),
    );
    let worker2 = Arc::new(
        BasicWorker::new("http://w2:8000".to_string(), CoreWorkerType::Regular)
            .with_labels(labels.clone()),
    );
    let worker3 = Arc::new(
        BasicWorker::new("http://w3:8000".to_string(), CoreWorkerType::Regular)
            .with_labels(labels.clone()),
    );

    policy.add_worker(worker1.as_ref());
    policy.add_worker(worker2.as_ref());
    policy.add_worker(worker3.as_ref());

    let workers: Vec<Arc<dyn Worker>> = vec![worker1.clone(), worker2.clone(), worker3.clone()];

    // Simulate phantom load: w1=100, w2=95, w3=90
    for _ in 0..100 {
        worker1.increment_load();
    }
    for _ in 0..95 {
        worker2.increment_load();
    }
    for _ in 0..90 {
        worker3.increment_load();
    }

    let mut selections = [0u32; 3];
    for i in 0..100 {
        let prompt = format!("request {}", i);
        if let Some(idx) = policy.select_worker(&workers, Some(&prompt)) {
            selections[idx] += 1;
        }
    }

    // With imbalanced phantom load, traffic concentrates on worker3
    assert!(
        selections[2] > 80,
        "worker3 (lowest phantom load) should get most traffic, got {} out of 100",
        selections[2]
    );
}

// ---------------------------------------------------------------------------
// Integration tests: end-to-end through the HTTP router with mock workers
// ---------------------------------------------------------------------------

/// Helper to build a full router + HTTP app with cache_aware policy
async fn setup_cache_aware_router(
    worker_configs: Vec<MockWorkerConfig>,
) -> (Vec<MockWorker>, Arc<dyn RouterTrait>, RouterConfig) {
    let mut workers = Vec::new();
    let mut worker_urls = Vec::new();

    for wc in worker_configs {
        let mut worker = MockWorker::new(wc);
        let url = worker.start().await.unwrap();
        worker_urls.push(url);
        workers.push(worker);
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    let config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: worker_urls.clone(),
        },
        policy: PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 2,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 0,
            max_tree_size: 100,
        },
        port: 0,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        ..Default::default()
    };

    let app_context = common::create_test_context(config.clone());
    let router = RouterFactory::create_router(&app_context).await.unwrap();
    let router = Arc::from(router);

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    (workers, router, config)
}

/// Helper to get total internal load counter across all workers.
/// This reads the router's own AtomicUsize load_counter, NOT the vLLM /get_load endpoint.
fn get_total_internal_load(router: &Arc<dyn RouterTrait>) -> usize {
    use vllm_router_rs::routers::http::router::Router;
    if let Some(r) = router.as_any().downcast_ref::<Router>() {
        r.get_workers().iter().map(|w| w.load()).sum()
    } else {
        panic!("expected Router type for load tracking test");
    }
}

#[tokio::test]
async fn test_non_streaming_request_load_returns_to_zero() {
    let (mut workers, router, config) = setup_cache_aware_router(vec![MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;

    let app = create_test_app(router.clone(), Client::new(), &config);

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            })
            .to_string(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let _ = resp.into_body().collect().await;

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    let total_load = get_total_internal_load(&router);
    assert_eq!(
        total_load, 0,
        "load should be 0 after non-streaming request completes, got {}",
        total_load
    );

    for w in &mut workers {
        w.stop().await;
    }
}

#[tokio::test]
async fn test_streaming_request_load_returns_to_zero() {
    let (mut workers, router, config) = setup_cache_aware_router(vec![MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;

    let app = create_test_app(router.clone(), Client::new(), &config);

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true
            })
            .to_string(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // Consume the entire streaming body to trigger [DONE] detection
    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(
        body_str.contains("[DONE]"),
        "streaming response should contain [DONE] marker"
    );

    // Wait for spawned task to complete decrement
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    let total_load = get_total_internal_load(&router);
    assert_eq!(
        total_load, 0,
        "load should be 0 after streaming request completes, got {}",
        total_load
    );

    for w in &mut workers {
        w.stop().await;
    }
}

#[tokio::test]
async fn test_inference_generate_tracks_in_flight_load() {
    let (mut workers, router, config) = setup_cache_aware_router(vec![MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 500,
        fail_rate: 0.0,
    }])
    .await;

    let app = create_test_app(router.clone(), Client::new(), &config);
    let mut requests = Vec::new();
    for _ in 0..4 {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/inference/v1/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "mock-model",
                    "token_ids": [1, 2, 3],
                    "sampling_params": {"max_tokens": 1},
                    "stream": false
                })
                .to_string(),
            ))
            .unwrap();
        requests.push(tokio::spawn(app.clone().oneshot(request)));
    }

    tokio::time::timeout(tokio::time::Duration::from_secs(2), async {
        while get_total_internal_load(&router) != 4 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all requests should enter cache-aware load tracking");
    assert_eq!(
        get_total_internal_load(&router),
        4,
        "/inference/v1/generate requests must participate in cache-aware load tracking"
    );

    for request in requests {
        let response = request.await.unwrap().unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let _ = response.into_body().collect().await;
    }
    assert_eq!(get_total_internal_load(&router), 0);

    for worker in &mut workers {
        worker.stop().await;
    }
}

#[tokio::test]
async fn test_failed_request_load_returns_to_zero() {
    let (mut workers, router, config) = setup_cache_aware_router(vec![MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 1.0, // always fail with 500
    }])
    .await;

    let app = create_test_app(router.clone(), Client::new(), &config);

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            })
            .to_string(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let _ = resp.into_body().collect().await;

    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    let total_load = get_total_internal_load(&router);
    assert_eq!(
        total_load, 0,
        "load should be 0 after failed request, got {} (phantom request!)",
        total_load
    );

    for w in &mut workers {
        w.stop().await;
    }
}

#[tokio::test]
async fn test_many_sequential_requests_no_phantom_accumulation() {
    // Core phantom request detection test: send many requests and verify
    // the load counter stays at 0 after each batch completes.
    let (mut workers, router, config) = setup_cache_aware_router(vec![
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        },
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        },
    ])
    .await;

    for i in 0..50 {
        let is_stream = i % 2 == 0;
        let app = create_test_app(router.clone(), Client::new(), &config);

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": format!("request {}", i)}],
                    "stream": is_stream
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let _ = resp.into_body().collect().await;
    }

    // Wait for all spawned streaming tasks to finish
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let total_load = get_total_internal_load(&router);
    assert_eq!(
        total_load, 0,
        "total load should be 0 after 50 requests complete, got {} (phantom requests!)",
        total_load
    );

    for w in &mut workers {
        w.stop().await;
    }
}

#[tokio::test]
async fn test_mixed_success_failure_no_phantom_accumulation() {
    // Mix of successes and 500 failures — load should still net to 0.
    let (mut workers, router, config) = setup_cache_aware_router(vec![
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.3,
        },
        MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.3,
        },
    ])
    .await;

    for i in 0..30 {
        let is_stream = i % 3 == 0;
        let app = create_test_app(router.clone(), Client::new(), &config);

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": format!("request {}", i)}],
                    "stream": is_stream
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let _ = resp.into_body().collect().await;
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let total_load = get_total_internal_load(&router);
    assert_eq!(
        total_load, 0,
        "total load should be 0 after mixed requests, got {} (phantom requests!)",
        total_load
    );

    for w in &mut workers {
        w.stop().await;
    }
}
