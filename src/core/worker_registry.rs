//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing

use crate::core::{ConnectionMode, Worker, WorkerType};
use dashmap::DashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

/// Strip the `@<rank>` DP suffix from a worker URL, returning the base URL.
///
/// E.g. `"http://host:8000@2"` → `"http://host:8000"`.
/// Returns the original string unchanged if there is no numeric suffix.
#[must_use]
pub fn strip_dp_rank(url: &str) -> &str {
    if let Some(at_pos) = url.rfind('@') {
        if url[at_pos + 1..].parse::<usize>().is_ok() {
            return &url[..at_pos];
        }
    }
    url
}

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for the model index to reduce complexity
type ModelIndex = Arc<DashMap<String, Arc<RwLock<Vec<Arc<dyn Worker>>>>>>;

/// Worker registry with model-based indexing
#[derive(Debug, Clone)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Workers indexed by model ID (stores WorkerId for reference)
    model_workers: Arc<DashMap<String, Vec<WorkerId>>>,

    /// Optimized model index for O(1) lookups (stores Arc<dyn Worker> directly)
    model_index: ModelIndex,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping (for backward compatibility)
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Tracks the last-known model IDs per worker URL.
    /// Used by the health checker to detect model changes, since
    /// the Worker trait's model_id() reads from immutable metadata.
    /// A worker can serve multiple models (e.g. base model + LoRA adapters).
    known_models: Arc<DashMap<String, Vec<String>>>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            model_workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            known_models: Arc::new(DashMap::new()),
        }
    }

    /// Register a new worker
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let worker_id = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            // Worker with this URL already exists, update it
            existing_id.clone()
        } else {
            WorkerId::new()
        };

        // Store worker
        self.workers.insert(worker_id.clone(), worker.clone());

        // Update URL mapping
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // Update model index (both ID-based and optimized)
        let model_id = worker.model_id().to_string();
        self.model_workers
            .entry(model_id.clone())
            .or_default()
            .push(worker_id.clone());

        // Update optimized model index for O(1) lookups
        self.model_index
            .entry(model_id)
            .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
            .write()
            .expect("RwLock for model_index is poisoned")
            .push(worker.clone());

        // Update type index
        self.type_workers
            .entry(worker.worker_type())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index
        self.connection_workers
            .entry(worker.connection_mode())
            .or_default()
            .push(worker_id.clone());

        worker_id
    }

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker)) = self.workers.remove(worker_id) {
            // Remove from URL mapping
            self.url_to_id.remove(worker.url());

            // Collect all models this worker was indexed under.
            // Always include worker.model_id() in case re-registration changed
            // the immutable label before sync_worker_models updated known_models.
            let mut models_to_clean: Vec<String> = match self.known_models.remove(worker.url()) {
                Some((_, models)) => models,
                None => Vec::new(),
            };
            let base_model = worker.model_id().to_string();
            if !models_to_clean.contains(&base_model) {
                models_to_clean.push(base_model);
            }

            // Remove from all model indexes (base model + LoRA adapters)
            let worker_url = worker.url();
            for model in &models_to_clean {
                if let Some(mut ids) = self.model_workers.get_mut(model.as_str()) {
                    ids.retain(|id| *id != *worker_id);
                }
                if let Some(entry) = self.model_index.get(model.as_str()) {
                    match entry.write() {
                        Ok(mut vec) => vec.retain(|w| w.url() != worker_url),
                        Err(e) => warn!("Poisoned model_index lock for '{}': {}", model, e),
                    }
                }
            }

            // Remove from type index
            if let Some(mut type_workers) = self.type_workers.get_mut(&worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            // Remove from connection mode index
            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(&worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove(&worker_id)
        } else {
            None
        }
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Get all workers for a model
    pub fn get_by_model(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_workers
            .get(model_id)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers for a model (O(1) optimized version)
    /// This method uses the pre-indexed model_index for fast lookups
    pub fn get_by_model_fast(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_index
            .get(model_id)
            .map(|workers| {
                workers
                    .read()
                    .expect("RwLock for model_index is poisoned")
                    .clone()
            })
            .unwrap_or_default()
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill { .. } => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    /// Re-index a worker under a new `model_id`.
    ///
    /// Removes the worker from its old model index and inserts it under
    /// `new_model_id`. No-op if the model hasn't changed or the worker
    /// is not in the registry.
    pub fn update_worker_model(&self, url: &str, new_model_id: &str) {
        self.sync_worker_models(url, &[new_model_id.to_string()]);
    }

    /// Sync a worker's model index to match the given set of models.
    ///
    /// Removes the worker from model indexes it no longer serves, and adds
    /// it to indexes for newly discovered models. This supports workers that
    /// serve multiple models (e.g. a base model + LoRA adapters).
    pub fn sync_worker_models(&self, url: &str, new_models: &[String]) {
        let Some(worker) = self.get_by_url(url) else {
            return;
        };

        let Some(worker_id) = self.url_to_id.get(url).map(|id| id.clone()) else {
            return;
        };

        let old_models: Vec<String> = match self.known_models.get(url) {
            Some(entry) => entry.clone(),
            None => vec![worker.model_id().to_string()],
        };

        if old_models == new_models {
            return;
        }

        let old_set: std::collections::HashSet<&str> =
            old_models.iter().map(|s| s.as_str()).collect();
        let new_set: std::collections::HashSet<&str> =
            new_models.iter().map(|s| s.as_str()).collect();

        // Remove from indexes for models no longer served
        for removed in old_set.difference(&new_set) {
            info!("Model removed on {}: '{}'", url, removed);
            if let Some(mut ids) = self.model_workers.get_mut(*removed) {
                ids.retain(|id| *id != worker_id);
            }
            if let Some(entry) = self.model_index.get(*removed) {
                match entry.write() {
                    Ok(mut vec) => vec.retain(|w| w.url() != url),
                    Err(e) => warn!("Poisoned model_index lock for '{}': {}", removed, e),
                }
            }
        }

        // Add to indexes for newly discovered models
        for added in new_set.difference(&old_set) {
            info!("Model added on {}: '{}'", url, added);
            self.model_workers
                .entry(added.to_string())
                .or_default()
                .push(worker_id.clone());
            match self
                .model_index
                .entry(added.to_string())
                .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
                .write()
            {
                Ok(mut vec) => vec.push(worker.clone()),
                Err(e) => warn!("Poisoned model_index lock for '{}': {}", added, e),
            }
        }

        // Record current models for next refresh cycle
        self.known_models.insert(url.to_string(), new_models.to_vec());
    }

    /// Get all model IDs with workers
    pub fn get_models(&self) -> Vec<String> {
        self.model_workers
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode (Http, Grpc)
    /// - healthy_only: Only return healthy workers
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the most efficient collection based on filters
        // Use model index when possible as it's O(1) lookup
        let workers = if let Some(model) = model_id {
            self.get_by_model_fast(model)
        } else {
            self.get_all()
        };

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified
                if let Some(ref conn) = connection_mode {
                    if w.connection_mode() != *conn {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        let total_models = self.get_models().len();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;

        for worker in self.get_all() {
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
        }
    }

    /// Start a health checker for all workers in the registry.
    ///
    /// Periodically checks `/health` on every worker and refreshes the model index
    /// by querying `/v1/models`. If a worker's loaded model changes (e.g. `LoRA`
    /// load/evict), the model index is updated automatically.
    #[must_use]
    pub fn start_health_checker(&self, check_interval_secs: u64) -> crate::core::HealthChecker {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let registry = self.clone();

        let handle = tokio::spawn(async move {
            const LOAD_RESET_INTERVAL: u64 = 10;
            const MODEL_REFRESH_INTERVAL: u64 = 5;

            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));
            let mut check_count = 0u64;

            loop {
                interval.tick().await;

                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                let workers: Vec<Arc<dyn crate::core::Worker>> = registry
                    .workers
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Perform health checks concurrently (not sequentially)
                let health_checks = workers.iter().map(|worker| {
                    let worker_url = worker.url().to_string();
                    let was_healthy = worker.is_healthy();

                    async move {
                        match worker.check_health_async().await {
                            Ok(()) => {
                                if !was_healthy {
                                    tracing::info!("Worker {} is now healthy", worker_url);
                                }
                            }
                            Err(e) => {
                                if was_healthy {
                                    tracing::warn!(
                                        "Worker {} health check failed: {}",
                                        worker_url,
                                        e
                                    );
                                } else {
                                    tracing::debug!(
                                        "Worker {} remains unhealthy: {}",
                                        worker_url,
                                        e
                                    );
                                }
                            }
                        }
                    }
                });
                futures::future::join_all(health_checks).await;

                check_count += 1;

                // Periodically refresh model discovery
                // TODO: notify PolicyRegistry on model changes so per-model
                // policies stay in sync (requires a callback or moving this
                // logic to a layer that has access to both registries).
                if check_count % MODEL_REFRESH_INTERVAL == 0 {
                    // Deduplicate by base URL so DP workers (@0, @1, …)
                    // sharing the same endpoint only trigger one fetch.
                    let mut fetched: std::collections::HashMap<String, Vec<String>> =
                        std::collections::HashMap::new();

                    for worker in &workers {
                        if !worker.is_healthy() {
                            continue;
                        }
                        let base_url = strip_dp_rank(worker.url()).to_string();
                        let new_models = match fetched.entry(base_url.clone()) {
                            std::collections::hash_map::Entry::Occupied(e) => e.get().clone(),
                            std::collections::hash_map::Entry::Vacant(e) => {
                                let models =
                                    crate::core::worker::fetch_models_from_worker(&base_url)
                                        .await;
                                e.insert(models.clone());
                                models
                            }
                        };
                        if !new_models.is_empty() {
                            registry.sync_worker_models(worker.url(), &new_models);
                        }
                    }
                }

                // Only reset loads when traffic is idle to prevent drift
                if check_count % LOAD_RESET_INTERVAL == 0 {
                    let max_load = workers.iter().map(|w| w.load()).max().unwrap_or(0);
                    if max_load <= 2 {
                        tracing::debug!(
                            "Resetting worker loads to prevent drift (max_load: {})",
                            max_load
                        );
                        for worker in &workers {
                            worker.reset_load();
                        }
                    }
                }
            }
        });

        crate::core::HealthChecker::new(handle, shutdown)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    pub total_workers: usize,
    pub total_models: usize,
    pub healthy_workers: usize,
    pub total_load: usize,
    pub regular_workers: usize,
    pub prefill_workers: usize,
    pub decode_workers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CircuitBreakerConfig, WorkerFactory};
    use std::collections::HashMap;

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker = WorkerFactory::create_regular_with_labels(
            "http://worker1:8080".to_string(),
            labels,
            CircuitBreakerConfig::default(),
        );

        // Register worker (WorkerFactory returns Box<dyn Worker>, convert to Arc)
        let worker_id = registry.register(Arc::from(worker));

        // Verify registration
        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        // Test stats
        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1 = WorkerFactory::create_regular_with_labels(
            "http://worker1:8080".to_string(),
            labels1,
            CircuitBreakerConfig::default(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2 = WorkerFactory::create_regular_with_labels(
            "http://worker2:8080".to_string(),
            labels2,
            CircuitBreakerConfig::default(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3 = WorkerFactory::create_regular_with_labels(
            "http://worker3:8080".to_string(),
            labels3,
            CircuitBreakerConfig::default(),
        );

        // Register workers
        registry.register(Arc::from(worker1));
        registry.register(Arc::from(worker2));
        registry.register(Arc::from(worker3));

        // Test get_by_model_fast for llama-3
        let llama_workers = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        // Test get_by_model_fast for gpt-4
        let gpt_workers = registry.get_by_model_fast("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        // Test get_by_model_fast for non-existent model
        let unknown_workers = registry.get_by_model_fast("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        // Test that both get_by_model and get_by_model_fast return same results
        let llama_workers_slow = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), llama_workers_slow.len());

        // Test removal updates the model index
        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }

    /// Simulates the autoscale scenario from issue #1092:
    ///
    /// - worker1 (inference-0) serves base model + LoRA adapter
    /// - worker2 (inference-1) scales up with only the base model
    /// - Requests for the LoRA model should only route to worker1
    #[test]
    fn test_lora_adapter_routing_after_autoscale() {
        let registry = WorkerRegistry::new();

        // worker1 = inference-0: has the base model
        let mut labels1 = HashMap::new();
        labels1.insert(
            "model_id".to_string(),
            "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string(),
        );
        let worker1 = WorkerFactory::create_regular_with_labels(
            "http://inference-0:8000".to_string(),
            labels1,
            CircuitBreakerConfig::default(),
        );
        registry.register(Arc::from(worker1));

        // worker2 = inference-1: autoscaled replica, also has base model only
        let mut labels2 = HashMap::new();
        labels2.insert(
            "model_id".to_string(),
            "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string(),
        );
        let worker2 = WorkerFactory::create_regular_with_labels(
            "http://inference-1:8000".to_string(),
            labels2,
            CircuitBreakerConfig::default(),
        );
        registry.register(Arc::from(worker2));

        // Both workers serve the base model
        assert_eq!(
            registry
                .get_by_model_fast("Qwen/Qwen3-30B-A3B-Instruct-2507")
                .len(),
            2
        );

        // Simulate health checker: inference-0's /v1/models returns base + LoRA
        registry.sync_worker_models(
            "http://inference-0:8000",
            &[
                "Qwen/Qwen3-30B-A3B-Instruct-2507".to_string(),
                "rft-s7lpo1teyeaarb2m28e5s81a".to_string(),
            ],
        );

        // inference-1 only has the base model
        registry.sync_worker_models(
            "http://inference-1:8000",
            &["Qwen/Qwen3-30B-A3B-Instruct-2507".to_string()],
        );

        // LoRA adapter should only route to inference-0
        let lora_workers = registry.get_by_model_fast("rft-s7lpo1teyeaarb2m28e5s81a");
        assert_eq!(
            lora_workers.len(),
            1,
            "LoRA adapter should be routable to inference-0"
        );
        assert_eq!(lora_workers[0].url(), "http://inference-0:8000");

        // Base model should still be routable to BOTH workers
        let base_workers = registry.get_by_model_fast("Qwen/Qwen3-30B-A3B-Instruct-2507");
        assert_eq!(
            base_workers.len(),
            2,
            "Base model should be routable to both workers"
        );
    }

    /// Tests that sync_worker_models correctly handles LoRA eviction:
    /// when a LoRA is unloaded, the worker is removed from that model's index
    /// but stays in the base model index.
    #[test]
    fn test_sync_worker_models_lora_eviction() {
        let registry = WorkerRegistry::new();

        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "base-model".to_string());
        let worker = WorkerFactory::create_regular_with_labels(
            "http://worker1:8000".to_string(),
            labels,
            CircuitBreakerConfig::default(),
        );
        registry.register(Arc::from(worker));

        // Worker loads a LoRA adapter
        registry.sync_worker_models(
            "http://worker1:8000",
            &[
                "base-model".to_string(),
                "rft-lora-adapter".to_string(),
            ],
        );
        assert_eq!(registry.get_by_model_fast("base-model").len(), 1);
        assert_eq!(registry.get_by_model_fast("rft-lora-adapter").len(), 1);

        // LoRA gets evicted — worker now only serves base model
        registry.sync_worker_models(
            "http://worker1:8000",
            &["base-model".to_string()],
        );
        assert_eq!(
            registry.get_by_model_fast("base-model").len(),
            1,
            "Worker should still serve base model after LoRA eviction"
        );
        assert_eq!(
            registry.get_by_model_fast("rft-lora-adapter").len(),
            0,
            "Evicted LoRA should have no workers"
        );
    }

    /// Tests that removing a worker cleans up all model indexes including LoRAs.
    #[test]
    fn test_remove_worker_cleans_up_lora_indexes() {
        let registry = WorkerRegistry::new();

        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "base-model".to_string());
        let worker = WorkerFactory::create_regular_with_labels(
            "http://worker1:8000".to_string(),
            labels,
            CircuitBreakerConfig::default(),
        );
        registry.register(Arc::from(worker));

        // Worker has base model + LoRA
        registry.sync_worker_models(
            "http://worker1:8000",
            &[
                "base-model".to_string(),
                "rft-lora-adapter".to_string(),
            ],
        );
        assert_eq!(registry.get_by_model_fast("rft-lora-adapter").len(), 1);

        // Remove the worker (simulates KEDA scale-down)
        registry.remove_by_url("http://worker1:8000");

        // Both base model and LoRA indexes should be empty
        assert_eq!(
            registry.get_by_model_fast("base-model").len(),
            0,
            "Base model index should be empty after worker removal"
        );
        assert_eq!(
            registry.get_by_model_fast("rft-lora-adapter").len(),
            0,
            "LoRA index should be empty after worker removal"
        );
    }
}
