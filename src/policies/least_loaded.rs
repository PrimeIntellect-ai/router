//! Session-affine least-loaded routing.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

#[derive(Debug, Default)]
struct State {
    assignments: HashMap<String, Weak<dyn Worker>>,
    next_tie: usize,
}

/// Assign each new session to the worker with the fewest active sessions.
///
/// Requests carrying the same `X-Session-ID` remain on their assigned worker
/// until the client explicitly releases the session. Worker load therefore
/// means active trajectories for this policy, not momentary HTTP requests.
#[derive(Debug, Default)]
pub struct LeastLoadedPolicy {
    state: Mutex<State>,
}

impl LeastLoadedPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    fn record_selection(&self, workers: &[Arc<dyn Worker>], idx: usize) -> usize {
        let worker = &workers[idx];
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision(self.name(), worker.url());
        idx
    }

    fn decrement_assignment(assignment: Weak<dyn Worker>) {
        if let Some(worker) = assignment.upgrade() {
            worker.decrement_load();
            RouterMetrics::set_worker_load(worker.url(), worker.load());
        }
    }

    fn choose_least_loaded(
        workers: &[Arc<dyn Worker>],
        healthy: Vec<usize>,
        next_tie: &mut usize,
    ) -> Option<usize> {
        let min_load = healthy.iter().map(|&idx| workers[idx].load()).min()?;
        let candidates: Vec<usize> = healthy
            .into_iter()
            .filter(|&idx| workers[idx].load() == min_load)
            .collect();
        let idx = candidates[*next_tie % candidates.len()];
        *next_tie = next_tie.wrapping_add(1);
        Some(idx)
    }
}

impl LoadBalancingPolicy for LeastLoadedPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }

        let session_id = headers
            .and_then(|headers| headers.get("x-session-id"))
            .filter(|value| !value.is_empty())?;

        let mut state = self.state.lock().unwrap();
        if let Some(assigned_worker) = state.assignments.get(session_id).and_then(Weak::upgrade) {
            if let Some(idx) = healthy
                .iter()
                .copied()
                .find(|&idx| Arc::ptr_eq(&workers[idx], &assigned_worker))
            {
                return Some(self.record_selection(workers, idx));
            }
        }

        if let Some(stale) = state.assignments.remove(session_id) {
            Self::decrement_assignment(stale);
        }

        let idx = Self::choose_least_loaded(workers, healthy, &mut state.next_tie)?;

        workers[idx].increment_load();
        RouterMetrics::set_worker_load(workers[idx].url(), workers[idx].load());
        state
            .assignments
            .insert(session_id.clone(), Arc::downgrade(&workers[idx]));
        Some(self.record_selection(workers, idx))
    }

    fn release_session(&self, session_id: &str) -> bool {
        let assignment = self.state.lock().unwrap().assignments.remove(session_id);
        if let Some(assignment) = assignment {
            Self::decrement_assignment(assignment);
            true
        } else {
            false
        }
    }

    fn name(&self) -> &'static str {
        "least_loaded"
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn reset(&self) {
        let assignments = std::mem::take(&mut self.state.lock().unwrap().assignments);
        for assignment in assignments.into_values() {
            Self::decrement_assignment(assignment);
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    fn workers() -> Vec<Arc<dyn Worker>> {
        (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(
                    format!("http://worker-{i}"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect()
    }

    fn headers(session_id: &str) -> RequestHeaders {
        HashMap::from([("x-session-id".to_string(), session_id.to_string())])
    }

    #[test]
    fn balances_new_sessions_and_keeps_them_sticky_until_release() {
        let policy = LeastLoadedPolicy::new();
        let workers = workers();

        let first = policy.select_worker_with_headers(&workers, None, Some(&headers("first")));
        policy.select_worker_with_headers(&workers, None, Some(&headers("second")));
        policy.select_worker_with_headers(&workers, None, Some(&headers("third")));

        assert_eq!(
            workers
                .iter()
                .map(|worker| worker.load())
                .collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&headers("first"))),
            first
        );
        assert_eq!(
            workers
                .iter()
                .map(|worker| worker.load())
                .collect::<Vec<_>>(),
            vec![1, 1, 1]
        );

        assert!(policy.release_session("first"));
        assert!(!policy.release_session("first"));
        let replacement =
            policy.select_worker_with_headers(&workers, None, Some(&headers("replacement")));

        assert_eq!(replacement, first);
        assert_eq!(
            workers
                .iter()
                .map(|worker| worker.load())
                .collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
    }

    #[test]
    fn requires_a_session_id() {
        let policy = LeastLoadedPolicy::new();
        let workers = workers();

        assert_eq!(
            policy.select_worker_with_headers(&workers, None, None),
            None
        );
    }
}
