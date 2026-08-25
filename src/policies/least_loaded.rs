//! Session-affine least-loaded routing.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

const SESSION_IDLE_TTL: Duration = Duration::from_secs(6 * 60 * 60);

#[derive(Debug)]
struct Assignment {
    worker: Weak<dyn Worker>,
    last_seen: Instant,
}

#[derive(Debug, Default)]
struct State {
    assignments: HashMap<String, Assignment>,
    next_tie: usize,
}

/// Assign each new session to the worker with the fewest active sessions.
///
/// Requests carrying the same `X-Session-ID` remain on their assigned worker
/// until the client explicitly releases the session. Worker load therefore
/// means active trajectories for this policy, not momentary HTTP requests.
#[derive(Debug)]
pub struct LeastLoadedPolicy {
    state: Mutex<State>,
    session_idle_ttl: Duration,
}

impl Default for LeastLoadedPolicy {
    fn default() -> Self {
        Self {
            state: Mutex::default(),
            session_idle_ttl: SESSION_IDLE_TTL,
        }
    }
}

impl LeastLoadedPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_session_idle_ttl(session_idle_ttl: Duration) -> Self {
        Self {
            state: Mutex::default(),
            session_idle_ttl,
        }
    }

    fn record_selection(&self, workers: &[Arc<dyn Worker>], idx: usize) -> usize {
        let worker = &workers[idx];
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision(self.name(), worker.url());
        idx
    }

    fn decrement_assignment(assignment: &Assignment) {
        if let Some(worker) = assignment.worker.upgrade() {
            worker.decrement_load();
            RouterMetrics::set_worker_load(worker.url(), worker.load());
        }
    }

    fn remove_expired(state: &mut State, now: Instant, ttl: Duration) {
        state.assignments.retain(|_, assignment| {
            if now.duration_since(assignment.last_seen) >= ttl {
                Self::decrement_assignment(assignment);
                false
            } else {
                true
            }
        });
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

        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        if let Some(assignment) = state.assignments.get_mut(session_id) {
            if now.duration_since(assignment.last_seen) < self.session_idle_ttl {
                assignment.last_seen = now;
                if let Some(assigned_worker) = assignment.worker.upgrade() {
                    if let Some(idx) = healthy
                        .iter()
                        .copied()
                        .find(|&idx| Arc::ptr_eq(&workers[idx], &assigned_worker))
                    {
                        return Some(self.record_selection(workers, idx));
                    }
                }
            }
        }

        if let Some(stale) = state.assignments.remove(session_id) {
            Self::decrement_assignment(&stale);
        }
        // Existing sessions stay on the fast path above. Scan only when assigning
        // new work, which is also the only time leaked load can affect a decision.
        Self::remove_expired(&mut state, now, self.session_idle_ttl);

        let idx = Self::choose_least_loaded(workers, healthy, &mut state.next_tie)?;

        workers[idx].increment_load();
        RouterMetrics::set_worker_load(workers[idx].url(), workers[idx].load());
        state.assignments.insert(
            session_id.clone(),
            Assignment {
                worker: Arc::downgrade(&workers[idx]),
                last_seen: now,
            },
        );
        Some(self.record_selection(workers, idx))
    }

    fn release_session(&self, session_id: &str) -> bool {
        let assignment = self.state.lock().unwrap().assignments.remove(session_id);
        if let Some(assignment) = assignment {
            Self::decrement_assignment(&assignment);
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
            Self::decrement_assignment(&assignment);
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

    #[test]
    fn expires_abandoned_sessions_before_routing_new_work() {
        let policy = LeastLoadedPolicy::with_session_idle_ttl(Duration::from_secs(1));
        let workers = workers();

        let abandoned =
            policy.select_worker_with_headers(&workers, None, Some(&headers("abandoned")));
        {
            let mut state = policy.state.lock().unwrap();
            state.assignments.get_mut("abandoned").unwrap().last_seen =
                Instant::now() - Duration::from_secs(2);
        }

        let replacement =
            policy.select_worker_with_headers(&workers, None, Some(&headers("replacement")));

        assert_ne!(replacement, abandoned);
        assert_eq!(workers[abandoned.unwrap()].load(), 0);
        assert_eq!(workers[replacement.unwrap()].load(), 1);
        let state = policy.state.lock().unwrap();
        assert!(!state.assignments.contains_key("abandoned"));
        assert!(state.assignments.contains_key("replacement"));
    }
}
