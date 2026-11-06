//! Sticky least-loaded session routing policy
//!
//! Adapted from SumanthRH/router commit b50d926f3cf8868e12e8395fa02ce1aa2b1ae5ae
//! (Apache-2.0), with typed-body routing and per-entry expiration fixes.
//!
//! This policy combines session stickiness (so all requests for a given
//! trajectory/session reuse the same vLLM replica and maximize KV cache reuse)
//! with load balancing across replicas (so *new* sessions are assigned to the
//! replica with the fewest active sessions).
//!
//! Behavior:
//! - The router bookkeeps the set of active sessions per replica.
//! - For an existing session: requests with the same session id are routed to
//!   the replica that session was originally assigned to.
//! - For a new session: the session is assigned to the healthy replica with the
//!   least number of active sessions. Ties are broken deterministically using
//!   consistent hashing (rendezvous / highest-random-weight) so that, given the
//!   same set of least-loaded replicas, a session is always assigned to the same
//!   replica.
//! - Sessions expire after a configurable TTL (default 2 hours, overridable via
//!   the `VLLM_ROUTER_SLL_SESSION_EXPIRATION_IN_S` environment variable). This
//!   protects against leaked sessions when a `finish_session` call is never
//!   received (e.g. a cancelled trajectory). The TTL counts idle time: a session
//!   with a request in flight (reported through `SessionRequestGuard`) never
//!   expires, and the clock restarts when the request completes. Otherwise a
//!   generation that outlives the TTL would send the session's next turn to a
//!   different replica, which prefills the whole context again.
//! - A session counts toward its replica's load only while it is *active*: it
//!   has a request in flight, or its last request started or finished within
//!   the active window (default: the session expiration, overridable via the
//!   `VLLM_ROUTER_SLL_ACTIVE_WINDOW_IN_S` environment variable). An inactive
//!   session stays pinned until it expires but no longer steers new sessions
//!   away from its replica, so sessions that clients never finish stop counting
//!   as load long before they expire.
//! - Optionally (`VLLM_ROUTER_SLL_REBALANCE_RATIO`, default 0 = off), a pinned
//!   session with no request in flight moves to the least-loaded healthy replica
//!   when its own replica has more than `ratio` times as many active sessions,
//!   and at least [`REBALANCE_MIN_GAP`] more. This drains replicas that took
//!   over sessions while another replica was unavailable.
//! - Sessions can be released explicitly via [`finish_session`].
//!
//! Requests without any session identifier are still load-balanced to the
//! least-loaded replica, but are *not* recorded as active sessions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use super::get_healthy_worker_indices;
use super::hash_key;
use super::ConsistentHashPolicy;
use super::LoadBalancingPolicy;
use super::RequestHeaders;
use crate::core::Worker;
use crate::metrics::RouterMetrics;

/// Default idle-session expiration (2 hours).
pub const DEFAULT_SESSION_EXPIRATION_SECS: u64 = 7200;

/// Environment variable to override the session expiration (in seconds).
pub const SESSION_EXPIRATION_ENV: &str = "VLLM_ROUTER_SLL_SESSION_EXPIRATION_IN_S";

/// Default maximum number of sessions retained by one policy instance.
pub const DEFAULT_MAX_SESSIONS: usize = 100_000;

/// Environment variable to override the maximum retained session count.
pub const MAX_SESSIONS_ENV: &str = "VLLM_ROUTER_SLL_MAX_SESSIONS";

/// Default maximum session identifier size in bytes.
pub const DEFAULT_MAX_SESSION_ID_BYTES: usize = 256;

/// Environment variable to override the maximum session identifier size.
pub const MAX_SESSION_ID_BYTES_ENV: &str = "VLLM_ROUTER_SLL_MAX_SESSION_ID_BYTES";

/// Environment variable to override the active window (in seconds): how long
/// after its last request starts or finishes a session still counts toward its
/// replica's load. Defaults to the session expiration.
pub const ACTIVE_WINDOW_ENV: &str = "VLLM_ROUTER_SLL_ACTIVE_WINDOW_IN_S";

/// Environment variable for the rebalance ratio; 0 (the default) disables
/// rebalancing.
pub const REBALANCE_RATIO_ENV: &str = "VLLM_ROUTER_SLL_REBALANCE_RATIO";

/// Minimum number of active sessions, not counting the moving session, by which
/// its replica must exceed the least-loaded one before rebalancing moves it.
pub const REBALANCE_MIN_GAP: usize = 4;

/// Minimum interval between full expiration sweeps to bound per-request cost.
const SWEEP_INTERVAL_SECS: u64 = 60;

/// Bookkeeping for a single pinned session.
#[derive(Debug)]
struct SessionEntry {
    worker_url: String,
    /// Start or end of the session's last request.
    last_access: Instant,
    /// Routed requests that have not completed yet.
    in_flight: usize,
    /// Whether the session counts toward `active_counts`.
    active: bool,
}

impl SessionEntry {
    fn is_idle(&self, period: Duration, now: Instant) -> bool {
        self.in_flight == 0 && now.duration_since(self.last_access) >= period
    }
}

/// Internal mutable state guarded by a single mutex so that "find least-loaded
/// replica + record session" is atomic across concurrent new-session requests.
#[derive(Debug)]
struct SllState {
    /// session id -> assigned replica
    sessions: HashMap<String, SessionEntry>,
    /// replica url -> number of active sessions, the load used for placement
    active_counts: HashMap<String, usize>,
    /// replica url -> number of pinned sessions, active or not
    pinned_counts: HashMap<String, usize>,
    /// Round-robin cursor for requests without a session identifier.
    stateless_cursor: usize,
    /// Timestamp of the last expiration sweep.
    last_sweep: Instant,
}

fn count(counts: &HashMap<String, usize>, worker_url: &str) -> usize {
    counts.get(worker_url).copied().unwrap_or(0)
}

fn increment_count(counts: &mut HashMap<String, usize>, worker_url: &str) {
    *counts.entry(worker_url.to_string()).or_insert(0) += 1;
}

/// Decrement the session count for a replica, removing the entry when it
/// reaches zero.
fn decrement_count(counts: &mut HashMap<String, usize>, worker_url: &str) {
    if let Some(count) = counts.get_mut(worker_url) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            counts.remove(worker_url);
        }
    }
}

impl SllState {
    /// Pin a session to `worker_url` as an active session.
    fn insert_session(
        &mut self,
        session_id: String,
        worker_url: String,
        now: Instant,
        in_flight: usize,
    ) {
        increment_count(&mut self.pinned_counts, &worker_url);
        increment_count(&mut self.active_counts, &worker_url);
        self.publish(&worker_url);
        self.sessions.insert(
            session_id,
            SessionEntry {
                worker_url,
                last_access: now,
                in_flight,
                active: true,
            },
        );
    }

    /// Unpin a session, returning its entry.
    fn remove_session(&mut self, session_id: &str) -> Option<SessionEntry> {
        let entry = self.sessions.remove(session_id)?;
        decrement_count(&mut self.pinned_counts, &entry.worker_url);
        if entry.active {
            decrement_count(&mut self.active_counts, &entry.worker_url);
        }
        self.publish(&entry.worker_url);
        Some(entry)
    }

    /// Count a pinned session toward its replica's load again.
    fn mark_active(&mut self, session_id: &str) {
        let Some(entry) = self.sessions.get_mut(session_id) else {
            return;
        };
        if entry.active {
            return;
        }
        entry.active = true;
        let worker_url = entry.worker_url.clone();
        increment_count(&mut self.active_counts, &worker_url);
        self.publish(&worker_url);
    }

    /// Export a replica's session counts.
    fn publish(&self, worker_url: &str) {
        RouterMetrics::set_sll_sessions(
            worker_url,
            count(&self.active_counts, worker_url),
            count(&self.pinned_counts, worker_url),
        );
    }
}

/// Sticky least-loaded routing policy.
#[derive(Debug)]
pub struct StickyLeastLoadedPolicy {
    state: Mutex<SllState>,
    session_expiration: Duration,
    active_window: Duration,
    rebalance_ratio: f64,
    max_sessions: usize,
    max_session_id_bytes: usize,
}

impl StickyLeastLoadedPolicy {
    /// Create a new policy, reading its settings from the environment (falling
    /// back to [`DEFAULT_SESSION_EXPIRATION_SECS`], an active window equal to the
    /// expiration, and no rebalancing).
    pub fn new() -> Self {
        let expiration_secs = Self::env_or(SESSION_EXPIRATION_ENV, DEFAULT_SESSION_EXPIRATION_SECS);
        let policy = Self::with_limits(
            expiration_secs,
            Self::env_or(MAX_SESSIONS_ENV, DEFAULT_MAX_SESSIONS),
            Self::env_or(MAX_SESSION_ID_BYTES_ENV, DEFAULT_MAX_SESSION_ID_BYTES),
        )
        .with_active_window_secs(Self::env_or(ACTIVE_WINDOW_ENV, expiration_secs))
        .with_rebalance_ratio(Self::env_or(REBALANCE_RATIO_ENV, 0.0));
        info!(
            "SLL: session expiration {}s, active window {}s, rebalance ratio {}",
            policy.session_expiration.as_secs(),
            policy.active_window.as_secs(),
            policy.rebalance_ratio
        );
        policy
    }

    /// Create a new policy with an explicit session expiration (in seconds).
    pub fn with_expiration_secs(secs: u64) -> Self {
        Self::with_limits(secs, DEFAULT_MAX_SESSIONS, DEFAULT_MAX_SESSION_ID_BYTES)
    }

    /// Count sessions toward their replica's load only for `secs` after their
    /// last request starts or finishes, or while a request is in flight.
    pub fn with_active_window_secs(mut self, secs: u64) -> Self {
        self.active_window = Duration::from_secs(secs);
        self
    }

    /// Move idle pinned sessions off replicas with more than `ratio` times the
    /// active sessions of the least-loaded healthy replica; 0 disables this.
    pub fn with_rebalance_ratio(mut self, ratio: f64) -> Self {
        self.rebalance_ratio = if ratio.is_finite() && ratio >= 0.0 {
            ratio
        } else {
            warn!(
                "Invalid {} value '{}', rebalancing disabled",
                REBALANCE_RATIO_ENV, ratio
            );
            0.0
        };
        self
    }

    fn with_limits(expiration_secs: u64, max_sessions: usize, max_session_id_bytes: usize) -> Self {
        Self {
            state: Mutex::new(SllState {
                sessions: HashMap::new(),
                active_counts: HashMap::new(),
                pinned_counts: HashMap::new(),
                stateless_cursor: 0,
                last_sweep: Instant::now(),
            }),
            session_expiration: Duration::from_secs(expiration_secs),
            active_window: Duration::from_secs(expiration_secs),
            rebalance_ratio: 0.0,
            max_sessions,
            max_session_id_bytes,
        }
    }

    fn env_or<T: std::str::FromStr + std::fmt::Display>(name: &str, default: T) -> T {
        match std::env::var(name) {
            Ok(value) => value.trim().parse().unwrap_or_else(|_| {
                warn!(
                    "Invalid {} value '{}', using default {}",
                    name, value, default
                );
                default
            }),
            Err(_) => default,
        }
    }

    /// Remove sessions idle for the expiration period, and stop counting
    /// sessions idle for the active window toward their replica's load.
    fn sweep(&self, state: &mut SllState, now: Instant) {
        let expired: Vec<String> = state
            .sessions
            .iter()
            .filter(|(_, entry)| entry.is_idle(self.session_expiration, now))
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired {
            if let Some(entry) = state.remove_session(&key) {
                debug!("SLL: expired session '{}' on '{}'", key, entry.worker_url);
            }
        }

        let mut deactivated: Vec<String> = Vec::new();
        for entry in state.sessions.values_mut() {
            if entry.active && entry.is_idle(self.active_window, now) {
                entry.active = false;
                decrement_count(&mut state.active_counts, &entry.worker_url);
                if !deactivated.contains(&entry.worker_url) {
                    deactivated.push(entry.worker_url.clone());
                }
            }
        }
        for worker_url in deactivated {
            state.publish(&worker_url);
        }

        state.last_sweep = now;
    }

    /// Sweep at most once per [`SWEEP_INTERVAL_SECS`] during normal routing.
    fn sweep_if_due(&self, state: &mut SllState, now: Instant) {
        if now.duration_since(state.last_sweep) < Duration::from_secs(SWEEP_INTERVAL_SECS) {
            return;
        }
        self.sweep(state, now);
    }

    /// Whether a pinned session should leave its healthy replica: rebalancing is
    /// on, none of its requests is in flight, and its replica has more than
    /// `rebalance_ratio` times, and at least [`REBALANCE_MIN_GAP`] more, active
    /// sessions than the least-loaded other healthy replica. Loads exclude the
    /// session itself.
    fn should_rebalance(
        &self,
        state: &SllState,
        entry: &SessionEntry,
        candidates: &[(usize, String)],
    ) -> bool {
        if self.rebalance_ratio <= 0.0 || entry.in_flight > 0 {
            return false;
        }
        let own = count(&state.active_counts, &entry.worker_url)
            .saturating_sub(usize::from(entry.active));
        candidates
            .iter()
            .filter(|(_, url)| *url != entry.worker_url)
            .map(|(_, url)| count(&state.active_counts, url))
            .min()
            .is_some_and(|least| {
                own >= least + REBALANCE_MIN_GAP && own as f64 > self.rebalance_ratio * least as f64
            })
    }

    /// Pick the least-loaded replica among `candidates`, breaking ties with
    /// consistent (rendezvous) hashing of `tie_break_key`.
    ///
    /// `candidates` is a slice of `(worker_index, worker_url)` for healthy
    /// replicas. Returns the chosen `worker_index`.
    fn select_least_loaded(
        counts: &HashMap<String, usize>,
        candidates: &[(usize, String)],
        tie_break_key: &str,
    ) -> usize {
        let min_count = candidates
            .iter()
            .map(|(_, url)| counts.get(url).copied().unwrap_or(0))
            .min()
            .unwrap_or(0);

        // Among replicas at the minimum load, deterministically pick one using
        // rendezvous hashing (highest hash wins).
        let mut best: Option<(usize, u64)> = None;
        for (idx, url) in candidates {
            let count = counts.get(url).copied().unwrap_or(0);
            if count != min_count {
                continue;
            }
            let weight = ConsistentHashPolicy::fbi_hash(&format!("{}:{}", tie_break_key, url));
            match best {
                Some((_, best_weight)) if best_weight >= weight => {}
                _ => best = Some((*idx, weight)),
            }
        }

        // `candidates` is non-empty (callers ensure healthy workers exist), so
        // `best` is always populated.
        best.map(|(idx, _)| idx).unwrap_or(candidates[0].0)
    }

    fn select_least_loaded_round_robin(
        counts: &HashMap<String, usize>,
        candidates: &[(usize, String)],
        offset: usize,
    ) -> usize {
        let min_count = candidates
            .iter()
            .map(|(_, url)| counts.get(url).copied().unwrap_or(0))
            .min()
            .unwrap_or(0);
        let tied_count = candidates
            .iter()
            .filter(|(_, url)| counts.get(url).copied().unwrap_or(0) == min_count)
            .count();
        let target = offset % tied_count;
        candidates
            .iter()
            .filter(|(_, url)| counts.get(url).copied().unwrap_or(0) == min_count)
            .nth(target)
            .map(|(idx, _)| *idx)
            .unwrap_or(candidates[0].0)
    }

    fn select_stateless(state: &mut SllState, candidates: &[(usize, String)]) -> usize {
        let idx = Self::select_least_loaded_round_robin(
            &state.active_counts,
            candidates,
            state.stateless_cursor,
        );
        state.stateless_cursor = state.stateless_cursor.wrapping_add(1);
        idx
    }

    fn record_selection(&self, worker: &Arc<dyn Worker>) {
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision(self.name(), worker.url());
    }

    /// Number of currently active (tracked) sessions. Exposed for tests/metrics.
    pub fn active_session_count(&self) -> usize {
        self.state.lock().unwrap().sessions.len()
    }

    /// Number of active sessions assigned to a specific replica. For tests.
    pub fn active_count_for(&self, worker_url: &str) -> usize {
        count(&self.state.lock().unwrap().active_counts, worker_url)
    }

    /// Number of sessions pinned to a specific replica, active or not. For tests.
    pub fn pinned_count_for(&self, worker_url: &str) -> usize {
        count(&self.state.lock().unwrap().pinned_counts, worker_url)
    }
}

impl LoadBalancingPolicy for StickyLeastLoadedPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let candidates: Vec<(usize, String)> = healthy_indices
            .iter()
            .map(|&idx| (idx, workers[idx].url().to_string()))
            .collect();

        let session_id = match hash_key::extract_session_id(request_text, headers) {
            Some(id) if id.len() > self.max_session_id_bytes => {
                debug!(
                    "SLL: session identifier exceeds {} bytes; routing without affinity",
                    self.max_session_id_bytes
                );
                None
            }
            session_id => session_id,
        };

        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        self.sweep_if_due(&mut state, now);

        // Requests without a session id: load-balance but don't record state.
        let session_id = match session_id {
            Some(id) => id,
            None => {
                let idx = Self::select_stateless(&mut state, &candidates);
                drop(state);
                self.record_selection(&workers[idx]);
                debug!("SLL: stateless request routed to '{}'", workers[idx].url());
                return Some(idx);
            }
        };

        // Requests still running on a previous assignment complete against the new one.
        let mut in_flight = 0;
        let mut rebalanced_from = None;

        // Existing session: route to the same replica if it's still healthy.
        if let Some((worker_url, expired)) = state.sessions.get(&session_id).map(|e| {
            (
                e.worker_url.clone(),
                e.is_idle(self.session_expiration, now),
            )
        }) {
            let mut rebalance = false;
            if let Some((idx, _)) = candidates
                .iter()
                .find(|(_, url)| *url == worker_url && !expired)
            {
                rebalance =
                    self.should_rebalance(&state, &state.sessions[&session_id], &candidates);
                if !rebalance {
                    let idx = *idx;
                    if let Some(entry) = state.sessions.get_mut(&session_id) {
                        entry.last_access = now;
                    }
                    state.mark_active(&session_id);
                    drop(state);
                    self.record_selection(&workers[idx]);
                    debug!(
                        "SLL: session '{}' -> existing replica '{}'",
                        session_id, worker_url
                    );
                    return Some(idx);
                }
            }

            // Expired session, unavailable or overloaded replica: release the
            // old assignment.
            in_flight = state.remove_session(&session_id).map_or(0, |e| e.in_flight);
            if rebalance {
                rebalanced_from = Some(worker_url);
            } else {
                debug!(
                    "SLL: session '{}' replica '{}' unavailable, reassigning",
                    session_id, worker_url
                );
            }
        }

        // New session: assign to least-loaded replica (tie-broken by hashing).
        if state.sessions.len() >= self.max_sessions {
            self.sweep(&mut state, now);
        }
        if state.sessions.len() >= self.max_sessions {
            let idx = Self::select_stateless(&mut state, &candidates);
            drop(state);
            self.record_selection(&workers[idx]);
            debug!(
                "SLL: {}-session limit reached; request routed without affinity to '{}'",
                self.max_sessions,
                workers[idx].url()
            );
            return Some(idx);
        }

        let idx = Self::select_least_loaded(&state.active_counts, &candidates, &session_id);
        let worker_url = workers[idx].url().to_string();
        state.insert_session(session_id.clone(), worker_url.clone(), now, in_flight);
        let new_count = count(&state.active_counts, &worker_url);
        drop(state);

        self.record_selection(&workers[idx]);
        match rebalanced_from {
            Some(from) => {
                RouterMetrics::record_sll_rebalance(&from, &worker_url);
                info!(
                    "SLL: rebalanced session '{}' from replica '{}' to '{}' (active sessions on replica: {})",
                    session_id, from, worker_url, new_count
                );
            }
            None => info!(
                "SLL: new session '{}' -> replica '{}' (active sessions on replica: {})",
                session_id, worker_url, new_count
            ),
        }
        Some(idx)
    }

    fn finish_session(&self, session_id: &str) {
        let mut state = self.state.lock().unwrap();
        if let Some(entry) = state.remove_session(session_id) {
            info!(
                "SLL: finished session '{}' on replica '{}'",
                session_id, entry.worker_url
            );
        } else {
            debug!("SLL: finish_session for unknown session '{}'", session_id);
        }
    }

    fn tracks_session_requests(&self) -> bool {
        true
    }

    fn on_session_request_start(&self, session_id: &str) {
        let mut state = self.state.lock().unwrap();
        if let Some(entry) = state.sessions.get_mut(session_id) {
            entry.in_flight += 1;
        }
        // A session with a request in flight always counts as load.
        state.mark_active(session_id);
    }

    fn on_session_request_complete(&self, session_id: &str) {
        let mut state = self.state.lock().unwrap();
        if let Some(entry) = state.sessions.get_mut(session_id) {
            entry.in_flight = entry.in_flight.saturating_sub(1);
            entry.last_access = Instant::now();
        }
        state.mark_active(session_id);
    }

    fn name(&self) -> &'static str {
        "sticky_least_loaded"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_request_body(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        let worker_urls: Vec<String> = state.pinned_counts.keys().cloned().collect();
        state.sessions.clear();
        state.active_counts.clear();
        state.pinned_counts.clear();
        for worker_url in worker_urls {
            state.publish(&worker_url);
        }
        state.stateless_cursor = 0;
        state.last_sweep = Instant::now();
        info!("SLL: policy reset - all sessions cleared");
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for StickyLeastLoadedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use crate::policies::SessionRequestGuard;
    use std::collections::HashMap as StdHashMap;

    fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(BasicWorker::new(url.to_string(), WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect()
    }

    fn header_with_session(session_id: &str) -> RequestHeaders {
        let mut headers: RequestHeaders = StdHashMap::new();
        headers.insert("x-session-id".to_string(), session_id.to_string());
        headers
    }

    #[test]
    fn test_same_session_sticky() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let headers = header_with_session("sess-1");

        let idx1 = policy.select_worker_with_headers(&workers, None, Some(&headers));
        let idx2 = policy.select_worker_with_headers(&workers, None, Some(&headers));
        let idx3 = policy.select_worker_with_headers(&workers, None, Some(&headers));

        assert!(idx1.is_some());
        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx3);

        // Only one active session should be tracked across repeated requests.
        assert_eq!(policy.active_session_count(), 1);
    }

    #[test]
    fn test_new_sessions_balance_across_replicas() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        // Assign three distinct sessions; each should land on a distinct replica
        // because new sessions go to the least-loaded replica.
        for i in 0..3 {
            let headers = header_with_session(&format!("sess-{}", i));
            policy
                .select_worker_with_headers(&workers, None, Some(&headers))
                .unwrap();
        }

        assert_eq!(policy.active_session_count(), 3);
        for w in &workers {
            assert_eq!(
                policy.active_count_for(w.url()),
                1,
                "expected each replica to hold exactly one session"
            );
        }
    }

    #[test]
    fn test_finish_session_releases_capacity() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = header_with_session("sess-x");
        let idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        let assigned_url = workers[idx].url().to_string();
        assert_eq!(policy.active_count_for(&assigned_url), 1);

        policy.finish_session("sess-x");
        assert_eq!(policy.active_session_count(), 0);
        assert_eq!(policy.active_count_for(&assigned_url), 0);
    }

    #[test]
    fn test_finish_unknown_session_is_noop() {
        let policy = StickyLeastLoadedPolicy::new();
        // Should not panic and should leave state empty.
        policy.finish_session("does-not-exist");
        assert_eq!(policy.active_session_count(), 0);
    }

    #[test]
    fn test_session_id_from_body() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let body = r#"{"session_id": "body-sess", "prompt": "hi"}"#;

        let idx1 = policy.select_worker_with_headers(&workers, Some(body), None);
        let idx2 = policy.select_worker_with_headers(&workers, Some(body), None);
        assert_eq!(idx1, idx2);
        assert_eq!(policy.active_session_count(), 1);

        policy.finish_session("body-sess");
        assert_eq!(policy.active_session_count(), 0);
    }

    #[test]
    fn test_stateless_requests_not_tracked() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let body = r#"{"prompt": "no session here"}"#;

        let routed: Vec<_> = (0..4)
            .map(|_| policy.select_worker_with_headers(&workers, Some(body), None))
            .collect();
        assert_eq!(routed, vec![Some(0), Some(1), Some(0), Some(1)]);
        // No session identifier -> nothing tracked.
        assert_eq!(policy.active_session_count(), 0);
    }

    #[test]
    fn test_routes_oversized_session_id_without_affinity() {
        let policy = StickyLeastLoadedPolicy::with_limits(7200, 10, 8);
        let workers = make_workers(&["http://w1:8000"]);
        let headers = header_with_session("ninebytes");

        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&headers)),
            Some(0)
        );
        assert_eq!(policy.active_session_count(), 0);
    }

    #[test]
    fn test_routes_new_session_without_affinity_at_capacity() {
        let policy = StickyLeastLoadedPolicy::with_limits(7200, 1, 256);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let admitted = header_with_session("admitted");
        let untracked = header_with_session("untracked");

        let original = policy.select_worker_with_headers(&workers, None, Some(&admitted));
        assert!(original.is_some());
        let first = policy.select_worker_with_headers(&workers, None, Some(&untracked));
        let second = policy.select_worker_with_headers(&workers, None, Some(&untracked));
        assert!(first.is_some());
        assert!(second.is_some());
        assert_ne!(first, second);
        assert_ne!(first, original);
        assert_ne!(second, original);
        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&admitted)),
            original
        );
        assert_eq!(policy.active_session_count(), 1);
    }

    #[test]
    fn test_reassign_when_replica_unhealthy() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = header_with_session("sess-move");

        let idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        let original_url = workers[idx].url().to_string();

        // Mark the assigned replica unhealthy; the session must be reassigned to
        // the remaining healthy replica.
        workers[idx].set_healthy(false);
        let new_idx = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_ne!(workers[new_idx].url(), original_url);
        assert!(workers[new_idx].is_healthy());

        // Old replica should no longer hold the session.
        assert_eq!(policy.active_count_for(&original_url), 0);
        assert_eq!(policy.active_count_for(workers[new_idx].url()), 1);
    }

    #[test]
    fn test_expired_session_is_swept() {
        // Zero expiration => any session is immediately eligible for expiry, but
        // the sweep only runs every SWEEP_INTERVAL_SECS. Force the sweep by
        // back-dating last_sweep.
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(0);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = header_with_session("sess-ttl");

        policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_eq!(policy.active_session_count(), 1);

        {
            let mut state = policy.state.lock().unwrap();
            state.last_sweep = Instant::now() - Duration::from_secs(SWEEP_INTERVAL_SECS + 1);
        }

        // A subsequent request triggers the sweep, removing the expired session
        // and creating a fresh one.
        let other = header_with_session("sess-other");
        policy
            .select_worker_with_headers(&workers, None, Some(&other))
            .unwrap();
        // The expired session was swept; only the freshly created one remains.
        assert_eq!(policy.active_session_count(), 1);
        assert_eq!(
            policy.active_count_for("http://w1:8000") + policy.active_count_for("http://w2:8000"),
            1
        );
    }

    #[test]
    fn test_capacity_check_sweeps_expired_sessions() {
        let policy = StickyLeastLoadedPolicy::with_limits(0, 1, 256);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let expired = header_with_session("expired");
        let fresh = header_with_session("fresh");

        policy
            .select_worker_with_headers(&workers, None, Some(&expired))
            .unwrap();
        assert_eq!(policy.active_session_count(), 1);

        policy
            .select_worker_with_headers(&workers, None, Some(&fresh))
            .unwrap();
        assert_eq!(policy.active_session_count(), 1);

        policy.finish_session("fresh");
        assert_eq!(policy.active_session_count(), 0);
    }

    #[test]
    fn test_concurrent_new_sessions_reserve_balanced_capacity() {
        let policy = Arc::new(StickyLeastLoadedPolicy::new());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let barrier = std::sync::Barrier::new(60);
        std::thread::scope(|scope| {
            for i in 0..60 {
                let policy = &policy;
                let workers = &workers;
                let barrier = &barrier;
                scope.spawn(move || {
                    let headers = header_with_session(&format!("session-{i}"));
                    barrier.wait();
                    let first = policy.select_worker_with_headers(workers, None, Some(&headers));
                    assert_eq!(
                        first,
                        policy.select_worker_with_headers(workers, None, Some(&headers))
                    );
                });
            }
        });
        assert_eq!(policy.active_session_count(), 60);
        for worker in &workers {
            assert_eq!(policy.active_count_for(worker.url()), 60 / workers.len());
        }
    }

    #[test]
    fn test_expired_session_rebinds_before_next_sweep() {
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(1);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        for session in ["expired", "active"] {
            policy.select_worker_with_headers(
                &workers[..1],
                None,
                Some(&header_with_session(session)),
            );
        }
        policy
            .state
            .lock()
            .unwrap()
            .sessions
            .get_mut("expired")
            .unwrap()
            .last_access = Instant::now() - Duration::from_secs(2);

        assert_eq!(
            policy.select_worker_with_headers(
                &workers,
                None,
                Some(&header_with_session("expired"))
            ),
            Some(1),
            "expired affinity must not be refreshed before the global sweep"
        );
        assert_eq!(policy.active_count_for(workers[0].url()), 1);
        assert_eq!(policy.active_count_for(workers[1].url()), 1);
    }

    /// Two sessions pinned to the first of two replicas, so a reassignment would
    /// go to the second.
    fn pinned_session_policy() -> (Arc<StickyLeastLoadedPolicy>, Vec<Arc<dyn Worker>>) {
        let policy = Arc::new(StickyLeastLoadedPolicy::with_expiration_secs(1));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        for session in ["long", "other"] {
            policy.select_worker_with_headers(
                &workers[..1],
                None,
                Some(&header_with_session(session)),
            );
        }
        (policy, workers)
    }

    fn start_request(policy: &Arc<StickyLeastLoadedPolicy>, session: &str) -> SessionRequestGuard {
        let policy: Arc<dyn LoadBalancingPolicy> = policy.clone();
        SessionRequestGuard::new(policy, None, Some(&header_with_session(session))).unwrap()
    }

    fn backdate(policy: &StickyLeastLoadedPolicy, session: &str) {
        policy
            .state
            .lock()
            .unwrap()
            .sessions
            .get_mut(session)
            .unwrap()
            .last_access = Instant::now() - Duration::from_secs(2);
    }

    #[test]
    fn test_in_flight_session_outlives_expiration() {
        let (policy, workers) = pinned_session_policy();
        let _request = start_request(&policy, "long");
        backdate(&policy, "long");
        policy.state.lock().unwrap().last_sweep =
            Instant::now() - Duration::from_secs(SWEEP_INTERVAL_SECS + 1);

        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&header_with_session("long"))),
            Some(0),
            "a session whose request is still generating must keep its replica"
        );
        assert_eq!(policy.active_count_for(workers[0].url()), 2);
    }

    #[test]
    fn test_completion_restarts_idle_clock() {
        let (policy, workers) = pinned_session_policy();
        let request = start_request(&policy, "long");
        backdate(&policy, "long");
        drop(request);

        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&header_with_session("long"))),
            Some(0)
        );
    }

    #[test]
    fn test_idle_session_expires_after_completion() {
        let (policy, workers) = pinned_session_policy();
        drop(start_request(&policy, "long"));
        backdate(&policy, "long");

        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&header_with_session("long"))),
            Some(1)
        );
    }

    #[test]
    fn test_rebind_keeps_in_flight_requests() {
        let (policy, workers) = pinned_session_policy();
        let _request = start_request(&policy, "long");
        workers[0].set_healthy(false);
        policy.select_worker_with_headers(&workers, None, Some(&header_with_session("long")));

        let state = policy.state.lock().unwrap();
        assert_eq!(state.sessions["long"].worker_url, "http://w2:8000");
        assert_eq!(state.sessions["long"].in_flight, 1);
    }

    #[test]
    fn test_repeated_finish_preserves_other_sessions() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000"]);
        for session in ["finished", "active"] {
            policy.select_worker_with_headers(&workers, None, Some(&header_with_session(session)));
        }
        policy.finish_session("finished");
        policy.finish_session("finished");
        assert_eq!(policy.active_session_count(), 1);
        assert_eq!(policy.active_count_for(workers[0].url()), 1);
    }

    #[test]
    fn test_removed_worker_rebinds_session() {
        let policy = StickyLeastLoadedPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = header_with_session("session");
        let original = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        let remaining = vec![workers[1 - original].clone()];
        assert_eq!(
            policy.select_worker_with_headers(&remaining, None, Some(&headers)),
            Some(0)
        );
        assert_eq!(policy.active_count_for(workers[original].url()), 0);
        assert_eq!(policy.active_count_for(remaining[0].url()), 1);
    }

    /// Pin `n` sessions named `{prefix}-{i}` to `workers` (a one-worker slice).
    fn pin(policy: &StickyLeastLoadedPolicy, workers: &[Arc<dyn Worker>], prefix: &str, n: usize) {
        for i in 0..n {
            let headers = header_with_session(&format!("{prefix}-{i}"));
            policy.select_worker_with_headers(workers, None, Some(&headers));
        }
    }

    fn sweep_now(policy: &StickyLeastLoadedPolicy) {
        let mut state = policy.state.lock().unwrap();
        policy.sweep(&mut state, Instant::now());
    }

    fn route(
        policy: &StickyLeastLoadedPolicy,
        workers: &[Arc<dyn Worker>],
        session: &str,
    ) -> usize {
        policy
            .select_worker_with_headers(workers, None, Some(&header_with_session(session)))
            .unwrap()
    }

    #[test]
    fn test_active_window_defaults_to_expiration() {
        let policy = StickyLeastLoadedPolicy::new();
        assert_eq!(policy.active_window, policy.session_expiration);
        assert_eq!(policy.rebalance_ratio, 0.0);

        // Sessions idle for less than the expiration keep counting as load, so
        // placement is unchanged.
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(3600);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        pin(&policy, &workers[..1], "idle", 2);
        pin(&policy, &workers[1..], "busy", 1);
        for session in ["idle-0", "idle-1"] {
            backdate(&policy, session);
        }
        sweep_now(&policy);
        assert_eq!(policy.active_count_for(workers[0].url()), 2);
        assert_eq!(policy.pinned_count_for(workers[0].url()), 2);
        assert_eq!(route(&policy, &workers, "new"), 1);
    }

    #[test]
    fn test_idle_session_stops_counting_after_active_window() {
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(3600).with_active_window_secs(1);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        pin(&policy, &workers[..1], "idle", 2);
        pin(&policy, &workers[1..], "busy", 1);
        for session in ["idle-0", "idle-1"] {
            backdate(&policy, session);
        }
        sweep_now(&policy);

        assert_eq!(policy.active_count_for(workers[0].url()), 0);
        assert_eq!(policy.pinned_count_for(workers[0].url()), 2);
        assert_eq!(
            route(&policy, &workers, "new"),
            0,
            "idle sessions must not steer new sessions away"
        );

        // An inactive session keeps its replica and counts again once it returns.
        assert_eq!(route(&policy, &workers, "idle-0"), 0);
        assert_eq!(policy.active_count_for(workers[0].url()), 2);
        assert_eq!(policy.pinned_count_for(workers[0].url()), 3);
    }

    #[test]
    fn test_in_flight_session_counts_beyond_active_window() {
        let policy = Arc::new(
            StickyLeastLoadedPolicy::with_expiration_secs(3600).with_active_window_secs(1),
        );
        let workers = make_workers(&["http://w1:8000"]);
        route(&policy, &workers, "long");
        let request = start_request(&policy, "long");
        backdate(&policy, "long");
        sweep_now(&policy);
        assert_eq!(policy.active_count_for(workers[0].url()), 1);

        // Completion restarts the window; going idle afterwards deactivates.
        drop(request);
        sweep_now(&policy);
        assert_eq!(policy.active_count_for(workers[0].url()), 1);
        backdate(&policy, "long");
        sweep_now(&policy);
        assert_eq!(policy.active_count_for(workers[0].url()), 0);
        assert_eq!(policy.pinned_count_for(workers[0].url()), 1);

        // A request in flight counts as load even without a new routing decision.
        let _request = start_request(&policy, "long");
        assert_eq!(policy.active_count_for(workers[0].url()), 1);
    }

    #[test]
    fn test_expiry_releases_inactive_session() {
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(1).with_active_window_secs(0);
        let workers = make_workers(&["http://w1:8000"]);
        route(&policy, &workers, "idle");
        sweep_now(&policy);
        assert_eq!(policy.active_count_for(workers[0].url()), 0);
        assert_eq!(policy.pinned_count_for(workers[0].url()), 1);

        backdate(&policy, "idle");
        sweep_now(&policy);
        assert_eq!(policy.active_session_count(), 0);
        assert_eq!(policy.active_count_for(workers[0].url()), 0);
        assert_eq!(policy.pinned_count_for(workers[0].url()), 0);
    }

    #[test]
    fn test_rebalance_off_by_default() {
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(3600);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        pin(&policy, &workers[..1], "s", 10);
        assert_eq!(route(&policy, &workers, "s-0"), 0);
        assert_eq!(policy.active_count_for(workers[0].url()), 10);
    }

    #[test]
    fn test_rebalance_moves_idle_session_off_overloaded_replica() {
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(3600).with_rebalance_ratio(1.5);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        pin(&policy, &workers[..1], "s", 6);

        // 5 other sessions vs 0: moved.
        assert_eq!(route(&policy, &workers, "s-0"), 1);
        assert_eq!(policy.active_count_for(workers[0].url()), 5);
        assert_eq!(policy.active_count_for(workers[1].url()), 1);
        assert_eq!(route(&policy, &workers, "s-0"), 1, "moved session sticks");

        // 4 other sessions vs 1: below the minimum gap, stays.
        assert_eq!(route(&policy, &workers, "s-1"), 0);
        assert_eq!(policy.pinned_count_for(workers[0].url()), 5);
    }

    #[test]
    fn test_rebalance_ratio_threshold() {
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(3600).with_rebalance_ratio(1.5);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        pin(&policy, &workers[1..], "light", 8);
        pin(&policy, &workers[..1], "heavy", 13);

        // 12 other sessions vs 8 is not more than 1.5x: stays.
        assert_eq!(route(&policy, &workers, "heavy-0"), 0);

        // 13 vs 8 is: moves.
        pin(&policy, &workers[..1], "extra", 1);
        assert_eq!(route(&policy, &workers, "heavy-0"), 1);
        assert_eq!(policy.active_count_for(workers[0].url()), 13);
        assert_eq!(policy.active_count_for(workers[1].url()), 9);
    }

    #[test]
    fn test_rebalance_counts_only_active_sessions() {
        let policy = StickyLeastLoadedPolicy::with_expiration_secs(3600)
            .with_active_window_secs(1)
            .with_rebalance_ratio(1.5);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        pin(&policy, &workers[..1], "s", 6);
        for i in 1..6 {
            backdate(&policy, &format!("s-{i}"));
        }
        sweep_now(&policy);
        assert_eq!(route(&policy, &workers, "s-0"), 0);
    }

    #[test]
    fn test_no_rebalance_while_in_flight() {
        let policy =
            Arc::new(StickyLeastLoadedPolicy::with_expiration_secs(3600).with_rebalance_ratio(1.5));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        pin(&policy, &workers[..1], "s", 6);

        let request = start_request(&policy, "s-0");
        assert_eq!(route(&policy, &workers, "s-0"), 0);
        drop(request);
        assert_eq!(route(&policy, &workers, "s-0"), 1);
    }

    #[test]
    fn test_invalid_rebalance_ratio_disables_rebalancing() {
        for ratio in [-1.0, f64::NAN, f64::INFINITY] {
            let policy =
                StickyLeastLoadedPolicy::with_expiration_secs(3600).with_rebalance_ratio(ratio);
            assert_eq!(policy.rebalance_ratio, 0.0);
        }
    }
}
