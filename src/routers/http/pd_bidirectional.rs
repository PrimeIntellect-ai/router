//! Bidirectional KV transfer for multi-turn sessions on the vLLM P/D path (NIXL).
//!
//! When vLLM's NIXL connector runs with
//! `kv_connector_extra_config.bidirectional_kv_xfer=true`, a decode instance keeps
//! the blocks of a finished request for `decoder_kv_blocks_ttl` seconds (480 by
//! default) and returns a handle to them in the response's `kv_transfer_params`
//! (`do_remote_decode=true`, `remote_block_ids`, `remote_num_tokens`, ...). When
//! the same session sends its next turn, the router hands that handle to prefill,
//! which pulls the previous turn's KV (prompt and completion) from decode instead
//! of recomputing it, computes only the new tokens, and then serves decode as
//! usual.
//!
//! Two rules keep this correct:
//!
//! * **Exact token prefix.** Prefill uses the pulled KV for the first
//!   `remote_num_tokens` positions of its prompt without checking which tokens
//!   produced it. A handle is therefore only used when the new prompt starts with
//!   exactly the tokens the handle covers (previous prompt + completion), which
//!   the router checks with a fingerprint of those token ids. This needs token-in
//!   requests (`/inference/v1/generate`); text requests are re-tokenized and chat
//!   templates often re-render history, so they never use handles.
//! * **Same request context.** The KV also depends on inputs beyond the token
//!   ids: the served model (which selects a LoRA adapter) and the prefix-cache
//!   salt are part of the handle's fingerprint, and multimodal requests
//!   (`features`, `content_parts`: identical placeholder tokens can carry
//!   different images) never capture or use a handle.
//! * **Single use.** The owner frees the blocks as soon as the reader reports the
//!   read, so a second read of the same handle (a retried turn, a client retry
//!   after an error, two in-flight requests of one session) returns freed memory
//!   that may already hold another request's KV. The cache removes an entry when a
//!   turn starts ([`BidirectionalKvCache::take`]), whatever happens to that turn,
//!   and retries never carry a handle.

use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

/// Outcome of looking up a session's decode handle at the start of a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleLookup {
    /// A live handle matching the prompt was found and removed from the cache.
    Hit,
    /// No handle is cached for the session.
    Miss,
    /// A handle was cached but is older than the TTL; it was dropped.
    Expired,
    /// A handle was cached but the prompt does not start with its tokens; dropped.
    PrefixMismatch,
    /// A handle was cached for a different model or cache salt; dropped.
    RequestMismatch,
}

impl HandleLookup {
    pub fn as_str(&self) -> &'static str {
        match self {
            HandleLookup::Hit => "decode_blocks",
            HandleLookup::Miss => "none",
            HandleLookup::Expired => "expired",
            HandleLookup::PrefixMismatch => "prefix_mismatch",
            HandleLookup::RequestMismatch => "request_mismatch",
        }
    }
}

/// Fingerprint of the token ids a decode handle covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenPrefix {
    len: usize,
    hash: u64,
}

impl TokenPrefix {
    pub fn of(tokens: &[i64]) -> Self {
        Self {
            len: tokens.len(),
            hash: hash_tokens(tokens),
        }
    }

    /// Whether `prompt` starts with exactly these tokens and has at least one
    /// more (prefill must compute something).
    pub fn is_prefix_of(&self, prompt: &[i64]) -> bool {
        prompt.len() > self.len && hash_tokens(&prompt[..self.len]) == self.hash
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Fingerprint of the request inputs besides the token ids that change the KV
/// of the same tokens: `model` (selects a LoRA adapter) and `cache_salt`
/// (prefix-cache isolation). Absent and `null` are the same.
pub fn request_key(request: &Value) -> u64 {
    let field = |name: &str| {
        request
            .get(name)
            .filter(|v| !v.is_null())
            .map(Value::to_string)
    };
    let mut h = DefaultHasher::new();
    field("model").hash(&mut h);
    field("cache_salt").hash(&mut h);
    h.finish()
}

/// Whether the request carries multimodal inputs (`features` or
/// `content_parts`). Identical placeholder tokens can then stand for different
/// images, so the token ids do not determine the KV and handles are not used.
pub fn has_multimodal_inputs(request: &Value) -> bool {
    ["features", "content_parts"]
        .iter()
        .any(|name| match request.get(*name) {
            None | Some(Value::Null) => false,
            Some(Value::Array(a)) => !a.is_empty(),
            Some(Value::Object(o)) => !o.is_empty(),
            Some(Value::String(s)) => !s.is_empty(),
            Some(_) => true,
        })
}

fn hash_tokens(tokens: &[i64]) -> u64 {
    let mut h = DefaultHasher::new();
    tokens.hash(&mut h);
    h.finish()
}

#[derive(Debug)]
struct Entry {
    params: Value,
    prefix: TokenPrefix,
    request_key: u64,
    stored_at: Instant,
}

/// Per-session cache of decode-side KV handles, each usable once.
#[derive(Debug)]
pub struct BidirectionalKvCache {
    ttl: Duration,
    max_sessions: usize,
    entries: Mutex<HashMap<String, Entry>>,
}

impl BidirectionalKvCache {
    /// `ttl` must stay below the decode engines' `decoder_kv_blocks_ttl`: after
    /// that, decode releases the blocks and a read would see reused memory (vLLM
    /// also declines reads close to the deadline it exports).
    pub fn new(ttl: Duration, max_sessions: usize) -> Self {
        Self {
            ttl,
            max_sessions: max_sessions.max(1),
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Remove the session's handle and return it if it is within the TTL, was
    /// captured with the same [`request_key`], and `prompt` starts with the
    /// tokens it covers. The entry is removed in every case, so a handle is never
    /// handed out twice.
    pub fn take(
        &self,
        session_id: &str,
        request_key: u64,
        prompt: &[i64],
    ) -> (Option<Value>, HandleLookup) {
        self.take_at(session_id, request_key, prompt, Instant::now())
    }

    fn take_at(
        &self,
        session_id: &str,
        request_key: u64,
        prompt: &[i64],
        now: Instant,
    ) -> (Option<Value>, HandleLookup) {
        let Some(e) = self.entries.lock().remove(session_id) else {
            return (None, HandleLookup::Miss);
        };
        if now.saturating_duration_since(e.stored_at) > self.ttl {
            (None, HandleLookup::Expired)
        } else if e.request_key != request_key {
            (None, HandleLookup::RequestMismatch)
        } else if !e.prefix.is_prefix_of(prompt) {
            (None, HandleLookup::PrefixMismatch)
        } else {
            (Some(e.params), HandleLookup::Hit)
        }
    }

    /// Store the handle decode returned for the session's next turn, replacing
    /// any older one. Returns `false` when the cache is full of live entries.
    pub fn put(
        &self,
        session_id: &str,
        params: Value,
        prefix: TokenPrefix,
        request_key: u64,
    ) -> bool {
        self.put_at(session_id, params, prefix, request_key, Instant::now())
    }

    fn put_at(
        &self,
        session_id: &str,
        params: Value,
        prefix: TokenPrefix,
        request_key: u64,
        now: Instant,
    ) -> bool {
        let mut entries = self.entries.lock();
        if entries.len() >= self.max_sessions && !entries.contains_key(session_id) {
            let ttl = self.ttl;
            entries.retain(|_, e| now.saturating_duration_since(e.stored_at) <= ttl);
            if entries.len() >= self.max_sessions {
                return false;
            }
        }
        entries.insert(
            session_id.to_string(),
            Entry {
                params,
                prefix,
                request_key,
                stored_at: now,
            },
        );
        true
    }

    /// Drop the session's handle (e.g. the session finished). Returns whether one was cached.
    pub fn remove(&self, session_id: &str) -> bool {
        self.entries.lock().remove(session_id).is_some()
    }

    /// Drop expired handles. Returns how many were removed.
    pub fn sweep(&self) -> usize {
        self.sweep_at(Instant::now())
    }

    fn sweep_at(&self, now: Instant) -> usize {
        let ttl = self.ttl;
        let mut entries = self.entries.lock();
        let before = entries.len();
        entries.retain(|_, e| now.saturating_duration_since(e.stored_at) <= ttl);
        before - entries.len()
    }

    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }
}

/// Token ids of a token-in request (`token_ids`), or `None` if absent or malformed.
pub fn request_token_ids(request: &Value) -> Option<Vec<i64>> {
    request
        .get("token_ids")?
        .as_array()?
        .iter()
        .map(Value::as_i64)
        .collect()
}

/// Prefill-side `kv_transfer_params` that make prefill read the previous turn's
/// blocks from decode and then serve this turn's decode as usual.
pub fn prefill_params_from_decode_handle(handle: &Value) -> Value {
    let mut params = handle.clone();
    if let Some(obj) = params.as_object_mut() {
        obj.insert("do_remote_decode".to_string(), json!(true));
        obj.insert("do_remote_prefill".to_string(), json!(false));
    }
    params
}

/// Extract a cacheable handle from decode's response.
///
/// `params` is the response's `kv_transfer_params`; `prompt` and `completion` are
/// the turn's prompt token ids and the generated token ids of its only choice.
/// vLLM marks blocks retained on decode with `do_remote_decode=true` and reports
/// how many tokens they cover in `remote_num_tokens` (prompt plus completion
/// minus the last sampled token); prefill also needs the block ids and the
/// engine, request and side-channel address. A loopback or unset `remote_host`
/// (vLLM's default side-channel host is `localhost`) is replaced by the host of
/// `decode_url`, the address the router reached decode on; an explicitly
/// configured host is kept.
pub fn decode_handle_from_response(
    params: &Value,
    decode_url: &str,
    prompt: &[i64],
    completion: &[i64],
) -> Option<(Value, TokenPrefix)> {
    let obj = params.as_object()?;
    if !obj
        .get("do_remote_decode")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }
    if !has_blocks(obj.get("remote_block_ids")?) {
        return None;
    }
    for key in ["remote_engine_id", "remote_request_id", "remote_port"] {
        if obj.get(key).is_none_or(Value::is_null) {
            return None;
        }
    }
    let covered = usize::try_from(obj.get("remote_num_tokens")?.as_u64()?).ok()?;
    if covered == 0 || covered > prompt.len() + completion.len() {
        return None;
    }
    let prefix = if covered <= prompt.len() {
        TokenPrefix::of(&prompt[..covered])
    } else {
        let mut tokens = Vec::with_capacity(covered);
        tokens.extend_from_slice(prompt);
        tokens.extend_from_slice(&completion[..covered - prompt.len()]);
        TokenPrefix::of(&tokens)
    };
    let mut handle = params.clone();
    let reported = obj.get("remote_host").and_then(Value::as_str).unwrap_or("");
    if is_loopback_or_unset(reported) {
        let host = url_host(decode_url)?;
        handle["remote_host"] = json!(host);
    }
    Some((handle, prefix))
}

/// `remote_block_ids` is either a flat list of block ids or one list per KV
/// cache group (hybrid models); a handle is useful if any group has a block.
fn has_blocks(blocks: &Value) -> bool {
    match blocks {
        Value::Array(items) => items.iter().any(|item| match item {
            Value::Array(group) => !group.is_empty(),
            Value::Number(_) => true,
            _ => false,
        }),
        _ => false,
    }
}

fn is_loopback_or_unset(host: &str) -> bool {
    matches!(host, "" | "localhost" | "0.0.0.0" | "::" | "::1" | "[::1]")
        || host.starts_with("127.")
}

fn url_host(url: &str) -> Option<String> {
    match url::Url::parse(url).ok()?.host()? {
        url::Host::Domain(d) => Some(d.to_string()),
        url::Host::Ipv4(ip) => Some(ip.to_string()),
        url::Host::Ipv6(ip) => Some(ip.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(host: &str) -> Value {
        json!({
            "do_remote_prefill": false,
            "do_remote_decode": true,
            "remote_block_ids": [[1, 2, 3], [], [7]],
            "remote_engine_id": "engine-d0",
            "remote_request_id": "req-1",
            "remote_host": host,
            "remote_port": 5600,
            "remote_num_tokens": 5,
            "tp_size": 1
        })
    }

    const PREV: [i64; 5] = [1, 2, 3, 4, 5];
    const NEXT: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    const KEY: u64 = 7;

    fn prefix() -> TokenPrefix {
        TokenPrefix::of(&PREV)
    }

    #[test]
    fn take_is_single_use() {
        let cache = BidirectionalKvCache::new(Duration::from_secs(450), 16);
        assert!(cache.put("s", handle("10.0.0.2"), prefix(), KEY));
        let (first, lookup) = cache.take("s", KEY, &NEXT);
        assert_eq!(lookup, HandleLookup::Hit);
        assert_eq!(first.unwrap()["remote_request_id"], "req-1");
        // A retry of the same turn, or a concurrent request of the session, must not
        // get the same handle again: decode already freed those blocks.
        let (second, lookup) = cache.take("s", KEY, &NEXT);
        assert!(second.is_none());
        assert_eq!(lookup, HandleLookup::Miss);
        assert!(cache.is_empty());
    }

    #[test]
    fn handle_needs_exact_token_prefix() {
        let cache = BidirectionalKvCache::new(Duration::from_secs(450), 16);
        // history re-rendered differently: same length, different tokens
        cache.put("s", handle("h"), prefix(), KEY);
        let (h, lookup) = cache.take("s", KEY, &[1, 2, 9, 4, 5, 6]);
        assert!(h.is_none());
        assert_eq!(lookup, HandleLookup::PrefixMismatch);
        assert!(cache.is_empty(), "a mismatched handle is dropped too");
        // prompt not longer than the covered tokens: prefill would have nothing to compute
        cache.put("s", handle("h"), prefix(), KEY);
        assert_eq!(cache.take("s", KEY, &PREV).1, HandleLookup::PrefixMismatch);
        cache.put("s", handle("h"), prefix(), KEY);
        assert_eq!(
            cache.take("s", KEY, &[1, 2, 3]).1,
            HandleLookup::PrefixMismatch
        );
        cache.put("s", handle("h"), prefix(), KEY);
        assert_eq!(
            cache.take("s", KEY, &[1, 2, 3, 4, 5, 0]).1,
            HandleLookup::Hit
        );
    }

    #[test]
    fn handle_needs_same_request_key() {
        let cache = BidirectionalKvCache::new(Duration::from_secs(450), 16);
        cache.put("s", handle("h"), prefix(), KEY);
        let (h, lookup) = cache.take("s", KEY + 1, &NEXT);
        assert!(h.is_none());
        assert_eq!(lookup, HandleLookup::RequestMismatch);
        assert!(cache.is_empty(), "a mismatched handle is dropped too");
    }

    #[test]
    fn request_key_covers_model_and_cache_salt() {
        let key = |v: Value| request_key(&v);
        let base = key(json!({"token_ids": [1], "model": "m", "cache_salt": "v1"}));
        assert_eq!(
            base,
            key(json!({"model": "m", "cache_salt": "v1", "token_ids": [2]}))
        );
        assert_ne!(base, key(json!({"model": "m", "cache_salt": "v2"})));
        assert_ne!(base, key(json!({"model": "lora-a", "cache_salt": "v1"})));
        assert_ne!(base, key(json!({"model": "m"})));
        // absent and null are the same
        assert_eq!(
            key(json!({})),
            key(json!({"model": null, "cache_salt": null}))
        );
        assert_eq!(
            key(json!({"model": "m"})),
            key(json!({"model": "m", "cache_salt": null}))
        );
        // salt and model values do not collide across fields
        assert_ne!(key(json!({"model": "x"})), key(json!({"cache_salt": "x"})));
    }

    #[test]
    fn multimodal_inputs_detected() {
        assert!(!has_multimodal_inputs(&json!({"token_ids": [1]})));
        for empty in [json!(null), json!([]), json!({}), json!("")] {
            assert!(!has_multimodal_inputs(&json!({"features": empty.clone()})));
            assert!(!has_multimodal_inputs(&json!({"content_parts": empty})));
        }
        assert!(has_multimodal_inputs(
            &json!({"features": {"mm_hashes": ["a"]}})
        ));
        assert!(has_multimodal_inputs(
            &json!({"features": [{"kind": "image"}]})
        ));
        assert!(has_multimodal_inputs(
            &json!({"content_parts": [{"type": "image_url"}]})
        ));
    }

    #[test]
    fn expired_handles_are_dropped_not_returned() {
        let cache = BidirectionalKvCache::new(Duration::from_secs(10), 16);
        let t0 = Instant::now();
        assert!(cache.put_at("s", handle("10.0.0.2"), prefix(), KEY, t0));
        let (h, lookup) = cache.take_at("s", KEY, &NEXT, t0 + Duration::from_secs(11));
        assert!(h.is_none());
        assert_eq!(lookup, HandleLookup::Expired);
        assert!(cache.is_empty());
    }

    #[test]
    fn put_replaces_and_sweep_expires() {
        let cache = BidirectionalKvCache::new(Duration::from_secs(10), 16);
        let t0 = Instant::now();
        cache.put_at("a", handle("10.0.0.2"), prefix(), KEY, t0);
        let mut newer = handle("10.0.0.2");
        newer["remote_request_id"] = json!("req-2");
        cache.put_at("a", newer, prefix(), KEY, t0 + Duration::from_secs(5));
        cache.put_at("b", handle("10.0.0.3"), prefix(), KEY, t0);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.sweep_at(t0 + Duration::from_secs(12)), 1);
        let (h, _) = cache.take_at("a", KEY, &NEXT, t0 + Duration::from_secs(12));
        assert_eq!(h.unwrap()["remote_request_id"], "req-2");
    }

    #[test]
    fn full_cache_evicts_expired_then_refuses() {
        let cache = BidirectionalKvCache::new(Duration::from_secs(10), 2);
        let t0 = Instant::now();
        assert!(cache.put_at("a", handle("h"), prefix(), KEY, t0));
        assert!(cache.put_at("b", handle("h"), prefix(), KEY, t0 + Duration::from_secs(5)));
        // "a" is expired at t0+11 and makes room.
        assert!(cache.put_at(
            "c",
            handle("h"),
            prefix(),
            KEY,
            t0 + Duration::from_secs(11)
        ));
        // Both remaining entries are live: refuse rather than evict a live handle.
        assert!(!cache.put_at(
            "d",
            handle("h"),
            prefix(),
            KEY,
            t0 + Duration::from_secs(12)
        ));
        // Replacing an existing session's handle is always allowed.
        assert!(cache.put_at(
            "b",
            handle("h"),
            prefix(),
            KEY,
            t0 + Duration::from_secs(12)
        ));
        assert!(cache.remove("c"));
        assert!(!cache.remove("c"));
    }

    #[test]
    fn prefill_params_read_from_decode() {
        let p = prefill_params_from_decode_handle(&handle("10.0.0.2"));
        assert_eq!(p["do_remote_decode"], true);
        assert_eq!(p["do_remote_prefill"], false);
        assert_eq!(p["remote_block_ids"], json!([[1, 2, 3], [], [7]]));
        assert_eq!(p["remote_engine_id"], "engine-d0");
        assert_eq!(p["remote_host"], "10.0.0.2");
    }

    #[test]
    fn decode_handle_covers_prompt_and_completion() {
        // 3 prompt tokens + 3 generated; decode reports 5 computed tokens
        let (h, p) =
            decode_handle_from_response(&handle("10.0.0.2"), "http://d:1", &[1, 2, 3], &[4, 5, 6])
                .unwrap();
        assert_eq!(h["remote_request_id"], "req-1");
        assert_eq!(p, TokenPrefix::of(&PREV));
        assert!(p.is_prefix_of(&NEXT));
        // covered within the prompt
        let mut short = handle("10.0.0.2");
        short["remote_num_tokens"] = json!(2);
        let (_, p) = decode_handle_from_response(&short, "http://d:1", &[1, 2, 3], &[4]).unwrap();
        assert_eq!(p, TokenPrefix::of(&[1, 2]));
        // more covered tokens than the router has seen: cannot verify, not cached
        assert!(
            decode_handle_from_response(&handle("10.0.0.2"), "http://d:1", &[1, 2], &[3]).is_none()
        );
        let mut none = handle("10.0.0.2");
        none["remote_num_tokens"] = json!(0);
        assert!(decode_handle_from_response(&none, "http://d:1", &PREV, &[6]).is_none());
        none.as_object_mut().unwrap().remove("remote_num_tokens");
        assert!(decode_handle_from_response(&none, "http://d:1", &PREV, &[6]).is_none());
    }

    #[test]
    fn decode_handle_rewrites_loopback_host_only() {
        let host = |reported: &str, url: &str| {
            decode_handle_from_response(&handle(reported), url, &PREV, &[6])
                .unwrap()
                .0["remote_host"]
                .clone()
        };
        assert_eq!(host("localhost", "http://decode-3:8102"), "decode-3");
        assert_eq!(host("127.0.0.1", "http://10.1.2.3:8100"), "10.1.2.3");
        assert_eq!(host("", "http://[fd00::5]:8100"), "fd00::5");
        assert_eq!(host("192.168.7.9", "http://decode-3:8102"), "192.168.7.9");
    }

    #[test]
    fn decode_handle_requires_retained_blocks() {
        let parse = |p: &Value| decode_handle_from_response(p, "http://d:1", &PREV, &[6]);
        // Prefill-side params (do_remote_prefill=true) are not decode handles.
        let mut p = handle("10.0.0.2");
        p["do_remote_decode"] = json!(false);
        p["do_remote_prefill"] = json!(true);
        assert!(parse(&p).is_none());
        // Engines without bidirectional transfer return no blocks.
        let mut p = handle("10.0.0.2");
        p["remote_block_ids"] = json!([[], []]);
        assert!(parse(&p).is_none());
        let mut p = handle("10.0.0.2");
        p["remote_block_ids"] = Value::Null;
        assert!(parse(&p).is_none());
        let mut p = handle("10.0.0.2");
        p["remote_engine_id"] = Value::Null;
        assert!(parse(&p).is_none());
        assert!(parse(&json!(null)).is_none());
        // Flat block lists (single KV cache group) are accepted.
        let mut p = handle("10.0.0.2");
        p["remote_block_ids"] = json!([4, 5]);
        assert!(parse(&p).is_some());
    }

    #[test]
    fn request_token_ids_parses_token_in_requests_only() {
        assert_eq!(
            request_token_ids(&json!({"token_ids": [1, 2, 3]})),
            Some(vec![1, 2, 3])
        );
        assert_eq!(request_token_ids(&json!({"prompt": "hi"})), None);
        assert_eq!(request_token_ids(&json!({"token_ids": [1, "x"]})), None);
    }
}
