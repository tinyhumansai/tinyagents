//! Process-wide endpoint-keyed pacing for outbound embedding requests.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use tokio::time::Instant;

pub const DEFAULT_REQUESTS_PER_MINUTE: u32 = 60;

static CONFIGURED_LIMIT: AtomicU32 = AtomicU32::new(DEFAULT_REQUESTS_PER_MINUTE);
static BUCKETS: OnceLock<Mutex<HashMap<String, Arc<TokenBucket>>>> = OnceLock::new();

pub fn set_rate_limit(per_minute: u32) {
    let previous = CONFIGURED_LIMIT.swap(per_minute, Ordering::Relaxed);
    if previous != per_minute
        && let Some(registry) = BUCKETS.get()
    {
        registry
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }
    tracing::debug!(
        target: "tinyagents::embeddings::rate_limit",
        per_minute,
        "[embeddings] configured outbound request rate"
    );
}

pub fn rate_limit() -> u32 {
    CONFIGURED_LIMIT.load(Ordering::Relaxed)
}

pub async fn acquire(base_url: &str) {
    acquire_with_limit(base_url, rate_limit()).await;
}

async fn acquire_with_limit(base_url: &str, limit: u32) {
    if limit == 0 || is_loopback_url(base_url) {
        return;
    }
    bucket_for(base_url, limit).acquire().await;
}

fn bucket_for(base_url: &str, per_minute: u32) -> Arc<TokenBucket> {
    let registry = BUCKETS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut buckets = registry.lock().unwrap_or_else(PoisonError::into_inner);
    buckets
        .entry(base_url.to_owned())
        .or_insert_with(|| Arc::new(TokenBucket::per_minute(per_minute)))
        .clone()
}

fn is_loopback_url(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };
    url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    })
}

struct TokenBucket {
    state: tokio::sync::Mutex<BucketState>,
    refill_per_second: f64,
}

struct BucketState {
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn per_minute(per_minute: u32) -> Self {
        Self {
            state: tokio::sync::Mutex::new(BucketState {
                tokens: 1.0,
                last_refill: Instant::now(),
            }),
            refill_per_second: f64::from(per_minute.max(1)) / 60.0,
        }
    }

    async fn acquire(&self) {
        loop {
            let wait = {
                let mut state = self.state.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(state.last_refill).as_secs_f64();
                state.last_refill = now;
                refill_and_take(&mut state.tokens, self.refill_per_second, elapsed)
            };
            let Some(wait) = wait else {
                return;
            };
            tracing::debug!(
                target: "tinyagents::embeddings::rate_limit",
                wait_ms = wait.as_millis(),
                "[embeddings] waiting for outbound request slot"
            );
            tokio::time::sleep(wait).await;
        }
    }
}

fn refill_and_take(
    tokens: &mut f64,
    refill_per_second: f64,
    elapsed_seconds: f64,
) -> Option<Duration> {
    *tokens = (*tokens + elapsed_seconds * refill_per_second).min(1.0);
    if *tokens >= 1.0 {
        *tokens -= 1.0;
        None
    } else {
        Some(Duration::from_secs_f64((1.0 - *tokens) / refill_per_second))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_detection_is_fail_closed() {
        assert!(is_loopback_url("http://localhost:11434"));
        assert!(is_loopback_url("http://127.0.0.1:8080"));
        assert!(is_loopback_url("http://[::1]:8080"));
        assert!(!is_loopback_url("https://api.openai.com"));
        assert!(!is_loopback_url("not a url"));
    }

    #[test]
    fn bucket_math_paces_without_bursting() {
        let mut tokens = 1.0;
        assert!(refill_and_take(&mut tokens, 1.0, 0.0).is_none());
        let wait = refill_and_take(&mut tokens, 1.0, 0.25).unwrap();
        assert!((wait.as_secs_f64() - 0.75).abs() < 1e-6);
    }

    #[tokio::test]
    async fn disabled_and_loopback_limits_never_block() {
        acquire_with_limit("https://api.example.com", 0).await;
        acquire_with_limit("http://127.0.0.1:1", 1).await;
    }
}
