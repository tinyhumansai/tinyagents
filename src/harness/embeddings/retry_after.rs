//! Retry-After parsing and bounded exponential backoff for embedding providers.

use chrono::{DateTime, Utc};

pub const MAX_RETRIES: u32 = 3;
pub const BASE_BACKOFF_MS: u64 = 1_000;
pub const MAX_BACKOFF_MS: u64 = 30_000;

pub fn parse_retry_after_ms(value: Option<&str>) -> Option<u64> {
    parse_retry_after_ms_at(value, Utc::now())
}

fn parse_retry_after_ms_at(value: Option<&str>, now: DateTime<Utc>) -> Option<u64> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(seconds.saturating_mul(1_000).min(MAX_BACKOFF_MS));
    }
    let retry_at = DateTime::parse_from_rfc2822(value)
        .ok()?
        .with_timezone(&Utc);
    let delay = retry_at.signed_duration_since(now).num_milliseconds();
    if delay <= 0 {
        return Some(0);
    }
    u64::try_from(delay).ok().map(|ms| ms.min(MAX_BACKOFF_MS))
}

pub fn backoff_ms_for_attempt(attempt: u32, retry_after: Option<&str>) -> u64 {
    parse_retry_after_ms(retry_after).unwrap_or_else(|| {
        BASE_BACKOFF_MS
            .saturating_mul(2u64.saturating_pow(attempt))
            .min(MAX_BACKOFF_MS)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_delta_seconds_and_caps() {
        assert_eq!(parse_retry_after_ms(Some(" 5 ")), Some(5_000));
        assert_eq!(parse_retry_after_ms(Some("999")), Some(MAX_BACKOFF_MS));
        assert_eq!(parse_retry_after_ms(Some("-1")), None);
    }

    #[test]
    fn parses_http_dates() {
        let now = DateTime::parse_from_rfc2822("Wed, 21 Oct 2015 07:27:55 GMT")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(
            parse_retry_after_ms_at(Some("Wed, 21 Oct 2015 07:28:00 GMT"), now),
            Some(5_000)
        );
    }

    #[test]
    fn falls_back_to_bounded_exponential_backoff() {
        assert_eq!(backoff_ms_for_attempt(0, None), 1_000);
        assert_eq!(backoff_ms_for_attempt(2, None), 4_000);
        assert_eq!(backoff_ms_for_attempt(20, None), MAX_BACKOFF_MS);
    }
}
