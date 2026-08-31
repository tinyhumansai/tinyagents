//! LIVE end-to-end proof for Anthropic prompt caching through the local ladder.
//!
//! The test is opt-in (`PROMPT_CACHE_LIVE=1`) and uses the loopback ladder's
//! Anthropic Messages-compatible endpoint. It sends two different user turns
//! under the same large cacheable system prefix, then requires the second
//! response to report provider cache-read tokens. No credential is logged.

use tinyinference::cache::CachePolicy;
use tinyinference::message::Message;
use tinyinference::model::{ChatModel, ModelRequest, PromptSegment, SegmentRole};
use tinyinference::providers::anthropic::AnthropicModel;

const LADDER_URL: &str = "http://127.0.0.1:6969/v1";

#[tokio::test]
async fn live_ladder_reuses_an_anthropic_prompt_cache_breakpoint() {
    if std::env::var("PROMPT_CACHE_LIVE").as_deref() != Ok("1") {
        eprintln!("skipping live prompt-cache check: set PROMPT_CACHE_LIVE=1");
        return;
    }
    let Ok(api_key) = std::env::var("LADDER_API_KEY") else {
        eprintln!("skipping live prompt-cache check: LADDER_API_KEY is not set");
        return;
    };

    let model = AnthropicModel::with_base_url(api_key, LADDER_URL).with_model("flash");
    // Anthropic caches only prefixes above its provider-specific minimum. This
    // deliberately stays comfortably above the common 1,024-token floor while
    // remaining bounded and deterministic.
    let stable_prefix = format!(
        "cache-test-run-{} ",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos()
    ) + &std::iter::repeat_n(
        "You are a precise test assistant. Preserve these operating rules exactly. ",
        180,
    )
    .collect::<String>();
    let request = |question: &str| {
        ModelRequest::new(vec![
            Message::system(stable_prefix.clone()),
            Message::user(question),
        ])
        .with_cache_segments(vec![
            PromptSegment {
                id: "system".into(),
                role: SegmentRole::System,
                cacheable: true,
            },
            PromptSegment {
                id: "turn".into(),
                role: SegmentRole::Volatile,
                cacheable: false,
            },
        ])
        .with_cache_policy(CachePolicy {
            protect_prompt_prefix: true,
            ..CachePolicy::default()
        })
        .with_timeout_ms(90_000)
        .with_max_tokens(8)
    };

    let first = model
        .invoke(&(), request("Reply with exactly: one"))
        .await
        .expect("first cached-prefix request succeeds");
    let second = model
        .invoke(&(), request("Reply with exactly: two"))
        .await
        .expect("second cached-prefix request succeeds");

    let _first_usage = first.usage.expect("first response reports usage");
    let second_usage = second.usage.expect("second response reports usage");
    assert!(
        second_usage.cache_read_tokens > 0,
        "second request did not reuse the stable prefix (cache read tokens: {})",
        second_usage.cache_read_tokens
    );
}
