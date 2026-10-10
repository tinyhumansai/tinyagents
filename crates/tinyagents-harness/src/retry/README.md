# harness::retry

Retry, fallback, and rate-limiting policy implementations for the harness's
model calls.

## Why this exists

Every level of the recursive harness — a top-level agent, a nested sub-agent,
a graph node calling out to a model — ultimately reduces to the same kind of
call: a request to a model provider. This module supplies the durability
policies applied uniformly at that one call site, so a flaky provider does not
collapse a deep recursion and a rate-limited provider does not burn every
retry attempt back-to-back.

Three independent policies live here:

- [`RetryPolicy`] — exponential backoff with optional jitter, a per-call
  attempt cap, and server-`Retry-After` awareness.
- [`FallbackPolicy`] — an ordered list of model names to try in sequence.
- [`RateLimiter`] — a token-bucket limiter for pacing provider calls.

A free function, [`is_retryable`], classifies a [`crate::TinyAgentsError`] so
callers can decide whether to retry or propagate immediately; provider error
taxonomy and `Retry-After` parsing are consumed directly from
`tinyinference_llm::failure` rather than reimplemented here.

## Public surface

- [`RetryPolicy`] — builder-style config (`with_max_attempts`,
  `with_initial_backoff_ms`, `with_max_backoff_ms`, `with_multiplier`,
  `with_jitter`, `with_backoff_sleep`, `with_max_retry_after_ms`,
  `with_retry_on`/`with_default_retry_on`) plus the decision/backoff API:
  `should_retry`, `should_retry_error`, `is_retryable_error`,
  `backoff_for_attempt`/`backoff_for_attempt_with`, `backoff_for_error`,
  `sleep_backoff`/`sleep_backoff_for_error`, and
  `max_attempts_capped_at` (reconciling with
  [`crate::limits::RunLimits::max_retries_per_call`]).
- [`RetryPredicate`] — the `Arc<dyn Fn(&TinyAgentsError) -> bool + Send +
  Sync>` type accepted by `RetryPolicy::with_retry_on`.
- [`is_retryable`] — the crate-wide default error classification (see its doc
  table for the per-variant heuristic).
- [`retry_after_hint`] — extracts a server-supplied `Retry-After` delay from an
  error, preferring the structured `ProviderError` field and falling back to
  message-text parsing.
- [`FallbackPolicy`] — `new`, `next_after` (advance to the next model on
  failure).
- [`FailoverReason`], [`decide`], [`FailoverDecision`], [`FailoverState`] —
  reason-aware failover (`failover.rs`): classify *why* a call failed and
  choose retry-same / fallback / surface. See the table below.
- [`RateLimiter`] — `new(capacity, refill_per_sec)`, `try_acquire(tokens,
  now)`, `available(now)`, `capacity`, `refill_per_sec`, `can_ever_acquire`.
- [`JITTER_FRACTION`] — the ± band jitter spreads backoff over.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Method implementations for all three policies, `is_retryable`, `retry_after_hint`. |
| `types.rs` | The declarative structs/type aliases (`RetryPolicy`, `FallbackPolicy`, `RateLimiter`, `RetryPredicate`) plus hand-written `Debug`/`PartialEq` where a boxed closure blocks the derive. |
| `failover.rs` | `FailoverReason` classification (the types themselves live in `types.rs`) (reusing `classify_provider_failure` and the provider-body matchers), the pure `decide` table, `FailoverState`. |
| `failover_tests.rs` | Classification and decision-table tests. |
| `jitter.rs` | A minimal, dependency-free `xorshift64*` RNG used only for backoff jitter spread — never security-relevant, and always bypassable in tests via an explicit `rand01`. |
| `mod_tests.rs` | Backoff growth/capping, jitter scaling/clamping, `should_retry` boundaries, `is_retryable` classification, `FallbackPolicy` traversal, and token-bucket behavior. |

## Reason-aware failover

The agent loop's model call asks `decide(reason, state)` after every failed
attempt; the same reason is logged (`[failover]` target lines) and drives:

| Reason | Decision |
| --- | --- |
| `RateLimit`, `Overloaded`, `Timeout`, `Transport`, `EmptyResponse`, `Unknown` | `RetrySame` while the retry policy calls the error transient and attempts remain, then `Fallback` |
| `Auth`, `Billing`, `ModelNotFound` | `Fallback` at once; no same-model retry |
| `AuthPermanent` (revoked / deactivated / suspended / banned credential) | `Fallback` at once and the model is skipped for the rest of the run (`LimitTracker::skip_model_for_run`; surfaced as `FallbackSkipped`). Fixable or per-project states ("disabled", an endpoint access policy) are plain `Auth` and are not remembered |
| `Format` (4xx without a better cause, adapter `Validation`/`Unsupported`, a 404 whose body does not name a model) | `Fallback`, never retried on the same model. 4xx rejections are often provider-specific (OpenAI strict schema, Gemini `Unknown name`, Anthropic `input_schema`); nothing is provably model-independent, so nothing surfaces |
| `ContextOverflow` | `Fallback` only to a candidate whose profile `max_input_tokens` is strictly larger than the current model's (`FailoverState::larger_window_available`; unknown windows never qualify); otherwise `Surface` — compaction is the remedy |
| Terminal `LimitExceeded` (run/provider budget) | `Surface` immediately; a configured fallback cannot bypass the exhausted budget |
| `StreamIdleTimeout` | `Fallback` after the current model reaches its breaker threshold; a caller's subagent retry policy may retry the whole child attempt |

A custom `RetryPolicy::retry_on` still vetoes retries for the transient
reasons but cannot re-enable same-model retries for the permanent ones.
The skip hint is advisory: a skipped model is still tried when no eligible
fallback remains. A hosted resolver's single decision is never failed over.

## Operational constraints

- **Deterministically testable by construction**: `RateLimiter` methods take
  an explicit `now: Instant`, and `RetryPolicy::backoff_for_attempt_with`
  takes an explicit `rand01: f64`, so neither policy needs a clock/RNG
  injection seam — tests just pass fixed values.
- `RetryPolicy::backoff_sleep` defaults to `true`. Tests and other
  latency-sensitive callers must opt out explicitly with
  `with_backoff_sleep(false)`; leaving it on and calling a retry loop in a
  test will really sleep.
- Jitter is **additive**, never multiplicative — it can only widen the delay
  band around the base backoff, never collapse it toward zero. See
  `JITTER_FRACTION` and `RetryPolicy::backoff_for_attempt_with` for the
  historical bug this guards against.
- A server-supplied `Retry-After` only ever *lengthens* the wait
  (`RetryPolicy::backoff_for_error` takes the `max` of computed backoff and the
  hint), and is clamped by `max_retry_after_ms` so a hostile or buggy header
  cannot park a run indefinitely.
- Callers holding a `RetryPolicy` should always go through
  `RetryPolicy::is_retryable_error` / `should_retry_error` rather than calling
  the free `is_retryable` directly, so a caller-supplied `retry_on` predicate
  is honored.
