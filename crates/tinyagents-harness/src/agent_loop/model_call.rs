//! Model invocation: cache-aware retry/fallback dispatch
//! (`invoke_model_with_retry`, `invoke_model_resolving`), the streaming
//! variant, and the innermost `ModelBaseCall`/`ToolBaseCall` impls that
//! the middleware wrap-onion terminates into.
//!
//! Split out of `agent_loop/mod.rs`; see that module's doc comment for
//! the full loop lifecycle, limits, and backoff design.

/// Timeout-message label for a call bounded by the run's remaining wall-clock
/// budget: the run is out of time, not (necessarily) this call wedged.
pub(super) const RUN_BOUND_LABEL: &str = "remaining wall-clock budget";

/// Timeout-message label for a model call bounded by the per-call ceiling
/// ([`RunLimits::max_model_call_ms`][crate::limits::RunLimits::max_model_call_ms]):
/// this one call ran past the time any single call is allowed, with run time
/// still left.
pub(super) const PER_CALL_BOUND_LABEL: &str = "per-model-call ceiling";

use super::*;
use crate::cache::{CacheSkipReason, apply_prompt_cache_breakpoints, scoped_cache_key};
use crate::no_progress::StreamTextStallDetector;
use crate::retry::{FailoverDecision, FailoverReason, FailoverState, decide};
use tinyinference_llm::cache::CachePolicy;

/// Converts a configured stream window to a [`Duration`]. `None` and a zero
/// window both mean "no bound": a zero-length window would fail every stream
/// before it could deliver anything.
fn positive_window(ms: Option<u64>) -> Option<Duration> {
    ms.filter(|ms| *ms > 0).map(Duration::from_millis)
}

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Previews which model `request` would reach through the local
    /// [`crate::model_registry::ModelRegistry`] and returns its capability
    /// profile, without dispatching anything.
    ///
    /// A pure registry lookup (no network call), so it is cheap enough to run
    /// before every `before_model` pass and before every tool-change patch.
    /// `None` when nothing resolves or the model advertises no profile. A
    /// Hosted runs use their resolver for the profile shown to middleware;
    /// this local preview is only their initial fallback.
    pub(super) fn preview_model_profile(
        &self,
        request: &ModelRequest,
    ) -> Option<tinyinference_llm::model::ModelProfile> {
        self.models
            .resolve_request(request, None, None)
            .and_then(|binding| binding.model.profile().cloned())
    }

    /// Resolves the model binding through the host's routing authority when
    /// this run is a hosted invocation; returns `None` for a plain SDK run so
    /// the caller falls through to local [`crate::model_registry::ModelRegistry`]
    /// resolution instead.
    ///
    /// The first call for a run (`ctx.depth() == 0`) is flagged
    /// [`as_team_lead`][crate::host::ModelResolveRequest::as_team_lead] so the
    /// host can apply lead-specific routing. An explicit `request.model` (or,
    /// failing that, the host binding's own pin) is forwarded as the model
    /// pin, and the request's required capabilities are forwarded so the host
    /// cannot resolve a model that cannot serve this call. Resolution is
    /// bounded by [`Self::model_call_budget`] and races cooperative
    /// cancellation; a `Cancelled`/`Timeout` failure is returned verbatim,
    /// any other host failure is logged and collapsed to a generic
    /// [`TinyAgentsError::Model`] so host-internal detail never leaks into the
    /// run's error surface.
    pub(super) async fn resolve_host_model(
        &self,
        ctx: &RunContext<Ctx>,
        request: &ModelRequest,
    ) -> Result<Option<ResolvedModelBinding<State>>> {
        let Some(host_run) = crate::runtime::host_invocation_binding::<State, Ctx>(ctx)? else {
            return Ok(None);
        };
        let mut resolve = crate::host::ModelResolveRequest::new(host_run.agent_id.clone());
        if ctx.depth() == 0 {
            resolve = resolve.as_team_lead();
        }
        if let Some(role) = host_run.role.clone() {
            resolve = resolve.with_role(role);
        }
        if let Some(pin) = request.model.clone().or(host_run.model_pin.clone()) {
            resolve = resolve.with_model_pin(pin);
        }
        if let Some(capabilities) = request.required_capabilities.clone() {
            resolve = resolve.with_required_capabilities(capabilities);
        }
        let resolution = host_run.host.models.resolve(&resolve);
        let (budget, bound) = self.model_call_budget(ctx);
        // Set only when the wrapper's own deadline fires, so a `Timeout` the
        // resolver itself returned is never mistaken for the per-call ceiling.
        let wrapper_expired = std::sync::atomic::AtomicBool::new(false);
        let model = ctx
            .bounded(budget, resolution, || {
                wrapper_expired.store(true, std::sync::atomic::Ordering::SeqCst);
                format!(
                    "host model resolution for run `{}` exceeded its {bound}",
                    ctx.run_id()
                )
            })
            .await
            .map_err(|error| match error {
                // `RunContext::bounded` always reports `Timeout`; when the
                // per-model-call ceiling was the tighter bound, surface it as
                // `CallTimeout` so the run is not mistaken for out of time.
                TinyAgentsError::Timeout(message)
                    if wrapper_expired.load(std::sync::atomic::Ordering::SeqCst)
                        && bound == PER_CALL_BOUND_LABEL =>
                {
                    TinyAgentsError::CallTimeout(message)
                }
                TinyAgentsError::Cancelled
                | TinyAgentsError::Timeout(_)
                | TinyAgentsError::CallTimeout(_) => error,
                _ => {
                    tracing::warn!(agent_id = %host_run.agent_id, "[host] model resolution failed");
                    TinyAgentsError::Model("host model resolution failed".to_string())
                }
            })?;
        let name = model
            .profile()
            .and_then(|profile| profile.model.clone())
            .unwrap_or_else(|| format!("host:{}", host_run.agent_id));
        Ok(Some(ResolvedModelBinding {
            resolved: ResolvedModel {
                name,
                requested: resolve.model_pin,
                source: if request.model.is_some() {
                    ModelResolutionSource::RequestOverride
                } else {
                    ModelResolutionSource::AgentDefault
                },
            },
            model,
        }))
    }
    /// Invokes a model, consulting the local response cache around the
    /// retry/fallback path.
    ///
    /// When caching is enabled for this call (see
    /// [`Self::response_cache_decision`]) the cache is checked **before** any
    /// provider call: on a hit an [`AgentEvent::CacheHit`] is emitted and the
    /// cached [`ModelResponse`] is returned *without* invoking the underlying
    /// [`ChatModel`] (the retry/fallback path is skipped entirely); on a miss an
    /// [`AgentEvent::CacheMiss`] is emitted, the provider is invoked normally,
    /// and the successful response is written back to the cache.
    ///
    /// # Key composition
    ///
    /// The key is **not** the request hash alone. [`Self::response_cache_decision`]
    /// produces the request half ([`cache_key`]); this method folds in the
    /// *resolved* model's
    /// [`cache_identity`][tinyinference_llm::model::ChatModel::cache_identity], the
    /// `streaming` flag, and the policy namespace via
    /// [`scoped_cache_key`][crate::cache::scoped_cache_key]. All three
    /// were previously absent from the key:
    ///
    /// * the request's `model` field is never set by the loop and the endpoint
    ///   and credential live inside the `Arc<dyn ChatModel>`, so one shared
    ///   cache served a hosted harness's answer to a local one;
    /// * `streaming` is a parameter of this function, not a request field, so a
    ///   warm streaming run could be served an entry written by a unary run.
    ///
    /// # Accounting
    ///
    /// A cache hit is still counted as a model "step"/call by the caller
    /// ([`Self::run_loop`] increments `model_calls`/`steps` and emits
    /// [`AgentEvent::ModelCompleted`] after this returns) so limit bookkeeping
    /// stays consistent whether or not a call was served from cache. The hit is
    /// stamped [`ModelResponse::served_from_cache`] so *token/cost* accounting
    /// can tell a replay from a real call and not re-bill it.
    ///
    /// # Failure policy
    ///
    /// A cache is an optimization. Neither a failed lookup nor a failed write
    /// fails the run: a read error degrades to a miss, and a write error is
    /// logged and dropped — the provider call has already succeeded and been
    /// paid for, so discarding its answer because the cache was unavailable is
    /// strictly worse than not caching at all.
    async fn invoke_model_with_retry(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
        call_id: &CallId,
        binding: ResolvedModelBinding<State>,
        shape: &super::dialect::CallShape,
    ) -> Result<ModelResponse> {
        let streaming = shape.streaming;
        let policy = self.effective_cache_policy(request);
        // The identity of the model that is actually about to be called — known
        // only *after* resolution, which is why the key cannot be finalized by
        // the request-hashing half alone.
        let identity = binding.model.cache_identity();
        let primary_name = binding.resolved.name.clone();

        // Claude Code executes file and shell tools inside the provider turn.
        // Replaying a cached first turn would skip those side effects entirely,
        // so this provider is never response-cacheable. Other providers retain
        // the normal request-policy behavior.
        let side_effecting_provider = binding
            .model
            .profile()
            .and_then(|profile| profile.provider.as_deref())
            == Some("claude-code");
        let decision = (!side_effecting_provider)
            .then(|| self.response_cache_decision(request))
            .flatten()
            .map(|(cache, base)| {
                let key = scoped_cache_key(
                    &base,
                    identity.as_deref(),
                    streaming,
                    policy.namespace.as_deref(),
                );
                (cache, key)
            });

        if side_effecting_provider {
            tracing::debug!(
                call_id = %call_id.as_str(),
                provider = "claude-code",
                "[cache] response cache disabled for side-effecting provider"
            );
        } else if decision.is_none() {
            let reason = self.cache_skip_reason(request);
            tracing::debug!(
                call_id = %call_id.as_str(),
                reason = reason.as_str(),
                "[cache] response cache not consulted for this model call"
            );
        }

        if let Some((cache, key)) = decision.as_ref() {
            // A read failure is a miss, not a run failure: `InMemoryResponseCache`
            // reports a poisoned mutex as a `Validation` error, and the trait is
            // explicitly designed for third-party implementations whose failure
            // modes we do not control.
            let looked_up = match cache.get(key).await {
                Ok(hit) => hit,
                Err(error) => {
                    tracing::warn!(
                        call_id = %call_id.as_str(),
                        %error,
                        "[cache] response-cache lookup failed; treating as a miss"
                    );
                    None
                }
            };
            if let Some(mut cached) = looked_up {
                if Self::should_skip_empty_response_cache(shape, &cached) {
                    tracing::debug!(
                        call_id = %call_id.as_str(),
                        "[cache] ignoring blank cached response; retry policy requires a provider call"
                    );
                } else {
                    ctx.emit(AgentEvent::CacheHit {
                        call_id: call_id.clone(),
                        key: key.clone(),
                    });
                    cached.served_from_cache = true;
                    if cached.resolved_model.is_none() {
                        cached.resolved_model = Some(binding.resolved.clone());
                    }
                    if streaming {
                        cached = self
                            .replay_cached_response_as_deltas(state, ctx, call_id, cached)
                            .await?;
                    }
                    return Ok(cached);
                }
            }
            ctx.emit(AgentEvent::CacheMiss {
                call_id: call_id.clone(),
                key: key.clone(),
            });
        }

        // Provider prompt-cache breakpoints are injected *after* the key is
        // derived (they mutate `provider_options`, which the key covers), so
        // the common path — no protection, no declared prefix — never pays for
        // a request clone.
        //
        // The *effective* policy is stamped onto the clone. A request that
        // carries no `cache_policy` of its own inherits the harness-level
        // `RunPolicy::cache`, but that inheritance used to stop here: both
        // `apply_prompt_cache_breakpoints` and the provider adapters read
        // `request.cache_policy`, so a host that set `protect_prompt_prefix` on
        // its run policy — the documented way — got no `prompt_cache_key` and
        // no `cache_control` markers on the wire, while the layout guard kept
        // reporting the prefix as protected. Stamping also runs when the run
        // policy does *not* protect but a middleware declared cacheable
        // segments: an adapter treats declared segments alone as the opt-in,
        // and the run policy must be able to veto that.
        let declares_prefix = request
            .cache_segments
            .iter()
            .any(|segment| segment.cacheable);
        let needs_stamp =
            request.cache_policy.is_none() && (policy.protect_prompt_prefix || declares_prefix);
        let mut breakpointed;
        let effective_request = if policy.protect_prompt_prefix || needs_stamp {
            breakpointed = request.clone();
            if needs_stamp {
                breakpointed.cache_policy = Some(policy.clone());
            }
            let injected =
                policy.protect_prompt_prefix && apply_prompt_cache_breakpoints(&mut breakpointed);
            tracing::debug!(
                call_id = %call_id.as_str(),
                protect_prompt_prefix = policy.protect_prompt_prefix,
                prompt_cache_key_injected = injected,
                cacheable_segments = breakpointed.cacheable_prefix_ids().len(),
                "[cache] effective cache policy applied to the outgoing request"
            );
            &breakpointed
        } else {
            request
        };

        // The provider is reached only past the cache lookup and the wrap
        // onion, so a cache hit or a short-circuiting middleware never claims
        // `provider_started`.
        ctx.mark_provider_started();
        let response = self
            .invoke_model_resolving(state, ctx, effective_request, call_id, binding, shape)
            .await?;

        if let Some((cache, key)) = decision.as_ref() {
            // A host that opted into empty-response recovery must be able to
            // reach the provider on the next identical request. Caching the
            // first successful-but-unusable completion would replay that blank
            // answer, mark it `served_from_cache`, and defeat the retry.
            let blank_for_retry = Self::should_skip_empty_response_cache(shape, &response);
            // Only the *primary* model's answer may be stored under this key.
            // `invoke_model_resolving` walks the fallback chain on failure and
            // can return a different model's response; writing that under the
            // primary's key poisons it — permanently, when no TTL is set — so
            // every later run of the primary silently gets the fallback's
            // answer, and reports model B after `ModelStarted` announced A.
            let served_by = response
                .resolved_model
                .as_ref()
                .map(|resolved| resolved.name.as_str());
            if blank_for_retry {
                tracing::debug!(
                    call_id = %call_id.as_str(),
                    "[cache] skipping write: completion has no visible answer and may be retried"
                );
            } else if served_by.is_some_and(|name| name != primary_name) {
                tracing::debug!(
                    call_id = %call_id.as_str(),
                    primary = %primary_name,
                    served_by = served_by.unwrap_or_default(),
                    "[cache] skipping write: a fallback model answered, not the keyed model"
                );
            } else if let Err(error) = cache
                .put_with_ttl(key, response.clone(), policy.ttl())
                .await
            {
                // The provider call already succeeded and was paid for.
                // Discarding its answer because the cache is unavailable would
                // be strictly worse than not caching.
                tracing::warn!(
                    call_id = %call_id.as_str(),
                    %error,
                    "[cache] response-cache write failed; returning the response uncached"
                );
            }
        }

        Ok(response)
    }

    /// A prior cache entry or a newly completed response must not short-circuit
    /// an opted-in retry when it contains no usable assistant answer.
    fn should_skip_empty_response_cache(
        shape: &super::dialect::CallShape,
        response: &ModelResponse,
    ) -> bool {
        // `ModelResponse` has no independent structured payload: the
        // StructuredExtractor reads visible text or tool-call arguments. The
        // call shape still excludes structured plans so their own output retry
        // policy remains the sole owner of failed extraction and caching.
        shape.retry_empty_final
            && response.continue_turn.is_none()
            && response.message.tool_calls.is_empty()
            && response.text().trim().is_empty()
            && !crate::finish_reason::is_length_stop(response.finish_reason.as_deref())
            && response.finish_reason.as_deref() != Some("tool_calls")
    }

    /// Resolves the effective [`CachePolicy`] for `request`: the per-request
    /// policy when present, otherwise the harness-level
    /// [`RunPolicy::cache`][crate::runtime::RunPolicy].
    pub(super) fn effective_cache_policy(&self, request: &ModelRequest) -> CachePolicy {
        request
            .cache_policy
            .clone()
            .unwrap_or_else(|| self.policy.cache.clone())
    }

    /// Explains why [`Self::response_cache_decision`] declined to consult the
    /// cache, for the diagnostic log line on the skip path.
    ///
    /// `docs/modules/harness/cache.md` specifies a richer decision surface than
    /// the bare `CacheHit`/`CacheMiss` pair; without this a caller seeing a 0%
    /// hit rate cannot tell "no cache attached" from "policy off" from "every
    /// request was multi-turn".
    pub(super) fn cache_skip_reason(&self, request: &ModelRequest) -> CacheSkipReason {
        if self.response_cache.is_none() {
            return CacheSkipReason::NoCacheAttached;
        }
        if !self.effective_cache_policy(request).response_cache_enabled {
            return CacheSkipReason::PolicyDisabled;
        }
        CacheSkipReason::MultiTurnTranscript
    }

    /// Replays a cache hit as synthetic stream deltas so a warm streaming run
    /// is observationally identical to a cold one.
    ///
    /// A cache hit short-circuits before [`Self::invoke_model_streaming_once`],
    /// so a streaming run served from cache used to emit **zero**
    /// [`AgentEvent::ModelDelta`] events and run **zero**
    /// [`on_model_delta`][crate::middleware::Middleware::on_model_delta]
    /// hooks — a UI rendering deltas showed nothing at all, contradicting the
    /// streaming contract documented on the harness entry points. LangChain
    /// replays hits as synthetic stream events for exactly this reason.
    ///
    /// The replay emits one text delta (when the cached message has text) and
    /// one delta per cached tool call, mirroring what the provider stream would
    /// have produced. It is a replay, not a re-derivation: no provider is
    /// contacted.
    async fn replay_cached_response_as_deltas(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        call_id: &CallId,
        mut cached: ModelResponse,
    ) -> Result<ModelResponse> {
        let content = cached.message.content.clone();
        let tool_calls = cached.tool_calls().to_vec();
        tracing::debug!(
            call_id = %call_id.as_str(),
            text_len = cached.text().len(),
            tool_calls = tool_calls.len(),
            "[cache] replaying a cache hit as synthetic stream deltas"
        );

        let mut deltas: Vec<MessageDelta> = Vec::new();
        for block in &content {
            match block {
                tinyinference_llm::message::ContentBlock::Text(text) => {
                    deltas.push(MessageDelta {
                        text: text.clone(),
                        reasoning: String::new(),
                        tool_call: None,
                    });
                }
                tinyinference_llm::message::ContentBlock::Thinking { text, .. } => {
                    deltas.push(MessageDelta {
                        text: String::new(),
                        reasoning: text.clone(),
                        tool_call: None,
                    });
                }
                _ => {}
            }
        }
        for call in &tool_calls {
            deltas.push(MessageDelta {
                text: String::new(),
                reasoning: String::new(),
                tool_call: Some(tinyinference_llm::tool::ToolDelta {
                    call_id: call.id.clone(),
                    content: serde_json::to_string(&call.arguments).unwrap_or_default(),
                    tool_name: Some(call.name.clone()),
                    ..Default::default()
                }),
            });
        }

        let mut streamed_text = String::new();
        let mut streamed_reasoning = String::new();
        let mut saw_streamed_content = false;
        let mut transformed_tools = StreamAccumulator::new();
        let mut saw_tool_delta = false;
        for delta in deltas {
            saw_tool_delta |= delta.tool_call.is_some();
            let mut model_delta = ModelDelta {
                call_id: call_id.as_str().to_string(),
                content: delta.text.clone(),
                reasoning: delta.reasoning.clone(),
                tool_call: delta.tool_call.clone(),
            };
            self.middleware
                .run_on_model_delta(ctx, state, &mut model_delta)
                .await?;
            if let Some(tool_call) = model_delta.tool_call.clone() {
                transformed_tools.push(&ModelStreamItem::ToolCallDelta(tool_call));
            }
            saw_streamed_content |= !delta.text.is_empty()
                || !delta.reasoning.is_empty()
                || !model_delta.content.is_empty()
                || !model_delta.reasoning.is_empty();
            streamed_text.push_str(&model_delta.content);
            streamed_reasoning.push_str(&model_delta.reasoning);
            ctx.emit(AgentEvent::ModelDelta {
                run_id: ctx.config.run_id.clone(),
                call_id: call_id.clone(),
                delta: MessageDelta {
                    text: model_delta.content.clone(),
                    reasoning: model_delta.reasoning.clone(),
                    tool_call: model_delta.tool_call.clone(),
                },
            });
            crate::runtime::emit_host_progress::<State, Ctx>(
                ctx,
                crate::host::ProgressEvent::Token {
                    run: ctx.run_id().clone(),
                    text: model_delta.content,
                },
            );
        }
        if saw_streamed_content {
            // Same rule as the live streaming path (see the matching comment
            // in `invoke_model_streaming_once`): keep the cached response's
            // own `Thinking` blocks (with their signature) verbatim unless
            // the synthetic replay deltas were actually transformed by
            // `on_model_delta`, since a signed thinking block must be
            // replayed byte-for-byte ahead of a tool call on the next turn.
            let cached_reasoning: String = cached
                .message
                .content
                .iter()
                .filter_map(|block| match block {
                    tinyinference_llm::message::ContentBlock::Thinking { text, .. } => {
                        Some(text.as_str())
                    }
                    _ => None,
                })
                .collect();
            let reasoning_untransformed = cached_reasoning == streamed_reasoning;

            let mut transformed_content = Vec::new();
            if reasoning_untransformed {
                transformed_content.extend(
                    cached
                        .message
                        .content
                        .iter()
                        .filter(|block| {
                            matches!(
                                block,
                                tinyinference_llm::message::ContentBlock::Thinking { .. }
                            )
                        })
                        .cloned(),
                );
            } else if !streamed_reasoning.is_empty() {
                transformed_content.push(tinyinference_llm::message::ContentBlock::Thinking {
                    text: streamed_reasoning,
                    signature: None,
                });
            }
            if !streamed_text.is_empty() {
                transformed_content.push(tinyinference_llm::message::ContentBlock::Text(
                    streamed_text,
                ));
            }
            transformed_content.extend(cached.message.content.drain(..).filter(|block| {
                !matches!(
                    block,
                    tinyinference_llm::message::ContentBlock::Text(_)
                        | tinyinference_llm::message::ContentBlock::Thinking { .. }
                )
            }));
            cached.message.content = transformed_content;
        }
        if saw_tool_delta {
            cached.message.tool_calls = transformed_tools.finish()?.message.tool_calls;
        }
        Ok(cached)
    }

    /// Invokes a model with retry and fallback (no caching).
    ///
    /// Retries are governed by [`RunPolicy::retry`][crate::runtime::RunPolicy]
    /// and apply only to retryable errors (see
    /// [`is_retryable`][crate::retry::is_retryable]); each scheduled
    /// retry emits [`AgentEvent::RetryScheduled`]. When retries are exhausted
    /// (or the error is non-retryable) and a [`crate::retry::FallbackPolicy`]
    /// is configured, the next model in the chain is tried. The computed backoff
    /// duration is intentionally not slept on (see the module docs).
    async fn invoke_model_resolving(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
        call_id: &CallId,
        binding: ResolvedModelBinding<State>,
        shape: &super::dialect::CallShape,
    ) -> Result<ModelResponse> {
        let streaming = shape.streaming;
        let mut current_name = binding.resolved.name.clone();
        let mut model = binding.model;
        let mut resolved = binding.resolved;
        let run_id = ctx.run_id().clone();
        // The request each attempt sends. `request.model` is the wire-level
        // override providers such as the Claude CLI adapters honour, so it must
        // always name the binding actually being called: every fallback
        // rebinding below retargets it, or a fallback would re-ask the model
        // that just failed while events report the fallback's name.
        let mut attempt_request = request.clone();
        // Tracks every model name already attempted in this fallback chain so
        // a chain containing a repeated name (e.g. `[primary, backup,
        // primary]`) cannot alternate between the same two models forever;
        // once a name has been tried it is never tried again.
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(current_name.clone());

        // Cross-call skip hint: a model that already failed with a permanent
        // credential error earlier in this run is not asked again while a
        // fallback can answer instead. With no eligible fallback it is tried
        // anyway — failing again is better than refusing to try.
        if ctx.limits.is_model_skipped(&current_name)
            && crate::runtime::host_invocation_binding::<State, Ctx>(ctx)?.is_none()
            && let Some((name, next_model)) =
                self.select_fallback(ctx, request, &current_name, &mut visited, None)
        {
            tracing::debug!(
                call_id = %call_id.as_str(),
                skipped = %current_name,
                to = %name,
                "[failover] skipping a model written off earlier in this run"
            );
            ctx.emit(AgentEvent::FallbackSkipped {
                model: current_name.clone(),
            });
            // The substitute counts as tried, like every fallback taken in the
            // loop below, so a repeated chain entry cannot select it twice.
            visited.insert(name.clone());
            resolved = ResolvedModel {
                name: name.clone(),
                requested: Some(name.clone()),
                source: ModelResolutionSource::Hint,
            };
            retarget_request_model(&mut attempt_request, &name);
            current_name = name;
            model = next_model;
        }

        loop {
            // Retry loop for the current model.
            let mut attempt = 0usize;
            // Track the source of a limit error explicitly. The idle breaker
            // may send its own limit error through fallback, while all other
            // limit errors remain terminal regardless of the timeout count.
            let mut idle_breaker_error = false;
            // Counts the deltas the *current* streaming attempt has already
            // handed to consumers. A stream that dies after 200 tokens has
            // already delivered them; the retry replays from scratch, so a UI
            // concatenating `ModelDelta.text` renders partial garbage followed
            // by the full answer. `StreamAccumulator` is discarded internally,
            // but consumers are never told to discard too — see the warning
            // below and the `AgentEvent` handoff noted in the module docs.
            let mut deltas_emitted = 0usize;
            let outcome = loop {
                // Observe cancellation before (re)issuing a model attempt so a
                // cancel requested during a retry/rate-limit wait stops the run
                // promptly instead of firing another provider call or falling
                // through to the fallback chain.
                if ctx.cancellation.is_cancelled() {
                    return Err(TinyAgentsError::Cancelled);
                }
                // Bound this individual provider call by the tighter of the
                // run's *remaining* wall-clock budget and the per-model-call
                // ceiling, so a hung or slow model call is interrupted
                // mid-flight, not merely detected by the between-call deadline
                // check. reqwest/futures are cancel-safe, so dropping the future
                // on elapse cancels the underlying request. Recomputed here,
                // inside the attempt loop, so every retry attempt gets a fresh
                // per-call window. When neither the run config nor the harness
                // policy configures any timeout the call is awaited unbounded.
                let (remaining, bound) = self.model_call_budget(ctx);
                let attempt_result = if streaming {
                    let fut = self.invoke_model_streaming_once(
                        state,
                        ctx,
                        &model,
                        &attempt_request,
                        call_id,
                        &mut deltas_emitted,
                        &current_name,
                        shape,
                    );
                    Self::with_call_budget(remaining, run_id.as_str(), "model call", bound, fut)
                        .await
                } else {
                    // Race the wall-clock-bounded unary call against cooperative
                    // cancellation, mirroring the streaming path: a cancel
                    // requested while a buffered (non-streamed) provider call is
                    // in flight drops the future — reqwest cancels the underlying
                    // request — and unwinds with `Cancelled` instead of paying
                    // for the call to run to completion. `cancelled()` is
                    // cancel-safe, and the pre-call `is_cancelled()` check above
                    // still short-circuits before the request is ever issued.
                    let fut = async {
                        model
                            .invoke(state, attempt_request.clone())
                            .await
                            .map_err(TinyAgentsError::from)
                    };
                    let budgeted = Self::with_call_budget(
                        remaining,
                        run_id.as_str(),
                        "model call",
                        bound,
                        fut,
                    );
                    // `with_call_budget` already applies its own deadline, so
                    // this only needs to race cancellation against an
                    // otherwise-unbounded future — `bounded`'s `None` arm,
                    // which never calls `timeout_message`.
                    ctx.bounded(None, budgeted, || {
                        unreachable!("with_call_budget already applies its own timeout")
                    })
                    .await
                };
                match attempt_result {
                    Ok(response) => break Ok(response),
                    Err(error) => {
                        // Provider and run-budget refusals are terminal: a
                        // retry or another model cannot make the exhausted
                        // budget available again.
                        if matches!(&error, TinyAgentsError::LimitExceeded(_)) {
                            break Err(error);
                        }
                        // The breaker outranks retry, not fallback: a model
                        // that keeps going silent is not retried again, but
                        // the chain below may still reach a healthy model.
                        // Only when the chain is exhausted does the run fail
                        // with the breaker's error.
                        if matches!(error, TinyAgentsError::CallTimeout(_))
                            && let Some(tripped) =
                                self.stream_idle_breaker_error(ctx, &current_name)
                        {
                            idle_breaker_error = true;
                            break Err(tripped);
                        }
                        // `RunLimits::max_retries_per_call` is a hard ceiling
                        // that a looser `RetryPolicy::max_attempts` cannot
                        // exceed; whichever is stricter wins.
                        // A registered `RetryMiddleware` (or any other
                        // `ModelMiddleware::overrides_retry`) already retries
                        // the whole wrap onion around this base call. Retrying
                        // again here would multiply attempts
                        // (`mw.max_attempts × policy.retry.max_attempts ×
                        // |fallback|` for one logical failure) and emit
                        // `RetryScheduled` for attempts the middleware cannot
                        // see, so the base call skips its own retry loop and
                        // defers entirely to the middleware (I-7); the
                        // fallback chain below is unaffected.
                        let retry_overridden = self.middleware.has_retry_override();
                        let max_attempts = self
                            .policy
                            .retry
                            .max_attempts_capped_at(self.policy.limits.max_retries_per_call);
                        // Route the retry decision through the shared
                        // `RetryPolicy::should_retry_error` engine (same
                        // classification + attempt-cap logic RetryMiddleware
                        // uses), applying the harness ceiling by capping a
                        // cloned policy first so the two sites cannot drift.
                        let capped = self.policy.retry.clone().with_max_attempts(max_attempts);
                        // Reason-aware failover: *why* the call failed decides
                        // whether a same-model retry can possibly help (see
                        // `crate::retry::decide` for the table). A rejected
                        // credential or a missing model is never re-sent.
                        let reason = FailoverReason::classify(&error);
                        let mut failover_state = FailoverState::for_error(&capped, attempt, &error);
                        if retry_overridden {
                            failover_state.attempts_remaining = false;
                        }
                        let decision = decide(reason, failover_state);
                        tracing::debug!(
                            call_id = %call_id.as_str(),
                            model = %current_name,
                            attempt,
                            reason = reason.as_str(),
                            ?decision,
                            "[failover] model attempt failed"
                        );
                        if decision == FailoverDecision::RetrySame {
                            // Compute the backoff from the *pre-increment*
                            // attempt number: `attempt == 0` is the first
                            // retry and must sleep `initial_backoff_ms`
                            // (`RetryPolicy::backoff_for_attempt(0)`). Sleeping
                            // on the post-increment value skipped
                            // `initial_backoff_ms` entirely and shifted the
                            // whole exponential schedule one step too high.
                            let backoff_attempt = attempt;
                            attempt += 1;
                            if streaming && deltas_emitted > 0 {
                                // The retry re-emits the whole response from
                                // the beginning. Until `AgentEvent` grows a
                                // dedicated discard marker, `RetryScheduled`
                                // for a streaming call *is* the signal that
                                // every delta seen so far for this `call_id`
                                // must be dropped.
                                tracing::warn!(
                                    call_id = %call_id.as_str(),
                                    discarded_deltas = deltas_emitted,
                                    attempt,
                                    "[stream] retrying a streaming call that already emitted \
                                     deltas; consumers must discard everything received so far \
                                     for this call_id"
                                );
                            }
                            deltas_emitted = 0;
                            ctx.emit(AgentEvent::RetryScheduled {
                                call_id: call_id.clone(),
                                attempt,
                            });
                            // Sleep for the backoff only when the policy opts in
                            // (`with_backoff_sleep`); otherwise this is a no-op so
                            // the loop stays fast and deterministic in tests.
                            self.policy
                                .retry
                                .sleep_backoff_for_error(backoff_attempt, &error)
                                .await;
                            continue;
                        }
                        break Err(error);
                    }
                }
            };

            match outcome {
                Ok(mut response) => {
                    if response.resolved_model.is_none() {
                        response.resolved_model = Some(resolved);
                    }
                    split_thinking_tags(&mut response, model.profile());
                    return Ok(response);
                }
                Err(error) => {
                    // A non-retryable, deadline-driven timeout must not feed
                    // back into the fallback chain: the run itself is out of
                    // wall-clock budget, so trying another model would just
                    // spin until the *next* deadline check fails identically.
                    if matches!(error, TinyAgentsError::Timeout(_))
                        || error.is_terminal_limit() && !idle_breaker_error
                    {
                        return Err(error);
                    }
                    // A hosted resolver owns routing authority. Its first
                    // decision must not fall through to harness-local
                    // fallback names the host did not approve.
                    if crate::runtime::host_invocation_binding::<State, Ctx>(ctx)?.is_some() {
                        return Err(error);
                    }
                    // Retries are over for this model. Ask the failover table
                    // whether another model could help. A context overflow is
                    // the one case that needs a *specific* kind of candidate:
                    // one whose declared window is strictly larger than this
                    // model's; with none, it surfaces here so compaction (not a
                    // lookalike sibling) handles it.
                    let reason = FailoverReason::classify(&error);
                    if reason.skips_model_for_run() {
                        tracing::warn!(
                            call_id = %call_id.as_str(),
                            model = %current_name,
                            reason = reason.as_str(),
                            "[failover] model written off for the rest of this run"
                        );
                        ctx.limits.skip_model_for_run(&current_name);
                    }
                    // Unknown current window => no evidence any sibling is
                    // larger, so nothing qualifies.
                    let larger_than = (reason == FailoverReason::ContextOverflow).then(|| {
                        model
                            .profile()
                            .and_then(|profile| profile.max_input_tokens)
                            .unwrap_or(u64::MAX)
                    });
                    let selected = self.select_fallback(
                        ctx,
                        request,
                        &current_name,
                        &mut visited,
                        larger_than,
                    );
                    let decision = decide(
                        reason,
                        FailoverState {
                            retryable: false,
                            attempts_remaining: false,
                            larger_window_available: selected.is_some(),
                        },
                    );
                    tracing::debug!(
                        call_id = %call_id.as_str(),
                        model = %current_name,
                        reason = reason.as_str(),
                        ?decision,
                        "[failover] model gave up; deciding between fallback and surfacing"
                    );
                    if decision == FailoverDecision::Surface {
                        return Err(error);
                    }
                    match selected {
                        Some((name, next_model)) => {
                            visited.insert(name.clone());
                            resolved = ResolvedModel {
                                name: name.clone(),
                                requested: Some(name.clone()),
                                source: ModelResolutionSource::Hint,
                            };
                            retarget_request_model(&mut attempt_request, &name);
                            current_name = name;
                            model = next_model;
                            if streaming && deltas_emitted > 0 {
                                // A fallback starts a fresh response under the
                                // same call id. RetryScheduled is the existing
                                // consumer-visible discard marker for partial
                                // streaming attempts.
                                ctx.emit(AgentEvent::RetryScheduled {
                                    call_id: call_id.clone(),
                                    attempt: attempt + 1,
                                });
                            }
                            continue;
                        }
                        None => return Err(error),
                    }
                }
            }
        }
    }

    /// Picks the next fallback model after `cursor`, or `None` when the chain
    /// is exhausted.
    ///
    /// Skips any name already in `visited` (so a chain with a repeated name
    /// cannot alternate between the same models forever), any model the run has
    /// written off ([`crate::limits::LimitTracker::is_model_skipped`]) and any
    /// candidate that fails the request's capability/lifecycle gate. Initial
    /// resolution gates the primary selection through `model_eligible`; without
    /// the same gate here a primary failure could silently fall back to a model
    /// that can't call tools, lacks vision, or has a smaller context window
    /// (issue #4641). `allow_retired` is `false` to match
    /// `ModelRegistry::resolve_request`. Every skipped candidate is recorded as
    /// visited and surfaced as [`AgentEvent::FallbackSkipped`]. With
    /// `larger_than = Some(n)` a candidate must also declare a
    /// `max_input_tokens` strictly greater than `n` (context-overflow failover).
    fn select_fallback(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
        cursor: &str,
        visited: &mut std::collections::HashSet<String>,
        larger_than: Option<u64>,
    ) -> Option<(String, Arc<dyn ChatModel<State>>)> {
        let required = request.required_capabilities.as_ref();
        // A steered model outside the chain falls back through the whole
        // chain (the original primary's), not from a position it lacks.
        let mut from_chain_head = self.fallback_starts_at_chain_head(ctx, cursor);
        let mut cursor = cursor.to_owned();
        loop {
            let next = self
                .policy
                .fallback
                .as_ref()
                .and_then(|fallback| {
                    if std::mem::take(&mut from_chain_head) {
                        fallback.models.first().map(String::as_str)
                    } else {
                        fallback.next_after(&cursor)
                    }
                })
                .map(str::to_owned)
                .filter(|name| !visited.contains(name));
            let (name, next_model) =
                next.and_then(|name| self.models.get(&name).map(|m| (name, m)))?;
            let written_off = ctx.limits.is_model_skipped(&name);
            // `larger_than` (context-overflow failover): the candidate must
            // *declare* a window strictly above the given one.
            let too_small = larger_than.is_some_and(|floor| {
                next_model
                    .profile()
                    .and_then(|profile| profile.max_input_tokens)
                    .is_none_or(|window| window <= floor)
            });
            if written_off || too_small || !model_eligible(next_model.as_ref(), required, false) {
                visited.insert(name.clone());
                ctx.emit(AgentEvent::FallbackSkipped {
                    model: name.clone(),
                });
                cursor = name;
                continue;
            }
            return Some((name, next_model));
        }
    }

    /// Computes the wall-clock budget for the next individual model call.
    ///
    /// The budget is the *tighter* of two remaining-time sources:
    ///
    /// - the run config's `timeout_ms` (the same deadline the between-call
    ///   [`RunContext::check_deadline`] enforces), tracked by the run's
    ///   [`crate::limits::LimitTracker`], and
    /// - the harness policy's
    ///   [`RunLimits::max_wall_clock_ms`][crate::limits::RunLimits::max_wall_clock_ms],
    ///   measured against the same tracker start.
    ///
    /// Either source may be absent; when both are absent the call is unbounded
    /// (`None`). Honoring the policy source lets a sub-agent whose child
    /// [`RunConfig`] carries no per-run timeout still be bounded by its
    /// harness's policy-level wall-clock cap.
    ///
    /// This is the budget for **tool calls**, which are bounded only by the
    /// run's remaining time (a sub-agent delegation is a tool call wrapping an
    /// entire child run). Model calls go through
    /// [`model_call_budget`](Self::model_call_budget), which additionally
    /// clamps to the per-call ceiling
    /// [`RunLimits::max_model_call_ms`][crate::limits::RunLimits::max_model_call_ms].
    pub(super) fn call_budget(&self, ctx: &RunContext<Ctx>) -> Option<Duration> {
        let config_budget = ctx.remaining_wall_clock();
        let policy_budget = self.policy.limits.max_wall_clock_ms.map(|ms| {
            Duration::from_millis(ms)
                .checked_sub(ctx.limits.elapsed())
                .unwrap_or(Duration::ZERO)
        });
        match (config_budget, policy_budget) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// Computes the wall-clock budget for the next individual **model** call:
    /// the tighter of the run's remaining budget ([`call_budget`](Self::call_budget))
    /// and the per-call ceiling
    /// [`RunLimits::max_model_call_ms`][crate::limits::RunLimits::max_model_call_ms].
    ///
    /// Computed afresh for every call *and every retry attempt*, so each
    /// attempt gets its own full per-call window rather than inheriting an
    /// earlier attempt's consumption — while still never overshooting the run
    /// deadline. Returns the budget together with the label of whichever bound
    /// is in force, so the timeout error names the ceiling that actually fired
    /// (a per-call ceiling means "this one call wedged"; the run's remainder
    /// means "the run is out of time") — field triage needs to tell them apart.
    pub(super) fn model_call_budget(
        &self,
        ctx: &RunContext<Ctx>,
    ) -> (Option<Duration>, &'static str) {
        let run_budget = self.call_budget(ctx);
        let per_call = self
            .policy
            .limits
            .max_model_call_ms
            .map(Duration::from_millis);
        match (run_budget, per_call) {
            (Some(run), Some(cap)) if cap < run => (Some(cap), PER_CALL_BOUND_LABEL),
            (None, Some(cap)) => (Some(cap), PER_CALL_BOUND_LABEL),
            (run, _) => (run, RUN_BOUND_LABEL),
        }
    }

    /// Returns the breaker error when the idle timeouts since the last output
    /// event on the current model have reached
    /// [`RunLimits::max_consecutive_stream_idle_timeouts`][crate::limits::RunLimits::max_consecutive_stream_idle_timeouts].
    ///
    /// [`TinyAgentsError::LimitExceeded`] is not retryable, so the caller
    /// stops retrying this model; the fallback chain is still consulted.
    fn stream_idle_breaker_error(
        &self,
        ctx: &RunContext<Ctx>,
        model: &str,
    ) -> Option<TinyAgentsError> {
        // A zero threshold would trip on the first timeout; treat it as off.
        let max = self
            .policy
            .limits
            .max_consecutive_stream_idle_timeouts
            .filter(|max| *max > 0)?;
        let consecutive = ctx.limits.consecutive_stream_idle_timeouts_for(model);
        if consecutive < max {
            return None;
        }
        tracing::warn!(
            run_id = %ctx.run_id(),
            consecutive_idle_timeouts = consecutive,
            max,
            "[stream] idle-timeout breaker tripped; no further retries on this model"
        );
        Some(TinyAgentsError::LimitExceeded(format!(
            "model stream idle-timeout breaker tripped for run `{}`: {consecutive} consecutive \
             idle timeouts on one model (limit {max}); the provider appears stalled",
            ctx.run_id()
        )))
    }

    /// Awaits a single call future (model or tool), optionally bounded by
    /// `budget`.
    ///
    /// When `budget` is `Some`, the future is wrapped in
    /// [`tokio::time::timeout`]; if it elapses the future is dropped (cancelling
    /// the in-flight provider/tool request) and a
    /// [`TinyAgentsError::Timeout`] is returned. When `budget` is `None` (no
    /// run timeout configured) the future is awaited without a bound.
    ///
    /// `budget` is either the run's *remaining* wall-clock budget at the time
    /// the call is issued (each successive call gets a tighter bound as the
    /// deadline approaches) or, for model calls, the per-call ceiling when that
    /// is tighter. `what` names the kind of call in the timeout message (e.g.
    /// `"model call"`, `"tool call"`); `bound` names which budget source is in
    /// force ([`RUN_BOUND_LABEL`] / [`PER_CALL_BOUND_LABEL`]) so the message
    /// says which ceiling fired.
    pub(super) async fn with_call_budget<T, F>(
        budget: Option<Duration>,
        run_id: &str,
        what: &str,
        bound: &str,
        fut: F,
    ) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match budget {
            Some(budget) => match tokio::time::timeout(budget, fut).await {
                Ok(result) => result,
                Err(_) => {
                    let message = format!(
                        "{what} for run `{run_id}` exceeded its {bound} ({} ms)",
                        budget.as_millis()
                    );
                    // Only the per-model-call ceiling is retryable: it means
                    // this one call wedged, not that the run is out of time.
                    // Every other bound this helper is used with (the run's
                    // remaining wall-clock budget, for model calls, tool
                    // calls, host resolution, tool authorization/screening,
                    // and host turn preparation) is terminal.
                    if bound == PER_CALL_BOUND_LABEL {
                        Err(TinyAgentsError::CallTimeout(message))
                    } else {
                        Err(TinyAgentsError::Timeout(message))
                    }
                }
            },
            None => fut.await,
        }
    }

    /// Drives one streaming model call to completion.
    ///
    /// Consumes [`tinyinference_llm::model::ChatModel::stream`], emitting an
    /// [`AgentEvent::ModelDelta`] and running every middleware's
    /// [`on_model_delta`][crate::middleware::Middleware::on_model_delta]
    /// hook for each [`ModelStreamItem::MessageDelta`] (and standalone
    /// [`ModelStreamItem::ToolCallDelta`]), then folds the items into the final
    /// [`ModelResponse`] via [`StreamAccumulator`]. Terminal provider metadata
    /// is retained, while terminal text/thinking is reconciled from those
    /// transformed deltas so the returned response agrees with streaming
    /// consumers.
    ///
    /// `deltas_emitted` is incremented for every delta actually handed to
    /// consumers, so the retry path can tell whether a failed attempt already
    /// published output that now has to be discarded.
    // `deltas_emitted` must stay an out-parameter: on the error path the
    // retry logic reads how much output already reached consumers, which a
    // return value could not carry alongside the error.
    #[allow(clippy::too_many_arguments)]
    async fn invoke_model_streaming_once(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        model: &Arc<dyn ChatModel<State>>,
        request: &ModelRequest,
        call_id: &CallId,
        deltas_emitted: &mut usize,
        model_name: &str,
        shape: &super::dialect::CallShape,
    ) -> Result<ModelResponse> {
        let recovery = &shape.recovery;
        // Start the first-event window before opening the provider stream so
        // a provider that hangs while establishing the stream is bounded too.
        // The same deadline is then used for the first item below, preserving
        // the configured window across both phases.
        let cancellation = ctx.cancellation.clone();
        let first_event_timeout = positive_window(self.policy.limits.stream_first_event_timeout_ms);
        let first_event_deadline =
            first_event_timeout.map(|window| tokio::time::Instant::now() + window);
        let stream_result = async {
            match first_event_deadline {
                Some(deadline) => {
                    tokio::time::timeout_at(deadline, model.stream(state, request.clone()))
                        .await
                        .map_err(|_| ())
                }
                None => Ok(model.stream(state, request.clone()).await),
            }
        };
        let mut stream = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(TinyAgentsError::Cancelled),
            result = stream_result => match result {
                Ok(result) => result?,
                Err(()) => {
                    let window_ms = first_event_timeout.map_or(0, |window| window.as_millis() as u64);
                    ctx.limits.record_stream_idle_timeout_for(model_name);
                    return Err(TinyAgentsError::CallTimeout(format!(
                        "model stream for run `{}` went idle waiting for the first output event: no event within {window_ms} ms",
                        ctx.run_id(),
                    )));
                }
            },
        };
        let mut accumulator = StreamAccumulator::new();
        // Tool-call markup a model narrates as text is held back from live
        // consumers and turned into calls on the terminal response instead.
        // Runs for every provider: native models narrate calls often enough.
        let mut text_scrubber = recovery.scrubber(call_id);
        // A terminal `Completed` response usually has richer provider metadata
        // than deltas (message id, usage, tool calls, and route information),
        // but its text is still the raw provider payload.  Keep the text and
        // reasoning that actually crossed the middleware boundary so that the
        // terminal item cannot restore content a delta middleware redacted or
        // transformed before it reached consumers.
        let mut streamed_text = String::new();
        let mut streamed_reasoning = String::new();
        // Characters of `streamed_reasoning`, kept as a running count so the
        // watchdog estimate below is in characters, not UTF-8 bytes, without
        // walking the whole text on every delta.
        let mut streamed_reasoning_chars: u64 = 0;
        let mut saw_streamed_content = false;
        let mut transformed_tools = StreamAccumulator::new();
        let mut saw_tool_delta = false;
        let mut stream_stall = StreamTextStallDetector::default();
        // Client-side bound on hidden reasoning (see
        // `RunPolicy::reasoning_watchdog`): the reasoning-token count at which
        // a call that has shown nothing visible is ended as a dead call.
        let reasoning_bound = self.policy.reasoning_watchdog.bound_for(request);
        let watchdog_started = std::time::Instant::now();

        // Some providers pad the very first streamed text chunk with
        // whitespace that is a wire-format artifact, not content (see
        // `ModelProfile::ignore_streamed_leading_whitespace`). Stripped once,
        // on the first delta that actually carries non-whitespace text;
        // deltas consisting only of leading whitespace are dropped outright
        // rather than surfaced empty.
        let mut strip_leading_whitespace = model
            .profile()
            .map(|profile| profile.ignore_streamed_leading_whitespace)
            .unwrap_or(false);

        // Deadline-based stream watchdog. Before the first output event the
        // only bound is the opt-in first-event window (none by default: hidden
        // reasoning and local prefill are legitimately silent for minutes, and
        // `max_model_call_ms` / the run deadline already bound the call). Once
        // output arrives, every further output event pushes the deadline to
        // `now + idle`. Stream-opened markers and usage updates never move it,
        // so a provider that keeps trickling those cannot hold the call open.
        let limits = &self.policy.limits;
        let idle_timeout = positive_window(limits.stream_idle_timeout_ms);
        let mut armed_window = first_event_timeout;
        let mut deadline = first_event_deadline;
        let mut saw_output = false;
        // Set when an output event arrives; the deadline is advanced at the top
        // of the next iteration so time spent in delta middleware does not eat
        // the provider's idle budget.
        let mut rearm = false;

        loop {
            if rearm {
                rearm = false;
                armed_window = idle_timeout;
                deadline = armed_window.map(|window| tokio::time::Instant::now() + window);
            }
            // Race the next provider chunk against cooperative cancellation. If
            // cancellation wins we drop the partially consumed stream and unwind
            // with `Cancelled`; the `cancelled()` future is cancel-safe.
            //
            // The wait for the next event is also bounded by the watchdog
            // deadline, so a provider that goes silent fails fast with a
            // retryable `CallTimeout` instead of holding the call until the
            // per-call or run deadline.
            let next_event = async {
                match deadline {
                    Some(deadline) => tokio::time::timeout_at(deadline, stream.next())
                        .await
                        .map_err(|_| ()),
                    None => Ok(stream.next().await),
                }
            };
            let next = tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    return Err(TinyAgentsError::Cancelled);
                }
                next = next_event => next,
            };
            let mut item = match next {
                Ok(Some(item)) => item,
                Ok(None) => break,
                Err(()) => {
                    let consecutive = ctx.limits.record_stream_idle_timeout_for(model_name);
                    let phase = if saw_output {
                        "between output events"
                    } else {
                        "waiting for the first output event"
                    };
                    let window_ms = armed_window.map_or(0, |window| window.as_millis() as u64);
                    tracing::warn!(
                        call_id = %call_id.as_str(),
                        window_ms,
                        consecutive_idle_timeouts = consecutive,
                        phase,
                        "[stream] model stream idle timeout"
                    );
                    return Err(TinyAgentsError::CallTimeout(format!(
                        "model stream for run `{}` went idle {phase}: no event within {window_ms} ms",
                        ctx.run_id(),
                    )));
                }
            };
            // Liveness is judged on what the provider sent, not on what is
            // left once the scrubber and delta middleware have rewritten it:
            // a model writing a long tool call as text has every delta
            // consumed by the scrubber, yet it is plainly producing tokens.
            // A delta with no payload (empty text, no reasoning, no call)
            // proves nothing and never counts.
            let raw_delta_is_output = match &item {
                ModelStreamItem::MessageDelta(delta) => {
                    !delta.text.is_empty()
                        || !delta.reasoning.is_empty()
                        || delta.tool_call.is_some()
                }
                ModelStreamItem::ToolCallDelta(_) => true,
                _ => false,
            };
            // Scrub tool-call markup from visible text before anything else
            // sees it; a delta the scrubber empties carries nothing to emit.
            if let (Some(scrubber), ModelStreamItem::MessageDelta(delta)) =
                (text_scrubber.as_mut(), &mut item)
                && !delta.text.is_empty()
            {
                delta.text = scrubber.feed(&delta.text);
                if delta.text.is_empty() && delta.reasoning.is_empty() && delta.tool_call.is_none()
                {
                    // Consumed by the scrubber, but still provider output.
                    saw_output = true;
                    rearm = true;
                    ctx.limits.reset_stream_idle_timeouts_for(model_name);
                    continue;
                }
            }
            if let (Some(scrubber), ModelStreamItem::Completed(_)) = (text_scrubber.as_mut(), &item)
            {
                let tail = scrubber.flush();
                if !tail.is_empty() {
                    // The held-back remainder is ordinary text after all.
                    // Route it through the same delta middleware pipeline as
                    // every other streamed delta (below): a naive direct
                    // emit skipped `run_on_model_delta` and host progress, so
                    // redaction/policy/transformation middleware could not
                    // inspect or suppress this tail and consumers saw it
                    // behave differently from every other delta.
                    let mut model_delta = ModelDelta {
                        call_id: call_id.as_str().to_string(),
                        content: tail,
                        reasoning: String::new(),
                        tool_call: None,
                    };
                    self.middleware
                        .run_on_model_delta(ctx, state, &mut model_delta)
                        .await?;
                    if stream_stall.observe(&model_delta.content) {
                        tracing::warn!("[stream] stopped repetitive model narration");
                        return Err(TinyAgentsError::GenerationStalled);
                    }
                    // Unconditional, not gated on the post-middleware content:
                    // the pre-middleware tail here is always non-empty (the
                    // surrounding `if` already checked it), matching the
                    // ordinary delta path below, which ORs the *pre*-middleware
                    // text against the post-middleware one. Gating on
                    // `model_delta.content` alone meant a middleware that
                    // suppressed the whole tail to `""` left
                    // `saw_streamed_content` false, which skipped terminal
                    // reconciliation and let the provider's raw (unscrubbed)
                    // `Completed` content silently restore the exact text the
                    // middleware had just suppressed.
                    saw_streamed_content = true;
                    streamed_text.push_str(&model_delta.content);
                    ctx.emit(AgentEvent::ModelDelta {
                        run_id: ctx.config.run_id.clone(),
                        call_id: call_id.clone(),
                        delta: MessageDelta::text(model_delta.content.clone()),
                    });
                    crate::runtime::emit_host_progress::<State, Ctx>(
                        ctx,
                        crate::host::ProgressEvent::Token {
                            run: ctx.run_id().clone(),
                            text: model_delta.content,
                        },
                    );
                    *deltas_emitted += 1;
                }
            }

            // Surface incremental message/tool-call fragments through events and
            // the `on_model_delta` middleware hook before merging them.
            let message_delta = match &item {
                ModelStreamItem::MessageDelta(delta) => Some(delta.clone()),
                ModelStreamItem::ToolCallDelta(tool_delta) => Some(MessageDelta {
                    text: String::new(),
                    reasoning: String::new(),
                    tool_call: Some(tool_delta.clone()),
                }),
                _ => None,
            };

            if let Some(mut message_delta) = message_delta {
                if strip_leading_whitespace && !message_delta.text.is_empty() {
                    let stripped = message_delta.text.trim_start();
                    if stripped.is_empty() {
                        message_delta.text.clear();
                    } else if stripped.len() == message_delta.text.len() {
                        // No leading whitespace to strip in this delta; the
                        // next delta carrying text is no longer the first.
                        strip_leading_whitespace = false;
                    } else {
                        message_delta.text = stripped.to_string();
                        strip_leading_whitespace = false;
                    }
                }
                saw_tool_delta |= message_delta.tool_call.is_some();
                // Build the middleware-facing delta first (it needs owned
                // copies of the fields), then move `message_delta` into the
                // event so the hot path clones the payload once instead of
                // twice per streamed token.
                let mut model_delta = ModelDelta {
                    call_id: call_id.as_str().to_string(),
                    content: message_delta.text.clone(),
                    reasoning: message_delta.reasoning.clone(),
                    tool_call: message_delta.tool_call.clone(),
                };
                self.middleware
                    .run_on_model_delta(ctx, state, &mut model_delta)
                    .await?;
                // A tool call middleware added counts as one the call has,
                // for this delta and every later one.
                saw_tool_delta |= model_delta.tool_call.is_some();
                if model_delta.tool_call.is_some() {
                    stream_stall.reset();
                } else if stream_stall.observe(&model_delta.content) {
                    tracing::warn!("[stream] stopped repetitive model narration");
                    return Err(TinyAgentsError::GenerationStalled);
                }
                saw_streamed_content |= !message_delta.text.is_empty()
                    || !message_delta.reasoning.is_empty()
                    || !model_delta.content.is_empty()
                    || !model_delta.reasoning.is_empty();
                streamed_text.push_str(&model_delta.content);
                streamed_reasoning.push_str(&model_delta.reasoning);
                streamed_reasoning_chars += model_delta.reasoning.chars().count() as u64;
                // Reasoning past the bound with nothing visible yet: end the
                // call here as the dead call it was going to be, instead of
                // waiting for the provider to reach the output cap. Dropping
                // `stream` on return closes the connection. The loop sees the
                // same response a cap-truncated call produces and runs its
                // truncated-empty recovery.
                if let Some(bound) = reasoning_bound
                    && streamed_text.trim().is_empty()
                    && !saw_tool_delta
                    && message_delta.tool_call.is_none()
                    && model_delta.tool_call.is_none()
                    && estimated_reasoning_tokens(streamed_reasoning_chars) > u64::from(bound)
                {
                    let estimated = estimated_reasoning_tokens(streamed_reasoning_chars);
                    let elapsed_ms = watchdog_started.elapsed().as_millis() as u64;
                    tracing::warn!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        call_id = %call_id,
                        bound,
                        estimated_reasoning_tokens = estimated,
                        elapsed_ms,
                        "[stream] reasoning watchdog ended a call that reasoned past its budget with nothing visible"
                    );
                    ctx.emit(AgentEvent::ControlApplied {
                        control: "reasoning_watchdog".to_string(),
                        detail: format!(
                            "model call `{call_id}` reasoned past its {bound}-token budget (about \
                             {estimated} tokens in {elapsed_ms} ms) with no visible output; the call \
                             was ended and is treated as a dead call"
                        ),
                    });
                    return Ok(watchdog_dead_response(
                        estimated,
                        std::mem::take(&mut streamed_reasoning),
                    ));
                }
                let forwarded_delta = MessageDelta {
                    text: model_delta.content.clone(),
                    reasoning: model_delta.reasoning.clone(),
                    tool_call: model_delta.tool_call.clone(),
                };
                ctx.emit(AgentEvent::ModelDelta {
                    run_id: ctx.config.run_id.clone(),
                    call_id: call_id.clone(),
                    delta: forwarded_delta,
                });
                crate::runtime::emit_host_progress::<State, Ctx>(
                    ctx,
                    crate::host::ProgressEvent::Token {
                        run: ctx.run_id().clone(),
                        text: model_delta.content.clone(),
                    },
                );
                item = match item {
                    ModelStreamItem::MessageDelta(_) => {
                        ModelStreamItem::MessageDelta(MessageDelta {
                            text: model_delta.content,
                            reasoning: model_delta.reasoning,
                            tool_call: model_delta.tool_call,
                        })
                    }
                    ModelStreamItem::ToolCallDelta(_) => model_delta.tool_call.map_or_else(
                        || {
                            // The middleware deliberately suppressed the raw
                            // tool fragment.  Keep a content-free item so the
                            // accumulator cannot reconstruct or dispatch it.
                            ModelStreamItem::MessageDelta(MessageDelta::default())
                        },
                        ModelStreamItem::ToolCallDelta,
                    ),
                    _ => item,
                };
                if matches!(
                    item,
                    ModelStreamItem::ToolCallDelta(_)
                        | ModelStreamItem::MessageDelta(MessageDelta {
                            tool_call: Some(_),
                            ..
                        })
                ) {
                    transformed_tools.push(&item);
                }
                *deltas_emitted += 1;
            }

            // Arm the watchdog on provider output: a non-empty raw delta,
            // measured before scrubbing, leading-whitespace removal and delta
            // middleware (see `raw_delta_is_output`), so a payload that a
            // later stage consumed or rewrote still proves the provider is
            // alive while text a middleware injected into an empty delta does
            // not. Block-aware adapters do not necessarily emit the
            // compatibility MessageDelta, so their non-empty payloads count
            // as progress too.
            let is_output = match &item {
                ModelStreamItem::MessageDelta(_) | ModelStreamItem::ToolCallDelta(_) => {
                    raw_delta_is_output
                }
                ModelStreamItem::BlockDelta { delta, .. } => match delta {
                    tinyinference_llm::model::BlockDelta::Text(text)
                    | tinyinference_llm::model::BlockDelta::Thinking(text)
                    | tinyinference_llm::model::BlockDelta::ToolArgs(text) => !text.is_empty(),
                },
                ModelStreamItem::BlockEnd { block, .. } => match block {
                    tinyinference_llm::message::ContentBlock::Text(text)
                    | tinyinference_llm::message::ContentBlock::Thinking { text, .. }
                    | tinyinference_llm::message::ContentBlock::RedactedThinking { data: text } => {
                        !text.is_empty()
                    }
                    _ => true,
                },
                ModelStreamItem::Completed(_) => true,
                _ => false,
            };
            if is_output {
                saw_output = true;
                rearm = true;
                ctx.limits.reset_stream_idle_timeouts_for(model_name);
            }

            // Reconcile even when nothing ordinary streamed: a response that
            // is *purely* text-dialect tool-call markup suppresses every
            // delta (so `saw_streamed_content` stays false) but still needs
            // its raw `<tool_call>`-style text replaced — otherwise that raw
            // markup survives in the terminal response's content block
            // alongside the structured calls the scrubber recovered below,
            // and gets persisted into the transcript to be replayed back to
            // the model next turn.
            // A withheld call (a turn with no callable tool) was scrubbed the
            // same way, so its raw markup must not come back from the
            // terminal text either.
            let scrubber_recovered_calls = text_scrubber
                .as_ref()
                .is_some_and(|scrubber| scrubber.has_calls() || scrubber.has_withheld());
            if let ModelStreamItem::Completed(response) = &mut item
                && (saw_streamed_content || scrubber_recovered_calls)
            {
                // Deltas represent only text/thinking, so preserve terminal
                // blocks that cannot be streamed as a `ModelDelta` (JSON,
                // images, and provider extensions).
                //
                // A signed `Thinking` block must be replayed *verbatim* on
                // the next model call when thinking + tool calls are both in
                // play (Anthropic requires the exact signed block ahead of a
                // `tool_use`); synthesizing a fresh, unsigned block here would
                // make that replay fail. So the terminal provider blocks are
                // kept as-is unless a delta middleware actually rewrote the
                // reasoning text: compare the concatenated `Thinking` text
                // that crossed `on_model_delta` against the terminal
                // response's own `Thinking` text. Equal means no middleware
                // touched it — keep the terminal blocks (signature intact).
                // Different means the delta stream was transformed — fall
                // back to a synthetic, unsigned block built from what
                // actually crossed the middleware boundary, same as before.
                // `RedactedThinking` carries no reasoning text at all (it is
                // opaque), so it is always kept verbatim.
                let terminal_reasoning: String = response
                    .message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        tinyinference_llm::message::ContentBlock::Thinking { text, .. } => {
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect();
                let reasoning_untransformed = terminal_reasoning == streamed_reasoning;

                let mut content = Vec::new();
                if reasoning_untransformed {
                    content.extend(
                        response
                            .message
                            .content
                            .iter()
                            .filter(|block| {
                                matches!(
                                    block,
                                    tinyinference_llm::message::ContentBlock::Thinking { .. }
                                )
                            })
                            .cloned(),
                    );
                    streamed_reasoning.clear();
                } else if !streamed_reasoning.is_empty() {
                    content.push(tinyinference_llm::message::ContentBlock::Thinking {
                        text: std::mem::take(&mut streamed_reasoning),
                        signature: None,
                    });
                }
                if !streamed_text.is_empty() {
                    content.push(tinyinference_llm::message::ContentBlock::Text(
                        std::mem::take(&mut streamed_text),
                    ));
                }
                content.extend(response.message.content.drain(..).filter(|block| {
                    !matches!(
                        block,
                        tinyinference_llm::message::ContentBlock::Text(_)
                            | tinyinference_llm::message::ContentBlock::Thinking { .. }
                    )
                }));
                response.message.content = content;
            }
            if let ModelStreamItem::Completed(response) = &mut item
                && text_scrubber
                    .as_ref()
                    .is_some_and(super::dialect::DeltaScrubber::has_calls)
                && let Some(scrubber) = text_scrubber.take()
            {
                // The streamed text held complete tool-call blocks, scrubbed
                // from the reconciled text above, so this is the only place
                // they can be dispatched from. Appended, not assigned: a
                // provider can legitimately return a native structured call
                // *and* narrate a second one as text in the same turn, and
                // gating this on `tool_calls.is_empty()` used to silently
                // drop the narrated one whenever a native call was present.
                response.message.tool_calls.extend(scrubber.into_calls());
            }
            if let ModelStreamItem::Completed(response) = &mut item
                && saw_tool_delta
            {
                // The terminal response carries richer metadata, but its raw
                // tool calls are not authoritative after delta middleware has
                // transformed them. Reconstruct from the exact forwarded
                // fragments so a changed name, id, or argument payload cannot
                // be restored just before dispatch.
                response.message.tool_calls =
                    transformed_tools.clone().finish()?.message.tool_calls;
            }

            accumulator.push(&item);
        }

        let mut response = accumulator.finish()?;
        split_thinking_tags(&mut response, model.profile());
        Ok(response)
    }
}

/// Splits a model's inline `<open>...</close>`-tagged reasoning span out of a
/// `Text` content block into a dedicated [`ContentBlock::Thinking`] block,
/// for a model whose [`ModelProfile::thinking_tags`] declares the tag pair it
/// emits inline instead of on a distinct reasoning channel.
///
/// A no-op when the profile declares no tag pair, or the response carries no
/// text block containing both tags. Only the first tagged span in each text
/// block is extracted — every provider that uses this convention emits at
/// most one reasoning span ahead of the visible answer — and any text before
/// or after the span is preserved as ordinary `Text` blocks in the same
/// position.
///
/// [`ContentBlock::Thinking`]: tinyinference_llm::message::ContentBlock::Thinking
/// [`ModelProfile::thinking_tags`]: tinyinference_llm::model::ModelProfile::thinking_tags
pub(super) fn split_thinking_tags(
    response: &mut tinyinference_llm::model::ModelResponse,
    profile: Option<&tinyinference_llm::model::ModelProfile>,
) {
    let Some((open, close)) = profile.and_then(|p| p.thinking_tags.as_ref()) else {
        return;
    };
    if open.is_empty() || close.is_empty() {
        return;
    }

    let mut rebuilt = Vec::with_capacity(response.message.content.len());
    for block in response.message.content.drain(..) {
        match block {
            tinyinference_llm::message::ContentBlock::Text(text) => {
                match split_one(&text, open, close) {
                    Some((before, thinking, after)) => {
                        if !before.is_empty() {
                            rebuilt.push(tinyinference_llm::message::ContentBlock::Text(before));
                        }
                        if !thinking.is_empty() {
                            rebuilt.push(tinyinference_llm::message::ContentBlock::Thinking {
                                text: thinking,
                                signature: None,
                            });
                        }
                        if !after.is_empty() {
                            rebuilt.push(tinyinference_llm::message::ContentBlock::Text(after));
                        }
                    }
                    None => rebuilt.push(tinyinference_llm::message::ContentBlock::Text(text)),
                }
            }
            other => rebuilt.push(other),
        }
    }
    response.message.content = rebuilt;
}

/// Splits `text` on the first `open`/`close` tag pair, returning
/// `(before, inside, after)` with the tags themselves removed and each
/// segment trimmed of the whitespace/newlines the tags typically pad. `None`
/// when the text does not contain a complete `open`...`close` span.
fn split_one(text: &str, open: &str, close: &str) -> Option<(String, String, String)> {
    let open_idx = text.find(open)?;
    let after_open = open_idx + open.len();
    let close_rel = text[after_open..].find(close)?;
    let close_idx = after_open + close_rel;
    let before = text[..open_idx].trim().to_string();
    let inside = text[after_open..close_idx].trim().to_string();
    let after = text[close_idx + close.len()..].trim().to_string();
    Some((before, inside, after))
}
/// The innermost model call wrapped by the model-wrap onion.
///
/// Implements [`ModelBaseCall`] over the harness's cache + retry + fallback core
/// ([`AgentHarness::invoke_model_with_retry`]) so a [`crate::middleware::ModelMiddleware`]
/// can proceed, short-circuit, retry, or fall back around the *whole* real model
/// call. The resolved binding is rebuilt per invocation so a wrap middleware
/// that retries `next` issues a fresh provider call each time.
///
/// # The binding is re-resolved from the request
///
/// [`ModelCallBase::call`] used to rebuild the binding purely from
/// [`Self::resolved`] / [`Self::model`], both captured **before** the wrap onion
/// ran, and ignore [`ModelRequest::model`] entirely. A wrap middleware steers by
/// mutating that field — it is the only lever it has — so
/// [`ModelFallbackMiddleware`][crate::middleware::ModelFallbackMiddleware]
/// re-invoked *the same failing model* once per configured fallback name,
/// emitting a misleading `FallbackSelected { from, to }` for each, and then
/// returned the original error. The asymmetry was easy to miss because
/// `before_model` **does** honour `request.model`: lifecycle resolution happens
/// after that hook, but before this one.
pub(super) struct ModelCallBase<'h, State: Send + Sync, Ctx: Send + Sync> {
    pub(super) harness: &'h AgentHarness<State, Ctx>,
    pub(super) call_id: CallId,
    pub(super) resolved: ResolvedModel,
    pub(super) model: Arc<dyn ChatModel<State>>,
    pub(super) required_capabilities: Option<tinyinference_llm::model::CapabilitySet>,
    pub(super) shape: super::dialect::CallShape,
}

impl<State: Send + Sync, Ctx: Send + Sync> ModelCallBase<'_, State, Ctx> {
    /// Produces the binding for one invocation, honouring a model override that
    /// a wrap middleware wrote into `request.model`.
    ///
    /// * `request.model` absent, or equal to the already-resolved name: reuse
    ///   the captured binding (the common path — no registry lookup).
    /// * `request.model` names something the registry resolves as a genuine
    ///   [`ModelResolutionSource::RequestOverride`]: use it. This is what makes
    ///   a wrap-layer fallback actually switch models.
    /// * `request.model` names something unresolvable (unregistered, missing a
    ///   required capability, provider-retired): fall back to the captured
    ///   binding and emit [`AgentEvent::ModelOverrideSkipped`], matching the
    ///   fail-closed behaviour `run_loop` already has for a pre-wrap override.
    ///   Silently substituting a different model is the one outcome that is
    ///   never acceptable.
    async fn rebind(
        &self,
        ctx: &mut RunContext<Ctx>,
        request: &ModelRequest,
    ) -> Result<ResolvedModelBinding<State>> {
        let captured = || ResolvedModelBinding {
            resolved: self.resolved.clone(),
            model: Arc::clone(&self.model),
        };
        let needs_host_resolution =
            request.model.is_some() || request.required_capabilities.is_some();
        if !needs_host_resolution {
            return Ok(captured());
        }
        let requested = request.model.as_deref();
        if request.required_capabilities == self.required_capabilities
            && (requested.is_none()
                || requested.is_some_and(|requested| {
                    self.resolved.source == ModelResolutionSource::RequestOverride
                        && self.resolved.requested.as_deref() == Some(requested)
                }))
        {
            return Ok(captured());
        }
        if let Some(binding) = self.harness.resolve_host_model(ctx, request).await? {
            return Ok(binding);
        }
        let Some(requested) = requested else {
            return Ok(captured());
        };
        if requested == self.resolved.name {
            return Ok(captured());
        }
        match self.harness.models.resolve_request(request, None, None) {
            Some(binding)
                if binding.resolved.source == ModelResolutionSource::RequestOverride
                    && binding.resolved.name == requested =>
            {
                tracing::debug!(
                    call_id = %self.call_id.as_str(),
                    from = %self.resolved.name,
                    to = %binding.resolved.name,
                    "[model] wrap layer overrode the model; re-resolved the binding"
                );
                Ok(binding)
            }
            _ => {
                tracing::warn!(
                    call_id = %self.call_id.as_str(),
                    requested = %requested,
                    resolved = %self.resolved.name,
                    "[model] wrap layer named an unresolvable model; keeping the resolved binding"
                );
                ctx.emit(AgentEvent::ModelOverrideSkipped {
                    requested: requested.to_string(),
                    resolved: self.resolved.name.clone(),
                });
                Ok(captured())
            }
        }
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> ModelBaseCall<State, Ctx>
    for ModelCallBase<'_, State, Ctx>
{
    fn call<'a>(
        &'a self,
        ctx: &'a mut RunContext<Ctx>,
        state: &'a State,
        request: ModelRequest,
    ) -> BoxModelFuture<'a> {
        Box::pin(async move {
            let mut request = request;
            let binding = self.rebind(ctx, &request).await?;
            // Claim the switch only after the wrap onion has elected to call
            // this base. A wrap middleware may short-circuit with a command or
            // replacement response without invoking the model at all.
            self.harness.announce_applied_model_switch(ctx, &request);
            crate::middleware::library::rehome_ephemeral_system_instructions(
                &mut request,
                binding.model.profile(),
            );
            super::run_loop::refresh_prompt_cache_fingerprint(&mut request);
            // Dropped-block counts describe the response this call returns.
            // Clear them first (a cache hit makes no attempt), and again on
            // failure: a wrap middleware may answer in place of the failed
            // attempt, and that answer attempted no call.
            self.shape.recovery.dropped.reset();
            ctx.call_streamed = self.shape.streaming;
            let result = self
                .harness
                .invoke_model_with_retry(state, ctx, &request, &self.call_id, binding, &self.shape)
                .await;
            if result.is_err() {
                self.shape.recovery.dropped.reset();
            }
            result
        })
    }
}

/// The innermost tool call wrapped by the tool-wrap onion.
///
/// Implements [`ToolBaseCall`] over a single resolved [`Tool`] so a
/// [`crate::middleware::ToolMiddleware`] can wrap the real tool
/// invocation.
pub(super) struct ToolCallBase<'h, State: Send + Sync, Ctx: Send + Sync> {
    /// Services the nested calls the tool makes (C9).
    pub(super) harness: &'h AgentHarness<State, Ctx>,
    /// Nesting level of this call: `0` for a model-issued call, `n` for a call
    /// nested `n` deep.
    pub(super) level: usize,
    pub(super) dispatch: Arc<dyn crate::tool::ToolDispatch<State, Ctx>>,
    pub(super) options: tinytools::ToolCallOptions,
    pub(super) timeout_settings: Option<crate::tool::ToolTimeoutSettings>,
    /// Nested-call ids, refusal budget and summaries for this logical call,
    /// kept across every attempt a wrap middleware makes (retries).
    pub(super) nested_state: super::nested::NestedState,
    /// Whether this call already runs under the run-wide nested serialization
    /// gate, so its own nested calls must not take it again.
    pub(super) gate_held: super::nested::GateHold,
}

impl<State: Send + Sync, Ctx: Send + Sync> ToolBaseCall<State, Ctx>
    for ToolCallBase<'_, State, Ctx>
{
    fn call<'a>(
        &'a self,
        ctx: &'a RunContext<Ctx>,
        state: &'a State,
        call: ToolCall,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            let timeout = self.timeout_settings.as_ref().map(|settings| {
                settings.resolve(self.dispatch.tool().timeout_policy(&call.arguments))
            });
            let timeout_result = super::tools::timeout_result(&call, timeout);
            let nested = super::nested::NestedCalls::new(
                self.harness,
                ctx,
                state,
                CallId::new(call.id.clone()),
                self.level,
                &self.nested_state,
                self.gate_held,
            );
            let future = super::tools::execute_tool_recovering_model_retry(self.dispatch.execute(
                state,
                CallId::new(call.id),
                call.arguments,
                self.options,
                ctx,
            ));
            let bounded = async {
                match timeout.and_then(|resolved| resolved.deadline) {
                    Some(deadline) => match tokio::time::timeout(deadline, future).await {
                        Ok(result) => result,
                        Err(_) => Ok(timeout_result),
                    },
                    None => future.await,
                }
            };
            let mut result = nested.drive(bounded).await?;
            nested.attach_summary(&mut result);
            Ok(result)
        })
    }
}

/// Reasoning tokens a streamed reasoning text of `reasoning_chars` characters
/// (not UTF-8 bytes: a CJK character is three bytes and about one token)
/// amounts to, estimated at three characters per token. Measured on deepseek-v4.1-flash, whose reasoning is
/// dense with code, numbers and short tokens, 36k characters were about 13k
/// tokens (2.7 per token); English prose runs nearer four. Three keeps the
/// estimate on the low side for this kind of text, so a bound built on it
/// fires a little late rather than early.
fn estimated_reasoning_tokens(reasoning_chars: u64) -> u64 {
    reasoning_chars / 3
}

/// The response the reasoning watchdog hands the loop in place of the call it
/// ended: the shape a call truncated at its output cap has (`finish_reason =
/// length`, no visible text, no tool call), with the estimated reasoning as
/// its output usage so the recovery can judge the call's rate, and the
/// reasoning that streamed kept as a `Thinking` block so the recovery can
/// carry it forward (`RunPolicy::truncated_empty_carry_reasoning_chars`).
fn watchdog_dead_response(estimated_reasoning_tokens: u64, reasoning: String) -> ModelResponse {
    let usage = tinyinference_llm::Usage::new(0, estimated_reasoning_tokens);
    let content = if reasoning.is_empty() {
        Vec::new()
    } else {
        vec![tinyinference_llm::message::ContentBlock::Thinking {
            text: reasoning,
            signature: None,
        }]
    };
    ModelResponse {
        message: tinyinference_llm::message::AssistantMessage {
            id: None,
            content,
            tool_calls: Vec::new(),
            usage: Some(usage),
            origin: None,
        },
        usage: Some(usage),
        finish_reason: Some("length".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

#[cfg(test)]
#[path = "model_call_failover_tests.rs"]
mod failover_test;

#[cfg(test)]
#[path = "model_call_watchdog_tests.rs"]
mod watchdog_tests;

/// Retargets an attempt's wire-level `request.model` at a fallback binding,
/// but only when the request already carried an explicit model. Registry names
/// are runtime aliases, not guaranteed provider model ids, so a request that
/// sent none (the provider's own configured model) keeps sending none.
fn retarget_request_model(request: &mut ModelRequest, name: &str) {
    if request.model.is_some() {
        request.model = Some(name.to_string());
    }
}
