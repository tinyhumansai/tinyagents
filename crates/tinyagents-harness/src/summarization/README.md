# harness::summarization

Explicit message trimming, summarization, and compression policies — the
harness's direct answer to "context rot." Context-window-aware gating
(`SummarizationPolicy`) decides *when* a run's transcript has grown large
enough to compress; the trimming/summarization primitives decide *what* to
keep verbatim versus fold into a summary. Every policy decision is an
explicit, inspectable data type: callers choose when to call, what to pass,
and how to handle the result — nothing here runs implicitly inside the agent
loop.

## Design

- **Token estimation** (`estimate_tokens`, `TokenEstimate`) is a cheap
  `chars / 4` heuristic, not a real tokenizer — good for budget checks with a
  ~30% error margin, not for exact accounting.
- **Trimming** (`trim.rs`) is synchronous and LLM-free: it drops messages
  from a slice according to a `TrimStrategy` (`KeepLast`, `KeepFirstAndLast`,
  `MaxTokens`). Every strategy is pairing-safe by default via
  `TrimOptions`/`TokenTrimPolicy`.
- **Summarization** (`Summarizer` trait, `ConcatSummarizer`,
  `SummarizationPolicy`) is async and typically LLM-backed.
  `SummarizationPolicy::plan` decides the split between `to_summarize` and
  `to_keep`; a `Summarizer` then condenses the former into a `SummaryRecord`
  carrying `CompressionProvenance`.
- **Dispatch tracking** (`dispatch.rs`): summarizer calls bypass the run
  context's provider-dispatch marker. `ContextCompressionMiddleware` scopes each
  summarization with `track_dispatch`, `ModelSummarizer` calls `mark_dispatched`
  right before it invokes its model, and the middleware then sets the run's
  `provider_started`, so a summarizer that fails without usage after dispatching
  still reports `TerminalOutcome::provider_started`. A rejection before dispatch
  does not.
- **Tool-call pairing** (`pairing.rs`) is the structural safety net both of
  the above rely on: a naive length-based cut point routinely separates an
  assistant tool-call turn from the tool results answering it, producing a
  transcript OpenAI/Anthropic reject outright. `find_safe_cutoff_point` and
  friends move the cut to preserve pairing (or repair an already-broken one).
- **Rendering** (`render.rs`) turns any `Message` — including tool calls,
  tool results, and reasoning blocks that `Message::text()` drops — into
  summarizable text, so `ConcatSummarizer`'s default output isn't a column of
  bare role labels for a tool-driven run.
- **Compaction** (`compaction.rs`, `pub mod compaction`) is the durable,
  rule-driven layer: `find_cut_point` (token-budget cut points, repaired via
  `pairing.rs`), `summarize_with_split` (split-turn summarization + merge),
  `Summarizer::summarize_request`/`SummaryRequest::previous_summary`
  (iterative summaries), `CompactionRecord`/`CompactionSink`/
  `CompactionReason`, `CompactionContext`/`CompactionDecision` (the
  `before_compaction` hook), and `OverflowClassifier` (table-driven provider
  overflow detection). `ContextCompressionMiddleware` uses all of this for
  its `before_model` (threshold) and `wrap_model` (overflow → compact →
  retry) paths. Full contract:
  `docs/modules/harness/compaction.md`.

## Public surface

### Token estimation
- `estimate_tokens(text) -> u64` / `TokenEstimate` — the heuristic counter.

### Trimming (`trim.rs`)
- `trim_messages(messages, strategy) -> Vec<Message>` — the common case,
  equivalent to `trim_messages_with(.., &TrimOptions::default())`.
- `trim_messages_with` / `trim_messages_to_token_budget_with` — the
  fully-configurable entry points (`TrimOptions`, `TokenTrimPolicy`).
- `TrimStrategy` — `KeepLast(n)` / `KeepFirstAndLast { first, last }` /
  `MaxTokens(limit)`.
- `TrimOptions` — `repair_tool_pairs` (default on — keep it on),
  `start_on`/`end_on` role boundaries (`starting_on`/`ending_on`/
  `without_pair_repair` builders).
- `TokenTrimPolicy` — `strict`/`preserve_system`/`drop_leading_orphan_tools`
  builders for `trim_messages_to_token_budget_with`.
- `MessageRole` — role-only view of a `Message`, used by the `start_on`/
  `end_on` boundaries.

### Summarization
- `Summarizer` (trait, object-safe) — `summarize(&self, messages) ->
  Result<SummaryRecord>`.
- `ConcatSummarizer` — deterministic, LLM-free default: concatenates
  `render_message_for_summary` output with positional `msg-N` ids.
- `SummarizationPolicy` — `from_profile`/`with_context_window`/
  `with_threshold_fraction` builders; `trigger_budget()`,
  `should_summarize(messages)`, and `plan(messages) -> (to_summarize,
  to_keep)`.
- `SummaryRecord` — `summary: Message` + `provenance: CompressionProvenance`.
- `CompressionProvenance` — `source_ids`, `original_token_estimate`,
  `summary_token_estimate`, `reason`.

### Tool-call pairing (`pairing.rs`, `pub mod pairing`)
- `find_safe_cutoff_point`, `tool_pairing_is_intact`,
  `retract_orphan_tool_calls`, `advance_past_orphan_tools`,
  `is_tool_calling_assistant` — re-exported at `summarization::*` as well as
  reachable via `summarization::pairing::*`.

### Rendering
- `render_message_for_summary(&Message) -> String`.
- `render_message_for_summary_with(&Message, ToolRendering) -> Option<String>`
  — the same, in a chosen mode; `None` when the mode omits the message.
- `ToolRendering` — `Full` (default: tool calls with ids and arguments, tool
  results elided) or `Markers` (conversation only: each tool call is one
  `[used <name>: <arguments>]` line, arguments cut to 120 chars with an
  ellipsis, no call ids; tool results omitted).

### `ModelSummarizer`
- `with_tool_rendering(ToolRendering)` — the transcript's rendering mode;
  `Full` unless set.
- `with_max_output_tokens(u32)` — sends `max_tokens` on the summary request.
  Unset by default; the request sets no sampling parameters of its own.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `estimate_tokens`, `Summarizer for ConcatSummarizer`, `SummarizationPolicy` impls; wires the submodules together. |
| `types.rs` | All public struct/enum/trait definitions. |
| `pairing.rs` | Tool-call-pairing-safe cut-point repair. |
| `render.rs` | `render_message_for_summary`, `render_message_for_summary_with`. |
| `render_tests.rs` | Coverage for both rendering modes and marker truncation. |
| `trim.rs` | `trim_messages`/`trim_messages_with`/`trim_messages_to_token_budget_with`. |
| `compaction.rs` | `find_cut_point`, `summarize_with_split`, `OverflowClassifier`, `CompactionContext`/`CompactionDecision`. |
| `response_overflow.rs` | `detect_response_overflow`, `ResponseOverflowDetection`: overflow read from a successful response (usage above the window, zero-output `length` stop, opt-in short `length` stop). |
| `file_ops.rs` | `FileOpExtractor`, `DefaultFileOpExtractor`, `FileOperations`, and the `<read-files>` / `<modified-files>` section writer/parser carried by compaction summaries. |
| `split_turn.rs` | `split_turn_start`, `summarize_split_turn`: a cut inside a turn summarizes the turn prefix with its own `SummaryKind::TurnPrefix` request. |
| `mod_tests.rs` | Coverage for token estimation, trim strategies, pairing repair, policy triggering/planning, and `ConcatSummarizer`. |
| `compaction/compaction_tests.rs` | Coverage for cut points, split-turn merge, iterative summaries, `OverflowClassifier`. |

## Key invariants

- **`repair_tool_pairs` (on `TrimOptions`) should stay on.** Turning it off
  restores transcripts providers reject outright (`400` on a stray
  `role:"tool"`, or a `tool_result` with no matching `tool_use`); it exists
  only for callers that repair pairing themselves or tests observing the
  unrepaired cut.
- **`SummarizationPolicy::plan`'s split point is never a blind
  `len - keep_last` index.** It always routes through
  `find_safe_cutoff_point`, so `keep_last` is a *minimum*, not an exact
  count, and `to_keep` is always a provider-acceptable slice.
- **`keep_recent_tokens` replaces the `keep_last` count when set.** The tail
  is then cut by `find_cut_point` (pairing-repaired, never empty).
- **`pin_turn_user_message` keeps the turn's assignment verbatim.** When the
  kept tail holds no user message, `plan` moves the most recent one out of the
  summarized head to the front of the tail (capped at
  `PINNED_USER_MESSAGE_MAX_TOKENS`, truncated with a marker beyond).
  `plan_split` reports its index; `ContextCompressionMiddleware` re-applies it
  with the fold and records it as `details.pinned_user_index` on the
  `CompactionRecord`, and the `FallbackTrim` front-drop keeps it too.
- **System messages are never placed in `to_summarize`** and are (by
  default) never dropped by trimming either — they carry persistent
  instructions.
- **`ConcatSummarizer` renders through `render_message_for_summary`, not
  `Message::text()`**, specifically so a tool-driven run's compacted history
  keeps its tool calls/results/reasoning instead of collapsing to empty
  strings.
- **`ModelSummarizer` fences the transcript and asks last.** Its user message
  is `<transcript>…</transcript>` (after any `<previous_summary>`), then the
  instruction. A bare transcript that ends on a tool result reads as a live
  agent loop, and a model continues it with its next tool call. The request
  declares no tools, so that call arrives as plain-text markup (DeepSeek V4:
  `<｜DSML｜invoke …>`).
- **`ToolRendering::Markers` drops tool output, never the conversation.**
  User, assistant and system text render exactly as under `Full`; only tool
  results are omitted and tool calls shortened to one marker line each, so a
  small instruction-light summarization model sees what was said and done.
  `Full` stays the default, byte-identical to `render_message_for_summary`.
- **A tool call is never accepted as a summary.** A reply the tool-call
  grammars parse as a call is retried once, then returned as an error, so
  `FaultTolerantCachingSummarizer` falls back to its deterministic trim
  instead of installing one stray command as the whole compacted history.
- This module never calls into the agent loop or decides *when* to invoke a
  `Summarizer` on its own — `should_summarize`/`plan` are pure decisions the
  caller (the agent loop, a middleware) acts on.

## Relation to neighbouring modules

`SummarizationPolicy::from_profile` reads
`tinyinference_llm::model::ModelProfile::max_input_tokens`. `token_estimation::estimate_slice_tokens`
(`crate::token_estimation`) backs the whole-slice estimates used by
`should_summarize`/`trigger_budget`. Consumers are the agent loop / context
middleware that decide when a run's transcript needs compacting before the
next model call.
