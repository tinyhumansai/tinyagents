# Codebase Audit

Date: 2026-06-29

This audit reviewed the current TinyAgents codebase for correctness gaps,
security issues, implementation weaknesses, and spec-contract mismatches. No
code changes were made as part of the audit.

## Summary

The crate is in a healthy build state: the build passes (the OpenAI adapter is
compiled in by default), the full test suite passes, and clippy is clean. The main issues are not broad
compile failures; they are fail-closed contract gaps around durable graph
recovery and observability/runtime hardening.

## Findings

## Resolved Findings

### Resolved: tool arguments are schema-validated before execution

The agent loop now validates model-supplied tool arguments against the
registered tool schema before emitting `ToolStarted` or invoking tool code:

- `crates/tinyagents-harness/src/tool/mod.rs`: `ToolSchema::validate_call` enforces the local JSON
  Schema subset used at the harness boundary: `type`, object `properties`,
  `required`, `additionalProperties: false`, array `items`, and `enum`.
- `crates/tinyagents-harness/src/agent_loop/mod.rs`: after middleware has had its chance to adjust
  the call, the loop validates the final call against the tool schema before
  wrapping/executing the tool.
- `crates/tinyagents-harness/src/tool/test.rs`: unit coverage exercises valid nested arguments,
  missing required fields, wrong types, and disallowed extra fields.
- `crates/tinyagents-harness/src/agent_loop/test.rs`: end-to-end harness coverage proves invalid
  arguments return a validation error before the tool implementation is called.

### Resolved (superseded): malformed OpenAI tool-call JSON is recovered, not failed closed

**Note (2026-09-19):** provider adapters, including OpenAI, have moved out of
this crate into `vendor/tinyinference/crates/tinyinference-llm`;
`crates/tinyagents-harness/src/providers/` now holds only `claude_agent_sdk/`
and `claude_code/`, so the file paths originally cited here no longer exist.
The intended *behavior* has also inverted since this entry was written:
malformed stringified tool arguments no longer fail the run. The provider
marks the call `ToolCall::invalid` (raw arguments preserved, with a parse
reason) instead of converting it to `null`:

- `vendor/tinyinference/crates/tinyinference-llm/src/providers/openai/convert.rs` (~403-436):
  unary response parsing turns unparseable stringified arguments into a
  `ToolCall::invalid` call rather than an error or `null`.
- `vendor/tinyinference/crates/tinyinference-llm/src/providers/openai/sse.rs` (~224, ~462):
  streamed tool-call reconstruction does the same for assembled arguments
  that fail to parse as JSON.
- `crates/tinyagents-harness/src/agent_loop/tools.rs` (~276-287): the agent
  loop checks `call.invalid` and, rather than aborting, injects a tool-error
  result carrying the parse detail and raw arguments so the model can retry
  with corrected JSON — the same recovery path used for schema-invalid
  arguments and unknown tool calls. This is applied unconditionally
  (independent of `InvalidArgsPolicy`, which only governs schema validation of
  well-formed arguments) and always resolves the call, so it cannot hang the
  loop.

In short: malformed provider-side tool-call JSON now fails *open* by design
(recovered as a tool error the model can act on), not closed.

### Resolved: required model capabilities are enforced during resolution

Model resolution now treats `ModelRequest::required_capabilities` as a hard
candidate filter:

- `crates/tinyagents-harness/src/model/types.rs`: `ModelSelection` carries the required
  capability set so direct registry resolution and request-based resolution use
  the same filter.
- `crates/tinyagents-harness/src/model/mod.rs`: request overrides, previous-model reuse, hints,
  agent defaults, and registry defaults are skipped unless their profile
  satisfies the required capabilities. Models with unknown profiles fail
  conservatively when non-empty requirements are present.
- `crates/tinyagents-harness/src/model/test.rs`: unit coverage exercises request overrides,
  previous reuse, hints, agent defaults, registry defaults, and unknown-profile
  rejection under required capabilities.
- `tests/harness_agent_loop.rs`: integration coverage proves the agent loop can
  select a capable hinted model over an incapable default after middleware adds
  required capabilities.

### Resolved: runtime `Command::goto` targets are validated before persistence

Runtime command routing now fails before a bad target can become the next active
set or be written into a checkpoint:

- `crates/tinyagents-graph/src/compiled/mod.rs`: command targets are validated in `route()` before
  scheduler activation. `END` is allowed, `START` is rejected, and every other
  target must name a compiled node.
- `crates/tinyagents-graph/src/compiled/test.rs`: unit coverage rejects unknown targets, rejects
  `START`, and proves invalid command routes fail before checkpoint persistence.
- `tests/graph_durable.rs`: integration coverage proves a threaded durable run
  with an invalid command target writes no poisoned checkpoint.
- `docs/modules/graph/routing.md`: documents the runtime validation contract.

### Resolved: embedded subgraph runs inherit the parent thread id

Subgraph node adapters now propagate the parent thread id into embedded child
runs when the parent run is threaded:

- `crates/tinyagents-graph/src/subgraph/mod.rs`: shared-state and adapter subgraph nodes call the
  child with `run_with_thread` when `NodeContext::thread_id` is present,
  preserving the existing unthreaded `run` behavior otherwise.
- `crates/tinyagents-graph/src/subgraph/test.rs`: unit coverage proves an embedded child with a
  checkpointer persists under the parent thread and embedding-node namespace.
- `tests/graph_durable.rs`: integration coverage proves the public durable graph
  surface writes both parent and child checkpoints for a threaded subgraph run.
- `docs/modules/graph/subgraphs.md`: documents the inherited thread and
  namespace persistence contract.

### Resolved: interrupts fail closed without resumable durability

The executor now returns a resume error instead of an interrupted execution when
a node emits an interrupt without the durability required to resume it:

- `crates/tinyagents-graph/src/compiled/mod.rs`: interrupt handling requires both a configured
  checkpointer and a thread id before persisting the interrupt checkpoint and
  returning `ExecutionStatus::Interrupted`.
- `crates/tinyagents-graph/src/compiled/test.rs`: unit coverage proves interrupts without a
  checkpointer and interrupts without a thread id return errors rather than
  unresumable paused executions.
- `tests/graph_durable.rs`: integration coverage proves the public run surface
  rejects an interrupt emitted without a thread id.
- `docs/modules/graph/interrupts.md`: documents that interrupted results are
  only returned after the checkpoint needed for resume is persisted.

### Resolved: event sink listeners are notified outside the sink lock

The harness event sink no longer holds its mutex while invoking listener
callbacks:

- `crates/tinyagents-harness/src/events/mod.rs`: `EventSink::emit` now assigns the record id and
  clones the listener list under lock, then releases the lock before notifying
  listeners in registration order.
- `crates/tinyagents-harness/src/events/types.rs`: public docs describe the lock-narrowing
  contract and synchronous callback order.
- `crates/tinyagents-harness/src/events/test.rs`: unit coverage proves a listener can emit once
  back into the same sink without deadlocking.

### Resolved: response cache keys use a collision-resistant digest

The local response cache key now uses SHA-256 over canonical request JSON:

- `crates/tinyagents-harness/src/cache/mod.rs`: `cache_key` recursively canonicalizes request JSON
  and returns a SHA-256 hex digest. The short FNV helper remains only for local
  prompt-layout fingerprints, not response-cache identity.
- `crates/tinyagents-harness/src/cache/test.rs`: unit coverage asserts deterministic SHA-256 key
  shape and different keys for different model requests.
- `docs/modules/harness/cache.md`: documents that every behavior-affecting
  serialized request field participates in the SHA-256 digest.

## Stale Prior Finding Resolved

A prior audit found that `cargo build --all-targets` failed
because the OpenAI module was missing. That is no longer current:
`crates/tinyagents-harness/src/providers/openai/` exists and is compiled in by default, so build
and test paths cover it without any feature flag.

## Verification

Commands run from the repository root:

- `cargo fmt --check`: passed
- `cargo build --all-targets`: passed
- `cargo test`: passed
- `cargo clippy --all-targets -- -D warnings`: passed

Observed test coverage from the commands:

- Unit tests (including the OpenAI adapter): passed.
- Doctests: passed.
- Live OpenAI tests ran and passed in this environment when `OPENAI_API_KEY`
  was set (they skip otherwise).

## Prioritization

All findings from this audit have been resolved in the codebase.
