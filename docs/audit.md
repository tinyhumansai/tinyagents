# Codebase Audit

Date: 2026-06-29

This audit reviewed the current RustAgents codebase for correctness gaps,
security issues, implementation weaknesses, and spec-contract mismatches. No
code changes were made as part of the audit.

## Summary

The crate is in a healthy build state: default and `openai` feature builds pass,
the full test suite passes, and clippy is clean. The main issues are not broad
compile failures; they are fail-closed contract gaps around tool execution,
provider parsing, model capability selection, and durable graph recovery.

## Findings

### High: tool arguments are not schema-validated before execution

The agent loop advertises tool schemas to the model, but when a model returns a
tool call it looks up the tool and dispatches the raw `serde_json::Value`
arguments directly:

- `src/harness/agent_loop/mod.rs`: tool lookup and dispatch happen in the tool
  loop without a central argument/schema validation step.
- `src/harness/tool/types.rs`: `Tool::call` documents a validated call, but the
  harness does not enforce `ToolSchema.parameters` before calling the tool.

This is the most important security boundary in the crate. Side-effecting tools
should reject malformed, missing, extra, or type-mismatched arguments before
tool code runs.

Recommended fix:

- Add a local JSON Schema validation step before `tool.call(...)`.
- Treat schema validation failures as fail-closed harness errors.
- Add regression tests for missing required fields, wrong types, extra fields
  when disallowed, and nested object validation.

### High: malformed OpenAI tool-call JSON is silently converted to `null`

The OpenAI provider parses function-call arguments from provider string payloads
with a fallback to `Value::Null`:

- `src/harness/providers/openai/mod.rs`: unary response parsing falls back to
  `Value::Null` on invalid JSON.
- `src/harness/providers/openai/mod.rs`: streaming response reconstruction does
  the same for assembled argument fragments.

This can hide malformed provider/model output and let the rest of the harness
observe a valid-looking tool call with `null` arguments. Combined with missing
schema validation, malformed arguments can reach tool implementations.

Recommended fix:

- Return a provider/serialization error when function arguments are invalid
  JSON.
- Preserve the raw malformed argument string in the error message or raw
  provider payload for debugging.
- Add unary and streaming tests for invalid function argument JSON.

### Medium: `ModelRequest::required_capabilities` is not enforced

The request model has a `required_capabilities` field, and `ModelProfile` has a
`satisfies` helper, but model resolution does not use that requirement before
selecting a model:

- `src/harness/model/types.rs`: `ModelRequest::required_capabilities` is
  documented as pre-call validation/filtering.
- `src/harness/model/mod.rs`: `ModelRegistry::resolve_request` ignores the
  field.
- `src/harness/agent_loop/mod.rs`: the loop resolves and invokes the selected
  model without checking capabilities.

Callers can therefore request streaming, tool calling, structured output,
modalities, or token capacity and still be routed to a model profile that lacks
those capabilities.

Recommended fix:

- Filter or reject model candidates whose profiles do not satisfy
  `required_capabilities`.
- Decide how to handle models with no profile: either fail conservatively when
  requirements are present or require callers to opt into unknown capability
  profiles.
- Add tests for request override, previous-model reuse, hints, agent default,
  registry default, and fallback selection under capability requirements.

### Medium: runtime `Command::goto` targets can poison durable graph state

Graph compile-time validation checks static and conditional topology, but
runtime command routing accepts targets returned by node code without validating
them before they become the next active set:

- `src/graph/compiled/mod.rs`: command results store `command.goto` directly in
  the step routing map.
- `src/graph/compiled/mod.rs`: `route()` returns command targets without
  checking each target is `END` or a known node.

With checkpointing enabled, an invalid target can be persisted as the next
active node and only fail on the next superstep as `MissingNode`.

Recommended fix:

- Validate each command target before appending it to `next`.
- Reject `START` and unknown nodes immediately with a deterministic validation
  or graph error.
- Add tests for invalid command target, `START` as a command target, and invalid
  target persistence under checkpointing.

### Medium: subgraph checkpoint namespaces are extended, but child runs are not threaded

Subgraph wrappers namespace the child graph, then invoke `child.run(...)`:

- `src/graph/subgraph/mod.rs`: shared-state and adapter subgraph nodes call
  `child.run(...)`.
- `src/graph/compiled/mod.rs`: `run()` explicitly persists no checkpoints
  without a thread id.

This means namespace isolation exists, but child checkpoint persistence is not
actually wired through the parent thread. Nested graph durability is weaker than
the namespace support implies.

Recommended fix:

- Add a child graph execution path that propagates the parent `thread_id`.
- Use a deterministic child namespace and parent checkpoint id linkage.
- Add tests proving child checkpoints persist and can be inspected/resumed under
  the parent thread.

### Medium: interrupts can return an unresumable interrupted status

When a node emits an interrupt, `persist_checkpoint` can return `Ok(None)` if no
checkpointer or thread id is configured, but the graph still returns an
interrupted execution status:

- `src/graph/compiled/mod.rs`: interrupt handling returns
  `ExecutionStatus::Interrupted`.
- `src/graph/compiled/mod.rs`: `persist_checkpoint` is a no-op without both a
  checkpointer and thread id.

That creates an interrupted run that cannot be resumed, even though the runtime
surface presents it as an interrupted execution.

Recommended fix:

- Require checkpointing and a thread id for interrupt-capable runs, or
  explicitly mark the result as non-resumable.
- Consider a separate error for "interrupt emitted without durability".
- Add tests for interrupt without checkpointer, interrupt with checkpointer but
  no thread, and interrupt with full durable configuration.

### Low: event listeners can deadlock the sink

`EventSink::emit` holds the sink mutex while invoking every listener:

- `src/harness/events/mod.rs`: listener callbacks run while the lock is held.

The docs warn listeners not to call back into the same sink, but the
implementation can avoid this class of deadlock.

Recommended fix:

- Clone the listener list under lock, release the lock, then notify listeners.
- Add a test with a listener that emits to another sink and, if supported, a
  guard test for same-sink callback behavior.

### Low: response cache keys use 64-bit FNV-1a

The response cache key is deterministic, but not collision-resistant:

- `src/harness/cache/mod.rs`: `cache_key` uses a 64-bit FNV-1a hash over the
  canonical request JSON.

This is acceptable for local tests and short-lived in-process caches, but risky
for any shared, durable, multi-tenant, or untrusted-input cache.

Recommended fix:

- Use a stronger hash for durable/shared caches, or store and compare the
  canonical request bytes alongside the hash.
- Document the current key as non-cryptographic and local-cache oriented.

## Stale Prior Finding Resolved

A prior audit found that `cargo build --all-targets --features openai` failed
because the feature-gated OpenAI module was missing. That is no longer current:
`src/harness/providers/openai/` exists and both build and test paths pass with
the `openai` feature enabled.

## Verification

Commands run from the repository root:

- `cargo fmt --check`: passed
- `cargo build --all-targets`: passed
- `cargo build --all-targets --features openai`: passed
- `cargo test`: passed
- `cargo clippy --all-targets -- -D warnings`: passed
- `cargo clippy --all-targets --features openai -- -D warnings`: passed
- `cargo test --features openai`: passed

Observed test coverage from the commands:

- Default unit tests: 421 passed.
- Default doctests: 40 passed, 1 ignored.
- `openai` feature unit tests: 432 passed.
- `openai` feature doctests: 42 passed, 1 ignored.
- Live OpenAI tests ran and passed in this environment when the `openai` feature
  was enabled.

## Prioritization

Fix order should be:

1. Central tool argument validation.
2. OpenAI invalid tool-argument JSON handling.
3. Model capability enforcement.
4. Runtime `Command::goto` target validation.
5. Durable subgraph checkpoint/thread propagation.
6. Interrupt durability guardrails.
7. Event sink lock narrowing.
8. Stronger cache key story for shared/durable caches.
