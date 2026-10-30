# Host Authorization and Tool Timeouts

Split out of [`README.md`](README.md) (see that module's overview for
architecture context) to keep the module `README.md` at or under the 500-line
limit. Covers the two host-facing pieces of the `Package Shape` runtime
surface: how a host capability bundle is bound to one invocation, and how
per-tool timeouts are resolved.

## Host-authorized invocations

`AgentHarness` is reusable process infrastructure: it owns durable model and
tool registries, middleware, policy, and caches. A host capability bundle is
instead supplied for each root through `runtime::AgentInvocation`:

```rust,no_run
use tinyagents_harness::{
    context::{RunConfig, RunContext},
    runtime::{AgentHarness, AgentInvocation, AgentTurnRequest},
};

# async fn example<State: Send + Sync + 'static>(
#     harness: &AgentHarness<State>,
#     host: tinyagents_harness::host::HostCapabilities<State>,
#     state: &State,
# ) -> tinyagents_harness::Result<()> {
let invocation = AgentInvocation::new(
    host,
    AgentTurnRequest::new("assistant", vec![]),
    RunContext::new(RunConfig::new("run-42"), ()),
);
let _run = harness.invoke_agent(invocation, state).await?;
# Ok(())
# }
```

This prevents concurrent roots from replacing one another's progress,
security, approval, or other host authority. The harness never stores a live
capability bundle (not even in a run-id map): it lives only in the
non-serializable `RunContext`, is never checkpointed, and is dropped with that
invocation. Recursive children inherit the exact parent bundle through their
live context and cannot select a bundle from their own harness. The lower-level
explicit-model `invoke*` APIs remain separate for SDK callers that intentionally
assemble a run without host capabilities.

Hosted invocations require `State: 'static` because their live capability
authority must be retained in the recursive context. The explicit-model
`invoke*`, streaming, and direct `SubAgent` paths do not install or inspect
that authority and continue to support borrowed state. Child execution is
planned by the host through `SubagentDriver`; the host supplies the child
context and enforces its own delegation policy.


## Hosted failures and timeout bounds

`invoke_agent` (and its streaming counterpart) fail with a `HostedError`, not a
raw `TinyAgentsError`. `kind` is a closed `HostedErrorKind` and `message` is a
fixed, sanitized string, so neither carries provider text. When
`kind == HostedErrorKind::Timeout`, `timeout_bound` says which wall-clock bound
fired, and it is `None` for every other kind:

| `timeout_bound` | Meaning | Host handling |
| --- | --- | --- |
| `Some(TimeoutBound::PerModelCall)` | One call exceeded the per-model-call ceiling (`RunLimits::max_model_call_ms`), including a wedged host model resolution. A `TimeoutMiddleware` expiry is reported as `Run`. The run still had time. | Retryable, but only safely when the run had no external effects. If tools already ran, continue from `HostedError::run`'s partial transcript (with the host's idempotency/effect reconciliation) instead of re-issuing the original turn, which could repeat a non-idempotent tool. |
| `Some(TimeoutBound::Run)` | The run's own timeout or deadline was exhausted. | Terminal for this run; do not retry it as-is. |

```rust,no_run
use tinyagents_harness::runtime::{HostedError, HostedErrorKind, TimeoutBound};

fn should_retry(error: &HostedError) -> bool {
    error.kind == HostedErrorKind::Timeout
        && error.timeout_bound == Some(TimeoutBound::PerModelCall)
}
```

`HostedErrorKind::ToolRefused` means host tool policy refused a call and asked
for the run to stop (`TinyAgentsError::ToolRefused`). Its message is
`hosted agent invocation stopped: host policy refused a tool call`, and the
hosted stream's `Failed` item and `RunFailed` event carry the same message when
the run's `TerminalOutcome::reason` is `ToolRefused`. It is not retryable as-is;
the host's policy decides whether the turn can run again.

`TimeoutBound` is `#[non_exhaustive]`; treat an unknown bound like `Run`.
`HostedError` gained the public `timeout_bound` field, so code that builds it
with a struct literal must set it. Recursive hosted children convert back to
`TinyAgentsError` with the bound preserved (`PerModelCall` becomes
`CallTimeout`, anything else `Timeout`).

## Tool timeout policy

Hosts enable per-tool deadlines with
`AgentHarness::with_tool_timeout_settings(ToolTimeoutSettings)`. The setting is
shared and dynamically updateable. Each tool supplies `ToolTimeout::Inherit`
(the default), `Millis(budget)`, or `Unbounded`; resolution happens at the
innermost tool call after wrap middleware has had a chance to rewrite its
arguments. On expiry the loop appends a recoverable tool-error result and keeps
running, allowing model repair. The independent run wall-clock limit remains a
hard error. See [`tool.md`](tool.md) for the tool contract and
[`runtime.md`](runtime.md) for harness assembly.
