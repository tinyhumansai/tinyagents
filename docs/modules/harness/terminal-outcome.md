# Terminal outcomes and lifecycle events

`TerminalOutcome` is the typed explanation for how an agent run ended. It is
attached to completed and failed run events and is available on `AgentRun`.
The `reason` identifies the specific cause, while `class` provides a stable
coarse-grained category for hosts that need only success, timeout,
cancellation, failure, or suspension.

`TerminalReason::ToolRefused` (class `Failure`) marks a run that host tool
policy stopped (`TinyAgentsError::ToolRefused`, such as a hook deny with
`continue_: Some(false)`). A refusal answered to the model does not end the
run. The full error-to-reason mapping and merge precedence live in the
`terminal` module docs.

Outcomes preserve the legacy human-readable message. Timeout outcomes also
record whether the deadline landed before a provider call, during one, or
after a call. Provider-start state is tracked separately so a host can tell a
first-call failure from a preflight failure.

The direct loop publishes the outcome on `AgentRun` before running
`after_agent` middleware. This lets middleware observe the same classification
as the terminal event. Session hooks receive the typed outcome before the
legacy terminal notification.

Turn lifecycle events announce appended messages and retract only messages
that were previously announced. Initial input messages are treated as a seed
prefix and are not re-announced on the first turn.

Mutations that are not appends are explicit. `MessageRetracted { index }` is
emitted (highest index first) when an announced message is popped, for example
an unusable assistant reply dropped before a retry; it always precedes the
`MessageAppended` of its replacement. `TranscriptRewritten { len, reason }` is
emitted when the transcript is rewritten in place (a tool-set change folded
into the leading system message); a mirror should resynchronise and count later
`MessageAppended` indices from `len`. Nested tool calls (a tool calling another
tool) produce `ToolStarted` / `ToolCompleted` events with a `parent_call_id`
but no transcript rows, so they never emit `MessageAppended`.

On `RunCompleted` / `RunFailed` the `outcome` field is `Some` for every run this
crate ends; it is `None` only when deserializing journals written before the
field existed.

Scope: the typed outcome and the turn and message lifecycle events
(`TurnStarted`, `TurnCompleted`, `MessageAppended`) are published by the direct
loop. Input messages are never announced, each appended message is announced
exactly once, and a nested tool call never produces a `MessageAppended`.
`MessageRetracted` and `TranscriptRewritten` describe direct-loop recovery
paths.

## `provider_started` and summarizers

`provider_started` is true once any provider call was dispatched, including a
context-window summarizer's. Summarizer calls bypass the run context's dispatch
marker, so the compaction middleware scopes each summarization with a dispatch
tracker (`summarization::dispatch`): the built-in model-backed summarizers (`ModelSummarizer`, `TaskStateSummarizer`)
mark it right before they call their model (host-defined `Summarizer` impls
cannot, since `mark_dispatched` is crate-private; they still count when they
fail with usage), and the middleware then sets the run-wide flag. A
summarizer that fails with no usage after dispatching (a transport error) still
reports `provider_started: true`; one rejected before dispatch (empty input,
validation) does not. Only the run-wide flag is set, so a timeout during the main
model call is still classified by that call's own dispatch.
