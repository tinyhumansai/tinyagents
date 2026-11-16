# Command hooks

The `command_hooks` module loads Cursor-compatible `hooks.json` files and
dispatches their commands or prompt evaluations at agent lifecycle events.
The host provides a [`HookEnvironment`](super::environment::HookEnvironment)
with its shell (or async command spawner), prompt evaluator, home directory, and product name.

## Running command hooks under a host gate

By default the engine spawns a command hook itself through the synchronous
`HookEnvironment::shell` factory. A host that must run hooks behind an async
gate (a sandbox, an approval, a remote box) calls
`HookEnvironment::with_command_spawner` with a `CommandSpawner`. The engine then
hands it a `CommandRequest` (command line, working directory, extra
environment, stdin bytes with the event JSON, and the timeout) and reads a
`CommandOutput` (exit code, stdout, stderr) back; exit-code handling is the same
as for the local path. The spawner owns enforcing `CommandRequest::timeout` and
killing whatever it started: the engine waits `SPAWNER_GRACE` past the deadline
so that teardown can finish, then drops the future. A spawner that refuses
returns `Err`, which is a hook failure like any other (fail-open, or a denial
when the definition sets `fail_closed`).

## Lifecycle entry points

Events with no tool-call seam are fired through `HookEngine` methods in
`ops.rs`: `session_started`, `session_ended`, `prompt_submitted` (returning a
`PromptVerdict`), `pre_compact`, `subagent_starting`, `subagent_stopped`, and
`agent_thought`. Each returns immediately when no hook is configured for its
event. `agent_thought` fires the observational `afterAgentThought` event with a
`TextPayload` (the reasoning text and its optional `duration_ms`); its hooks
run in the background, matchers select on the thought text, and their output
is ignored.

## Configuration and trust

Configuration is read from system, user, workspace, and project layers. Lists
are concatenated in that order, so a more specific layer cannot remove a
broader policy. Gating decisions merge with denial taking precedence over
approval. Each hook receives the arguments produced by preceding hooks, so
policy hooks evaluate rewrites before later hooks can use them.

`loop_limit` omitted uses the engine default of five; `null` means unlimited;
an integer sets a per-hook cap. Follow-up counts are scoped to a session and
hook definition. A sessionless invocation receives at most one follow-up.

## Execution and operational constraints

Gating events run sequentially and wait for completion. Observing events run in
the background. Command and prompt hooks use the configured timeout, falling
back to the engine default. Failures are fail-open by default; set
`fail_closed` to turn a failure, including a timeout, into a denial. Command
hooks receive the event envelope on stdin and return a decision on stdout.

Only `sessionStart` output may contribute session environment variables. These
variables stay scoped to the session and are cleared when the host calls
`HookEngine::forget_session`.

See the module root documentation for the module map and the public types for
event, payload, and decision schemas.
