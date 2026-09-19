# Graph Error Handling And Fault Tolerance

The graph runtime survives transient network failures and makes a hard failure
restartable, using two opt-in mechanisms plus the existing checkpoint/resume
machinery.

## Node retry (surviving network blips)

`CompiledGraph::with_node_retry(RetryPolicy)` wraps every node handler. When a
handler fails with a [retryable] error — `TinyAgentsError::Model` or
`TinyAgentsError::Tool`, the transient class classified by
`harness::retry::is_retryable` — the executor re-runs the node **from its start**
up to the policy's `max_attempts`, emitting `GraphEvent::NodeRetryScheduled`
before each retry. Because a node is never suspended mid-flight, a retry rebuilds
the handler future and context from scratch, matching the durable-execution
model.

Backoff between attempts is computed from the `RetryPolicy` (exponential, with
optional jitter) but only **slept on** when the policy opts in via
`RetryPolicy::with_backoff_sleep(true)`. The default is sleep-free so unit tests
stay deterministic; production callers enable it so retries wait a real, growing
delay. The same opt-in switch governs the harness model-call retry loop and
`RetryMiddleware`.

Both the sequential and parallel runners drive handlers through one shared
`run_node_with_retry` helper, which also applies the per-node timeout — so a
retried attempt is still bounded by `node_timeout`.

## Resumable failures (restarting / continuing after a hard failure)

When a handler fails **beyond** the retry budget, or with a non-retryable error,
the run is not lost. On a checkpointed thread the executor:

1. folds the updates of the branches that completed *before* the failing one
   into committed state (partial parallel progress is preserved, not discarded);
2. persists a **failure-boundary checkpoint** whose `next_nodes` schedule the
   failed node (and the not-yet-run tail of the step) for re-run, records the
   successful nodes as `completed_tasks`, and stamps the rendered `error` and
   `failed_node` into the checkpoint metadata;
3. records a `Failed` run status carrying that checkpoint id;
4. returns the error to the caller.

Without a checkpointer/thread the run aborts immediately with the error, exactly
as before — the failure checkpoint is a no-op.

### Restarting and continuing

Because `resume`/`resume_from` replay from a checkpoint's `next_nodes`, a failed
run resumes exactly like an interrupted one:

- `CompiledGraph::retry(thread)` — shorthand for `resume` with an empty command;
  re-runs the failed node and the not-yet-run tail from the failure boundary.
- **Continue on user feedback** — inspect committed state with `get_state`, edit
  it with `update_state` (attributed through the reducers), then `retry`/`resume`.
  The re-run sees the edited state.

See the `resilient_graph` example (`cargo run --example resilient_graph`) for
both mechanisms end to end.

### Per-thread execution lease

`CompiledGraph::execute` holds a per-`(thread_id, namespace)` lock for a run's
whole lifetime: an in-process `ThreadLockMap` guard first (always active), and,
when a checkpointer is configured, a durable lease claimed via
`Checkpointer::try_claim`/`renew`/`release` (owner = run id, default TTL 5
minutes). A second concurrent `run_with_thread`/`resume`/`retry` call for the
same thread either waits on the in-process lock (same process) or is refused
with `TinyAgentsError::Validation` if a live lease is held by another owner
(cross-process). A lease whose owner crashed without releasing it is
reclaimable once its TTL elapses. `SqliteCheckpointer` and `FileCheckpointer`
both implement the lease; the trait's default is a no-op that always succeeds,
so out-of-tree backends keep compiling unprotected.

## Panic safety

Each node handler future is polled through
`futures::FutureExt::catch_unwind(AssertUnwindSafe(..))`. A panic inside a
handler no longer unwinds through the executor: it is converted into
`TinyAgentsError::Graph("node `{id}` panicked: {msg}")` and flows through the
same resumable failure boundary as any other node error (checkpoint,
`RunFailed` event, `Failed` status). `retry(thread)` re-runs the panicking
node exactly as it would a node that returned `Err`.

## Cooperative cancellation

`RunOptions { cancellation: Option<tinyagents_harness::CancellationToken> }`
carries an optional cancellation token into `CompiledGraph::run_with_options`,
`run_with_thread_options`, and `resume_with_options`. The token is checked
before every superstep and raced against that step's in-flight handler
futures. On cancellation the executor persists a checkpoint whose
`next_nodes` is the still-pending active set, records a `Cancelled` run
status (and emits `GraphEvent::RunCancelled`), and returns `Ok` rather than
an error — a cancelled run is resumable exactly like an interrupted one.

A `Drop` guard armed for the run's lifetime protects against the run future
itself being dropped mid-flight (a host timeout racing `tokio::select!`,
`JoinHandle::abort`, and similar). It disarms on every normal terminal exit
(success, failure, interrupt, explicit cancel); if it is still armed when
dropped, it best-effort persists a `Cancelled` status from a detached task so
the run is never left stuck at `Running`. This is a status guarantee, not a
full flush guarantee: an in-flight `AsyncCheckpointWrites` write already
spawned before the drop still completes in the background (tokio does not
abort a `JoinHandle`'s underlying task on drop), but the guard's own tracking
of that write's outcome is lost.

## Error taxonomy

`TinyAgentsError` distinguishes structural/config errors (non-resumable) from
node failures (resumable on a checkpointed thread):

- structural / not retried, not resumable: missing start, missing node, missing
  edge target, missing route, invalid command/parent/send target, recursion
  limit, node-visit limit, sub-agent depth, checkpoint required/missing, resume
  mismatch, reducer conflict, invalid concurrent update, serialization failure,
  checkpoint backend failure;
- transient / retryable and resumable: `Model`, `Tool` (and `Timeout` aborts the
  node but leaves a resumable boundary like any other node failure).

## Future work

- Route node errors to a node-specific or default error-handler node.
- Cooperative drain/shutdown with a drain reason.
- Populate the checkpoint `pending_writes` list explicitly (today partial
  progress is folded into committed state instead).
- Renew the durable execution lease mid-run for long-running steps that could
  outlive its TTL (today it is claimed once, at `execute` entry, and released
  at exit — no heartbeat loop).
- Force-drain (not just best-effort track) in-flight async checkpoint writes
  from the run-future drop guard.

[retryable]: ../../../crates/tinyagents-harness/src/retry/mod.rs
