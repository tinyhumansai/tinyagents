# Graph Parallel Agents And Context Forking

Parallel agents are a graph pattern where one supervisor node starts multiple
child agents in the same superstep, lets them work from the same committed
state, and merges their outputs through normal graph reducers.

Context forking is the runtime mechanism that makes this safe. A fork creates a
child task context from the current graph context. The child receives shared
read-only runtime dependencies, including cache and stores, while getting its
own task id, node id, event namespace, cancellation scope, and mutable scratch
space.

This pattern is intended for workflows such as:

- run two specialist sub-agents against the same user request
- compare two strategies and merge the better result
- dispatch research and verification agents in parallel
- fan out over multiple tools or retrievers while reusing cached model/tool
  calls
- run a critique agent and a builder agent from the same initial context

## Target Shape

```rust
pub struct ContextForkOptions {
    pub node: NodeId,
    pub task_name: Option<String>,
    pub namespace_label: Option<String>,
    pub input: serde_json::Value,
    pub cache: CacheForkPolicy,
    pub stores: StoreForkPolicy,
    pub timeout: Option<TimeoutPolicy>,
    pub metadata: serde_json::Value,
}

pub enum CacheForkPolicy {
    InheritReadWrite,
    InheritReadOnly,
    Isolated,
}

pub enum StoreForkPolicy {
    Inherit,
    IsolatedNamespace(StoreNamespace),
}
```

Default behavior should be `InheritReadWrite` for cache and `Inherit` for
stores. That lets parallel child agents reuse expensive cached model/tool
results while still writing their own child-run events and outputs.

## Parallel Sub-Agent Fanout

A supervisor node can spawn two sub-agents by returning `Send` packets or a
command with multiple destinations:

```rust
Command::new()
    .goto([
        Send::new("research_agent", json!({ "question": state.question })),
        Send::new("critic_agent", json!({ "question": state.question })),
    ])
```

Each child agent runs as a separate graph task:

- unique child `task_id`
- unique child `run_id`
- shared `root_run_id`
- parent run id set to the supervisor task run
- namespace such as `supervisor:<task_id>/research_agent:<task_id>`
- inherited `thread_id`
- inherited checkpointer according to subgraph/sub-agent policy
- inherited cache handle according to `CacheForkPolicy`
- inherited stores according to `StoreForkPolicy`

The children must not mutate the parent state directly. They return writes that
are merged at the superstep boundary by channels/reducers.

## Forked Context Contract

Forking a context copies identity and dependency handles deliberately:

Shared across forks:

- root run id
- thread id
- immutable user context
- store registry unless isolated by policy
- cache handle unless isolated by policy
- event sink
- stream sink
- cancellation parent
- budget counters when policy says children share budget

Unique per fork:

- task id
- node id
- child run id
- checkpoint namespace suffix
- event namespace
- recursion frame
- mutable scratchpad
- task-local resume values
- task-local pending writes

The fork must be cheap. It should clone `Arc` handles for shared services rather
than copying stores, cache contents, or large state snapshots.

## Shared Cache Semantics

Parallel agents should be able to reuse cache entries produced before the fork
and entries produced by sibling forks when durability policy allows it.

Rules:

- cache keys must include the called component, normalized input, relevant
  config, provider/model id, tool version, and cache namespace
- cache keys must not include child task ids unless the cached result is
  intentionally task-local
- child cache writes are visible to sibling forks only after the cache backend
  commits them
- cache hits are emitted as task events with `TaskCached`
- cached task writes are replayed through graph writes, not injected as final
  parent state
- cache failures do not fail the child task unless cache is marked required

This allows two sub-agents to share expensive retrieval, embedding, prompt,
model, or tool results without sharing mutable state.

## Merge Semantics

Parallel child outputs merge through state channels:

- independent fields use `LastValue` only when a single writer is guaranteed
- lists use append or topic reducers
- ranked candidates use custom merge reducers
- chat messages use message merge by id
- conflicting writes fail with an invalid concurrent update unless a reducer
  defines deterministic resolution

Example:

```rust
pub struct AgentFanoutUpdate {
    pub candidates: Vec<CandidateAnswer>,
    pub critiques: Vec<Critique>,
    pub usage: UsageDelta,
}
```

The parent graph should have reducers that append `candidates`, append
`critiques`, and aggregate `usage`.

## Checkpointing

Forked child tasks participate in normal checkpointing:

- task start appears in checkpoint task metadata
- completed child writes can be persisted as pending writes
- failed sibling tasks do not force successful child agents to rerun once
  pending writes are saved
- child checkpoints include namespace and parent checkpoint config
- resuming from interrupt restarts the interrupted child task, not unrelated
  completed siblings

If a forked sub-agent interrupts, the parent run should surface the interrupt
with enough namespace information to resume the correct child.

## Observability

Parallel agents must be visible as distinct child runs, not hidden futures.

Required events:

- `ContextForked`
- `SubAgentStarted`
- `TaskStarted`
- `TaskCached`
- `TaskCompleted`
- `TaskFailed`
- `SubAgentCompleted`
- `ContextForkJoined`
- `StateUpdated`

Every event must include:

- root run id
- parent run id
- child run id
- task id
- node id
- namespace
- recursion depth
- checkpoint id when available

UIs should be able to render the supervisor task with two child sub-agent lanes,
their streamed messages, cache hits, writes, and final reducer merge.

## Failure And Cancellation

Default policy:

- required child failure fails the superstep
- optional child failure records an error update and lets siblings complete
- parent cancellation cancels all children
- child timeout fails only that child unless policy escalates
- successful sibling writes can be preserved as pending writes

Future policies:

- race: first successful child wins and cancels siblings
- quorum: continue when N of M children succeed
- compare: require all children, then run a judge node
- fallback: run secondary agent only if primary fails

These should be explicit policies, not implicit behavior hidden in node code.

## Reusable map/reduce helper {#map-reduce}

`graph::parallel::map_reduce` is a standalone helper — independent of the graph
executor — for the common "run N items concurrently and reduce the results"
pattern. Where `Send` fanout is the low-level superstep primitive, `map_reduce`
gives a reusable async function with deterministic input-order results, a
concurrency cap, per-item success/failure isolation, and a configurable failure
policy:

```rust
pub async fn map_reduce<I, T, F, Fut>(
    items: Vec<I>,
    options: ParallelOptions,
    f: F,
) -> Result<ParallelOutcome<T>>
where
    F: Fn(usize, I) -> Fut,
    Fut: Future<Output = Result<T>>;
```

`f` receives each item's 0-based input index and the item. Results are collected
into a `ParallelOutcome<T>` of per-item `ItemOutcome<T>` values **in input order**
regardless of completion order (concurrency is driven by an ordered buffered
stream). `ParallelOutcome` exposes `success_count`, `failure_count`, `successes`,
and `into_successes`.

### Failure policy

`FailurePolicy` decides what happens when items fail:

- `FailFast` — stop at the first failing item (in input order) and return its
  error; remaining in-flight work is dropped.
- `CollectAll` (default) — run every item to completion, always return `Ok`,
  recording per-item success/failure for the caller to inspect.
- `Quorum(n)` — run every item; return `TinyAgentsError::Graph` unless at least
  `n` items succeeded.
- `BestEffort` — always `Ok`, silently keeping only the successful outputs.

### Execution controls

`ParallelOptions` is a builder. It is `Clone` (no longer `Copy`) because it can
hold a cancellation token.

| Builder | Effect |
| --- | --- |
| `with_max_concurrency(n)` | Cap simultaneous items (`0` = unbounded). |
| `with_failure_policy(p)` | Set the `FailurePolicy`. |
| `with_item_timeout(dur)` | Per-item wall-clock bound; a slow item becomes a recoverable per-item failure (then handled by the policy) rather than stalling the batch. |
| `with_total_timeout(dur)` | Overall wall-clock bound; on elapse the call returns `TinyAgentsError::Timeout` and drops remaining in-flight work. |
| `with_cancellation(token)` | Attach a cooperative `CancellationToken`; when cancelled the call returns `TinyAgentsError::Cancelled` and drops in-flight work. |

Per-item timeout — a slow item fails in isolation while fast items still
succeed:

```rust
use tinyagents::graph::parallel::{map_reduce, FailurePolicy, ParallelOptions};
use tinyagents::TinyAgentsError;

// Item 1 hangs; a 50ms per-item timeout turns it into a recoverable failure.
let out = map_reduce(
    vec![0u64, 3_600_000, 0],
    ParallelOptions::default()
        .with_failure_policy(FailurePolicy::CollectAll)
        .with_item_timeout(std::time::Duration::from_millis(50)),
    |_i, ms| async move {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
        Ok::<_, TinyAgentsError>(ms)
    },
)
.await?;
assert_eq!(out.success_count(), 2);
assert_eq!(out.failure_count(), 1); // the hanging item timed out
```

Total timeout — the whole batch aborts and drops in-flight work:

```rust
let err = map_reduce(
    vec![3_600_000u64],
    ParallelOptions::default().with_total_timeout(std::time::Duration::from_millis(20)),
    |_i, ms| async move {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
        Ok::<_, TinyAgentsError>(ms)
    },
)
.await
.expect_err("total timeout aborts the batch");
assert!(matches!(err, TinyAgentsError::Timeout(_)));
```

Cancellation — a cancelled token stops the batch:

```rust
use tinyagents::CancellationToken;

let token = CancellationToken::new();
token.cancel();
let err = map_reduce(
    vec![0u64],
    ParallelOptions::default().with_cancellation(token),
    |_i, _n| async move {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        Ok::<_, TinyAgentsError>(0u64)
    },
)
.await
.expect_err("a cancelled token aborts");
assert!(matches!(err, TinyAgentsError::Cancelled));
```
