# Subagent lifecycle

`tinyagents_orchestration::subagent` coordinates a host-resolved subagent
lifecycle without making any product-policy decision. Its direct dependency
direction is:

```text
orchestration::subagent -> tinyagents-runtime -> {tinyagents-harness, tinyagents-session}
```

Hosts provide three object-safe seams. `SubagentPlanner<C, H>` transforms a
live `SubagentRequest<C, H>` into a complete `PreparedSubagent<C>`: resolved
agent identity, model messages, immutable `ToolSnapshot`, and an explicit,
owned `RunContext<C>`. `H` is opaque per-call host options and is forwarded
unchanged only to the planner. `SubagentExecutor<C>` executes precisely that plan. The
planner and executor own prompts, model choice, tool authorization, workspace
policy, artifact resolution, and host context data; this module owns none of
them. `SubagentPersistence` owns a host's durable resume and lifecycle records.

`SubagentTaskKey` is the durable lifecycle identity. It combines the original
root run, immediate parent run, optional thread, and host-local task id.
Fresh-task callers use `SubagentRequest::fresh_from_parent`, which derives the
key from an actual parent and validates the owned child context. Continuations
use `continue_with_key`: hosts must authorize and recover the original key,
then retain it even with a fresh owned execution context. Request fields are
private, so callers cannot bypass these identity checks. The key is the sole
durable thread identity; a request carries no duplicate thread field.

Hosts must assign unique durable `RunConfig` ids to parent and child runs. The
constructors reject an equal parent/child id but cannot prove global uniqueness.
Persistence,
in-flight coalescing, and terminal caching all use this scoped key: two parents
may reuse a task id without sharing state, while repeat calls for the same
durable scope deduplicate and resume correctly.

`SubagentDriver<C, H>::run` uses the validated request key before it loads or
prepares anything. It then loads a resume only when the caller did not provide one,
then prepares and executes. An awaiting-input result calls `save_pause`; all
other results call `record_terminal`. These operations are mutually exclusive.
The driver caches only successfully persisted terminal outcomes by scoped
`SubagentTaskKey`, so
repeated calls through the same driver return them without executing or
recording again. A pause is never cached: the next call loads its resume state
and executes the continuation. Concurrent calls for the same task coalesce;
different scoped task keys, including nested child tasks on the same driver,
execute independently. Persistence implementations must also enforce
idempotency by `SubagentTaskKey` across processes and driver instances.

Cancellation is cooperative. It is observed before planning, after resume
load and planning, and immediately after execution. A cancellation that wins
after the executor produced an outcome preserves that outcome's output,
history, usage, and neutral artifact references while reporting `Cancelled`.
The supplied cancellation token is installed on the prepared `RunContext`, so
the executor and the actual harness context observe one cancellation tree. If
host context `C` itself embeds a separate cancellation token, the executor is
responsible for synchronizing that host-owned token with this supplied token;
the neutral lifecycle cannot inspect opaque host data.
Persistence `Ok(())` is the commit boundary: if cancellation wins before it,
the driver abandons that uncommitted operation and records one `Cancelled`
terminal outcome; if persistence commits first, the committed result remains
truthful. Planner and executor task ids must exactly match the durable key task id
or the driver returns a typed error before persistence or caching.
Each coalesced caller retains its own cancellation token: cancelling a follower
returns a local cancelled outcome promptly, without cancelling the leader or
creating an additional persistence action.
An absent host seam is rejected at driver construction with a typed
`MissingCapability` error; no partial lifecycle runs.
