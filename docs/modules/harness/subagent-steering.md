# Sub-Agent And Orchestrator Steering

Steering is runtime control sent to an already-created agent run. It is broader
than a prompt edit and narrower than mutating arbitrary state. TinyAgents should
support steering at both the harness level and graph level so parent
orchestrators, humans, tests, and UIs can guide long-running work without
breaking run hierarchy, checkpoints, or observability.

## Steering Relationships

Supported steering relationships:

- parent orchestrator agent steers a sub-agent
- human steers a sub-agent
- human steers a parent orchestrator agent
- graph supervisor node steers child sub-agent tasks
- middleware steers an agent loop under explicit policy
- test harness steers fake agents deterministically

Every steering operation must preserve `root_run_id`, `parent_run_id`,
`thread_id`, `agent_id`, `task_id` when graph-backed, source actor, source
scope, and checkpoint or event offsets when available.

## Non-Goals

- Steering is not hidden prompt injection.
- Steering is not direct mutation of another run's private memory.
- Steering is not a way to bypass model, tool, graph, store, or budget
  allowlists.
- Steering is not an untracked side channel; every accepted steering instruction
  emits events and appears in run inspection.

## Core Types

```rust
pub struct SteeringCommand {
    pub target: SteeringTarget,
    pub actor: SteeringActor,
    pub kind: SteeringKind,
    pub payload: serde_json::Value,
    pub priority: SteeringPriority,
    pub delivery: SteeringDelivery,
    pub policy: SteeringPolicyRef,
    pub correlation_id: CorrelationId,
}

pub enum SteeringTarget {
    AgentRun(RunId),
    SubAgentRun { parent_run_id: RunId, child_run_id: RunId },
    GraphTask { run_id: RunId, task_id: TaskId },
    GraphNode { run_id: RunId, node_id: NodeId },
    Thread(ThreadId),
}

pub enum SteeringActor {
    Human { user_id: String },
    ParentAgent { run_id: RunId, agent_id: ComponentId },
    GraphSupervisor { run_id: RunId, node_id: NodeId },
    Middleware { name: String },
    Test,
}

pub enum SteeringKind {
    AddInstruction,
    ReplaceInstruction,
    AskClarifyingQuestion,
    ProvideContext,
    ConstrainTools,
    ConstrainModel,
    Pause,
    Resume,
    Cancel,
    Approve,
    Reject,
    Redirect,
    RequestStatus,
    UpdateBudget,
}
```

`SteeringCommand` is the durable input shape. The harness may expose ergonomic
helpers such as `steer_subagent`, `steer_agent`, `pause_agent`,
`resume_agent`, and `cancel_agent`, but those helpers should lower into typed
commands.

## Harness-Level Steering

Harness-level steering applies to direct model calls and model-tool loops that
are not graph-backed.

Required behavior:

- queue steering commands on the target run
- deliver commands only at safe boundaries: before model call, after model
  response, before tool dispatch, after tool result, before loop continuation,
  or while paused
- expose pending steering in `RunContext`
- allow middleware to accept, reject, transform, or defer steering
- convert accepted steering into model-visible messages, tool allowlist changes,
  runtime policy changes, or harness control outcomes
- record which steering commands affected which model request or tool call

The default delivery rule should be conservative: steering becomes visible on
the next agent-loop boundary, not in the middle of a provider stream or
side-effecting tool call.

## Graph-Level Steering

Graph-backed steering is a command against graph state and task scheduling. It
should use the graph runtime whenever the target run has checkpoints, active
tasks, interrupts, child sub-agents, or subgraphs.

Graph-level steering can lower to:

- `Command::resume` for an interrupt
- `Command::goto` for an explicit redirect
- `Command::update` for a reducer-mediated state update
- `Command::parent` for child-to-parent handoff
- task cancellation through graph policy
- a queued steering event delivered to a child sub-agent node

Graph steering must be checkpoint-aware. If a child task is paused, a steering
command targets the child namespace and resumes only that child unless policy
explicitly escalates to the parent run.

## Parent Orchestrator Steering

A parent orchestrator may steer sub-agents it created when policy grants that
relationship.

Examples:

```text
orchestrator -> spawn research_subagent
orchestrator -> steer research_subagent: "focus only on billing evidence"
orchestrator -> steer critic_subagent: "compare answer against policy version 3"
orchestrator -> wait research_subagent, critic_subagent
orchestrator -> merge child outputs through graph reducers
```

Rules:

- the parent can steer only descendants in its run tree by default
- sibling agents cannot steer each other unless a graph supervisor grants it
- steering cannot expand a child's tool/model allowlist beyond the child
  policy
- parent steering is visible to the child as a structured instruction with
  provenance, not as anonymous user text
- the child may reject steering that violates its policy and must emit a
  rejection event
- parent steering and child output both merge through normal graph or harness
  state contracts

## Human Steering

Humans can steer orchestrators or sub-agents through explicit control surfaces.

Human steering modes:

- interrupt response: answer a pending approval/review question
- instruction injection: add a human-authored instruction at the next safe loop
  boundary
- policy edit: narrow model/tool/budget limits for a run
- pause/cancel: stop a run or child task cooperatively
- redirect: send a graph-backed run to a review, retry, or finalize node
- state update: apply a reducer-mediated correction to graph state

Human steering must be auditable:

- actor id
- target run/task/node
- timestamp
- accepted/rejected/deferred status
- policy that allowed the action
- model/tool/graph operation affected
- redacted payload for UI and event export

Human steering of sub-agents should not require steering the parent first. A UI
must be able to target a specific child run or graph task using run-tree
metadata. The parent is notified through events and rollups, not by opaque
shared state.

## Delivery And Conflict Rules

Steering commands are ordered by:

1. target run tree position
2. explicit priority
3. creation time
4. correlation id for deterministic tie-breaking

Conflict rules:

- `Cancel` wins over new instructions.
- `Pause` defers new model/tool work until resumed.
- human `ConstrainTools` can narrow but not widen a parent-granted tool set.
- parent `AddInstruction` appends to the child steering context unless a policy
  allows replacement.
- simultaneous redirects require a graph policy: first accepted, highest
  priority, or explicit conflict error.
- stale steering commands that refer to a completed checkpoint are rejected
  unless they target a new fork.

## Events

Required events:

- `steering.requested`
- `steering.accepted`
- `steering.rejected`
- `steering.deferred`
- `steering.delivered`
- `steering.applied`
- `steering.superseded`
- `steering.failed`

Events must include root run id, parent run id, target run id, target task/node
when graph-backed, actor, kind, correlation id, policy id, checkpoint id when
available, and redacted payload summary.

## Testkit

The testkit should include:

- fake parent orchestrator steering a fake sub-agent
- human steering a paused sub-agent interrupt
- human steering a parent orchestrator
- rejected steering because of missing authority
- rejected tool allowlist expansion
- deterministic ordering of simultaneous steering commands
- graph-backed targeted child resume
- harness-loop steering delivered at the next safe boundary
