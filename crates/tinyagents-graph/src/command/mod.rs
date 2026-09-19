//! Command, interrupt, and node-result constructors.
//!
//! Commands are how a node steers the recursive runtime from the inside: a
//! [`Command`] couples a partial state update with explicit `goto` routing
//! (one or many targets — the fanout primitive), so a node — including one that
//! has just run a sub-agent or a subgraph — can dynamically decide which nodes
//! execute next instead of relying only on static edges. [`Interrupt`] pauses a
//! run for human-in-the-loop input, which the durable executor checkpoints and
//! later resumes.
//!
//! See `types` for the definitions.

mod types;

pub use types::{Command, Interrupt, NodeResult, RouteTarget, Send};

use tinyagents_harness::ids::{NodeId, TaskId};

impl<Update> Command<Update> {
    /// Creates an empty command (no update, no routing, no resume).
    pub fn new() -> Self {
        Self {
            update: None,
            goto: Vec::new(),
            resume: None,
            resume_by_task: std::collections::HashMap::new(),
        }
    }

    /// Creates a command that routes to one or more explicit node targets.
    pub fn goto(targets: impl IntoIterator<Item = impl Into<NodeId>>) -> Self {
        Self {
            update: None,
            goto: targets
                .into_iter()
                .map(|t| RouteTarget::Node(t.into()))
                .collect(),
            resume: None,
            resume_by_task: std::collections::HashMap::new(),
        }
    }

    /// Creates a command that fans out to one or more [`Send`] packets, each
    /// delivering a custom per-invocation argument to its target node. This is
    /// the map-reduce / per-branch-custom-input primitive; targets may repeat.
    pub fn send(sends: impl IntoIterator<Item = Send>) -> Self {
        Self {
            update: None,
            goto: sends.into_iter().map(RouteTarget::Send).collect(),
            resume: None,
            resume_by_task: std::collections::HashMap::new(),
        }
    }

    /// Creates a command carrying a partial state update.
    pub fn update(update: Update) -> Self {
        Self {
            update: Some(update),
            goto: Vec::new(),
            resume: None,
            resume_by_task: std::collections::HashMap::new(),
        }
    }

    /// Creates a resume command carrying a value for an interrupted node.
    pub fn resume(value: serde_json::Value) -> Self {
        Self {
            update: None,
            goto: Vec::new(),
            resume: Some(value),
            resume_by_task: std::collections::HashMap::new(),
        }
    }

    /// Creates a resume command carrying a distinct value per interrupted
    /// task (I1): a `Send` fan-out of the same node produces several
    /// concurrently-interrupted tasks, and this is how a caller delivers each
    /// its own resume value in one call, keyed by
    /// [`crate::builder::NodeContext::task_id`].
    pub fn resume_tasks(values: impl IntoIterator<Item = (TaskId, serde_json::Value)>) -> Self {
        Self {
            update: None,
            goto: Vec::new(),
            resume: None,
            resume_by_task: values.into_iter().collect(),
        }
    }

    /// Attaches a partial update to this command.
    pub fn with_update(mut self, update: Update) -> Self {
        self.update = Some(update);
        self
    }

    /// Appends explicit node routing targets to this command.
    pub fn with_goto(mut self, targets: impl IntoIterator<Item = impl Into<NodeId>>) -> Self {
        self.goto
            .extend(targets.into_iter().map(|t| RouteTarget::Node(t.into())));
        self
    }

    /// Appends [`Send`] fanout packets to this command's routing targets.
    pub fn with_sends(mut self, sends: impl IntoIterator<Item = Send>) -> Self {
        self.goto.extend(sends.into_iter().map(RouteTarget::Send));
        self
    }

    /// Attaches a resume value to this command.
    pub fn with_resume(mut self, value: serde_json::Value) -> Self {
        self.resume = Some(value);
        self
    }
}

impl<Update> Default for Command<Update> {
    fn default() -> Self {
        Self::new()
    }
}

impl Interrupt {
    /// Creates an interrupt with an auto-generated unique id.
    ///
    /// I7 (`docs/runtime-comparison/code-review-graph.md`): built from
    /// [`tinyagents_harness::ids::process_nonce`] +
    /// [`tinyagents_harness::ids::next_seq`] — the same restart-safe scheme
    /// [`tinyagents_harness::ids::new_checkpoint_id`] uses — rather than a
    /// bare process-local counter. A bare counter restarts at `0` in every
    /// new process, so two pauses minted in different process lifetimes
    /// could collide on `(node, seq)` and conflate two distinct interrupts
    /// in `GraphRunStatus::pending_interrupts` or a UI keyed on interrupt id.
    pub fn new(node: impl Into<NodeId>, payload: serde_json::Value) -> Self {
        let node = node.into();
        let id = format!(
            "interrupt-{node}-{}-{}",
            tinyagents_harness::ids::process_nonce(),
            tinyagents_harness::ids::next_seq()
        );
        Self {
            id,
            node,
            payload,
            task_id: None,
        }
    }

    /// Creates an interrupt with a caller-supplied id.
    pub fn with_id(
        id: impl Into<String>,
        node: impl Into<NodeId>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            node: node.into(),
            payload,
            task_id: None,
        }
    }

    /// Returns this interrupt with its scheduled task id set (R5/I1).
    ///
    /// The interrupt boundary calls this on the emitted interrupt before
    /// persisting/returning it, so a `Send` fan-out of the same node (or a
    /// re-emitted subgraph interrupt) is resumable by its own task rather
    /// than sharing the node's identity with its siblings.
    pub fn with_task_id(mut self, task_id: tinyagents_harness::ids::TaskId) -> Self {
        self.task_id = Some(task_id);
        self
    }
}

#[cfg(test)]
mod test;
