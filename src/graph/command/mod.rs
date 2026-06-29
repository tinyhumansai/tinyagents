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
//! See [`types`] for the definitions.

mod types;

pub use types::{Command, Interrupt, NodeResult, RouteTarget, Send};

use std::sync::atomic::{AtomicU64, Ordering};

use crate::harness::ids::NodeId;

static INTERRUPT_SEQ: AtomicU64 = AtomicU64::new(0);

impl<Update> Command<Update> {
    /// Creates an empty command (no update, no routing, no resume).
    pub fn new() -> Self {
        Self {
            update: None,
            goto: Vec::new(),
            resume: None,
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
        }
    }

    /// Creates a command carrying a partial state update.
    pub fn update(update: Update) -> Self {
        Self {
            update: Some(update),
            goto: Vec::new(),
            resume: None,
        }
    }

    /// Creates a resume command carrying a value for an interrupted node.
    pub fn resume(value: serde_json::Value) -> Self {
        Self {
            update: None,
            goto: Vec::new(),
            resume: Some(value),
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
    pub fn new(node: impl Into<NodeId>, payload: serde_json::Value) -> Self {
        let node = node.into();
        let seq = INTERRUPT_SEQ.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("interrupt-{node}-{seq}"),
            node,
            payload,
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
        }
    }
}

#[cfg(test)]
mod test;
