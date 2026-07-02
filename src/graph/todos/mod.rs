//! Per-thread **task board** (kanban todos): a list of task cards per thread.
//!
//! Where [`graph::goals`](crate::graph::goals) holds a single durable objective
//! per thread, a task board holds the concrete work items: an ordered list of
//! [`TaskBoardCard`]s with a small kanban lifecycle. This module owns the data
//! model and markdown rendering ([`types`]),
//! harness-[`Store`](crate::harness::store::Store)-backed CRUD with the
//! single-`InProgress` invariant ([`store`]), and the model-facing multiplexer
//! tool ([`tool`]).
//!
//! Ported from OpenHuman's task board / `todos` modules, minus the app-specific
//! coupling (progress events, RPC envelopes, in-memory scratch fallback): a
//! board is always `(Store, thread_id)`.

pub mod store;
mod tool;
mod types;

pub use tool::{TodoTool, register_todo_tools, todo_tools};
pub use types::{
    CardPatch, TaskApprovalMode, TaskBoard, TaskBoardCard, TaskCardStatus, TodosSnapshot,
    normalise_board, parse_status, render_markdown,
};

#[cfg(test)]
mod test;
