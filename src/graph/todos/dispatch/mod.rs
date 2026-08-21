//! Dispatch policy for a task board: **what runs next, what it is told to do,
//! and how a run in flight is tracked and cancelled**.
//!
//! [`store`](crate::graph::todos::store) owns the board and
//! [`runs`](crate::graph::todos::runs) owns claims; this module is the layer
//! between them that a scheduler is built from:
//!
//! - [`select`] — pure policy. Pick the highest-urgency dispatchable card,
//!   decide whether it needs plan approval first, and pace a polling sweep so
//!   an idle board is not swept at full rate forever.
//! - [`prompt`] — render a card into the prompt its run works from, plus the
//!   addendum that keeps the card current while the run works.
//! - [`registry`] — track in-flight runs so a cancel can reach a detached task,
//!   with race-free removal so the card's terminal write-back happens once.
//!
//! Actually *executing* a card is deliberately not here: that needs an agent,
//! a model, and a host's own tool belt. The intended shape is a loop that asks
//! [`select::pick_next_card`] for work, claims it with
//! [`store::claim_card`](crate::graph::todos::store::claim_card), opens a run
//! with [`runs::create_run`](crate::graph::todos::runs::create_run), spawns the
//! work with [`prompt::build_task_prompt`], and registers the handle.

pub mod prompt;
pub mod registry;
pub mod select;

pub use prompt::{TaskPromptTools, build_progress_instruction, build_task_prompt};
pub use registry::{ActiveRun, ActiveRunRegistry};
pub use select::{
    PollCadence, card_urgency, has_card_in_progress, pick_next_card, requires_plan_approval,
};

#[cfg(test)]
mod test;
