//! Keeps earlier turns' reasoning out of the requests a run builds.
//!
//! See [`ReasoningReplay`] and [`drop_past_turn_reasoning`].

mod strip;
mod types;

pub use strip::drop_past_turn_reasoning;
pub use types::ReasoningReplay;
