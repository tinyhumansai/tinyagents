//! Runs the autonomous subconscious-loop graph example.
//!
//! ```text
//! cargo run --example subconscious_loop
//! ```

mod autonomous_loop;

use autonomous_loop::{SystemState, run_subconscious_loop};
use tinyagents::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let final_state = run_subconscious_loop(SystemState::new(
        "telegram",
        "Reallocate the resource matrix and report whether the system remains stable.",
    ))
    .await?;

    println!("=== Autonomous closed-loop subconscious example ===");
    println!("channel response : {}", final_state.channel_response);
    println!("retrieved context: {:?}", final_state.retrieved_context);
    println!("semantic history : {:?}", final_state.semantic_history);
    println!("sequential diffs : {:?}", final_state.sequential_diffs);
    println!("steering         : {}", final_state.subconscious_steering);
    println!("context usage    : {:.2}", final_state.context_utilization);
    println!("visited events   :");
    for event in &final_state.event_log {
        println!("  - {event}");
    }

    Ok(())
}
