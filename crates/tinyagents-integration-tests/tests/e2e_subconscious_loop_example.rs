//! Integration coverage for the runnable subconscious-loop example.

#[path = "../examples/subconscious_loop/autonomous_loop.rs"]
mod autonomous_loop;

use autonomous_loop::{
    SystemState, WorldDiff, build_subconscious_loop_graph, run_subconscious_loop,
};

#[tokio::test]
async fn normal_loop_retrieves_memory_and_holds_single_diff() {
    let graph = build_subconscious_loop_graph().expect("graph compiles");
    let run = graph
        .run(SystemState::new(
            "web",
            "Shift the resource matrix by a small amount.",
        ))
        .await
        .expect("graph runs");

    let visited: Vec<_> = run.visited.iter().map(|node| node.as_str()).collect();
    assert_eq!(
        visited,
        vec![
            "channel_ingestion",
            "frontend_agent",
            "agent_execution",
            "summarization_gate",
            "frontend_agent",
            "context_manager_hook",
        ]
    );
    assert!(run.state.channel_response.contains("Completed. Output"));
    assert_eq!(run.state.retrieved_context.len(), 1);
    assert_eq!(run.state.sequential_diffs.len(), 1);
    assert!(run.state.gated_world_summary.is_none());
    assert!(!run.state.trigger_subconscious);
    assert!(run.state.subconscious_steering.is_empty());

    let helper_state = run_subconscious_loop(SystemState::new("web", "Routine helper smoke."))
        .await
        .expect("helper runs");
    assert!(helper_state.channel_response.contains("Completed. Output"));
}

#[tokio::test]
async fn reasoning_escalation_forces_gate_and_resets_trigger() {
    let graph = build_subconscious_loop_graph().expect("graph compiles");
    let run = graph
        .run(SystemState::new(
            "telegram",
            "CRITICAL cascading sub-agent failure during resource reallocation.",
        ))
        .await
        .expect("graph runs");

    let visited: Vec<_> = run.visited.iter().map(|node| node.as_str()).collect();
    assert!(visited.contains(&"subconscious_eval"));
    assert!(run.state.sequential_diffs.is_empty());
    assert!(run.state.gated_world_summary.is_none());
    assert!(!run.state.trigger_subconscious);
    assert!(
        run.state
            .subconscious_steering
            .contains("Lower sub-agent temperature")
    );
    assert!(
        run.state
            .semantic_history
            .iter()
            .any(|trace| trace.contains("error_code=SUBAGENT_CASCADE"))
    );
}

#[tokio::test]
async fn sequential_diff_threshold_routes_through_subconscious() {
    let graph = build_subconscious_loop_graph().expect("graph compiles");
    let mut state = SystemState::new("web", "Shift the resource matrix again.");
    state.sequential_diffs = vec![
        WorldDiff::new("resource_shift", 9, "stable"),
        WorldDiff::new("resource_shift", 11, "stable"),
    ];

    let run = graph.run(state).await.expect("graph runs");

    let visited: Vec<_> = run.visited.iter().map(|node| node.as_str()).collect();
    assert!(visited.contains(&"subconscious_eval"));
    assert!(run.state.sequential_diffs.is_empty());
    assert!(run.state.gated_world_summary.is_none());
    assert_eq!(
        run.state.subconscious_steering,
        "STEERING_DIRECTIVE: System stable. Preserve current execution policy."
    );
    assert!(
        run.state
            .event_log
            .iter()
            .any(|entry| entry.contains("Aggregated 3 operational shifts"))
    );
}

#[tokio::test]
async fn context_manager_evicts_semantic_history_to_long_term_memory() {
    let graph = build_subconscious_loop_graph().expect("graph compiles");
    let mut state = SystemState::new("web", "Perform routine work.");
    state.context_utilization = 0.84;
    let initial_memory_count = state.long_term_memory.len();

    let run = graph.run(state).await.expect("graph runs");

    let visited: Vec<_> = run.visited.iter().map(|node| node.as_str()).collect();
    assert!(!visited.contains(&"subconscious_eval"));
    assert_eq!(run.state.context_utilization, 0.2);
    assert_eq!(
        run.state.semantic_history,
        vec!["--- semantic history evicted to vector DB ---".to_string()]
    );
    assert!(run.state.long_term_memory.len() > initial_memory_count);
}
