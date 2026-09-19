//! Host-neutral composition of durable agent work.
//!
//! This crate coordinates typed team and workflow plans over the established
//! TinyAgents graph, harness, and session layers. Hosts supply persistence
//! roots and concrete worker execution; policy, credentials, tools, progress,
//! and RPC remain host concerns.
//!
//! Dependency direction is deliberately one way:
//! `orchestration -> {graph, harness, session}`. The lower-level crates never
//! depend on this composition layer.

pub mod subagent;
pub mod teams;
pub mod workflow;

#[cfg(test)]
mod boundary_tests {
    #[test]
    fn dependency_direction_stays_one_way_and_host_free() {
        let manifest = include_str!("../Cargo.toml");
        assert!(!manifest.contains("openhuman"));

        for lower_layer in [
            include_str!("../../tinyagents-graph/Cargo.toml"),
            include_str!("../../tinyagents-harness/Cargo.toml"),
            include_str!("../../tinyagents-session/Cargo.toml"),
            include_str!("../../tinyagents-runtime/Cargo.toml"),
        ] {
            assert!(
                !lower_layer.contains("tinyagents-orchestration"),
                "lower TinyAgents layers must not depend on orchestration"
            );
        }
    }

    #[test]
    fn public_team_surface_compiles() {
        fn assert_ledger<L: crate::teams::TeamLedger>() {}
        let _ = assert_ledger::<crate::teams::SessionTeamLedger>;
    }

    #[test]
    fn public_workflow_surface_compiles() {
        fn assert_store<S: crate::workflow::WorkflowStore>() {}
        let _ = assert_store::<crate::workflow::SessionWorkflowStore>;
    }
}
