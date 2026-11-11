use super::*;
use crate::cost::CostSource;
use crate::progress::AgentProgress;

#[test]
fn subagent_usage_cost_carries_the_known_spend_and_its_certainty() {
    let usage = SubagentUsage {
        input_tokens: 10,
        output_tokens: 5,
        cached_input_tokens: 2,
        charged_amount_usd: 0.25,
        cost_source: CostSource::Charged,
    };
    let tally = usage.cost();
    assert_eq!(tally.known_usd, 0.25);
    assert_eq!(tally.source, CostSource::Charged);
    assert_eq!(tally.usd(), Some(0.25));
}

#[test]
fn subagent_usage_with_an_unknown_cost_has_no_usd() {
    let usage = SubagentUsage {
        cost_source: CostSource::Unknown,
        ..SubagentUsage::default()
    };
    let tally = usage.cost();
    assert_eq!(tally.source, CostSource::Unknown);
    assert_eq!(tally.usd(), None);
}

#[test]
fn subagent_usage_omits_an_unknown_cost_source_on_the_wire() {
    let usage = SubagentUsage {
        cost_source: CostSource::Unknown,
        ..SubagentUsage::default()
    };
    let json = serde_json::to_value(usage).unwrap();
    assert!(json.get("cost_source").is_none());
    // Absent on the wire reads back as an unknown cost, never a charged one.
    let round_trip: SubagentUsage = serde_json::from_value(json).unwrap();
    assert_eq!(round_trip, usage);
}

#[test]
fn context_compacting_is_a_progress_event() {
    let event = AgentProgress::ContextCompacting;

    assert!(matches!(event, AgentProgress::ContextCompacting));
    assert_ne!(
        std::mem::discriminant(&AgentProgress::ContextCompacting),
        std::mem::discriminant(&AgentProgress::TurnStarted),
    );
}

#[test]
fn subagent_usage_exposes_its_known_cost() {
    let usage = super::SubagentUsage {
        charged_amount_usd: 0.25,
        cost_source: crate::cost::CostSource::Charged,
        ..Default::default()
    };

    assert_eq!(usage.cost().usd(), Some(0.25));
}
