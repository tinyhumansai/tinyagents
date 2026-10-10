use super::tests::{meta, turn_usage};
use super::*;
use crate::transcript::{MessageUsage, append_transcript_turn, read_transcript};

fn priced(input: u64, cost: f64, source: Option<UsageCostSource>) -> TurnUsage {
    let mut usage = turn_usage(input, 10, 0);
    usage.usage.cost_usd = cost;
    usage.usage.cost_source = source;
    usage
}

/// A record written before `cost_source` existed may carry a guessed rate: a
/// glm-5.3-flash thread recorded $4.25 at a $3/$15 per-MTok default when the
/// provider billed about $0.30. Its cost must land in the unpriced split, not
/// be summed in with charges.
#[test]
fn cost_split_keeps_stated_charges_apart_from_records_without_a_source() {
    let mut split = CostSplit::default();
    split.add(&priced(1_000, 0.25, Some(UsageCostSource::Charged)));
    split.add(&priced(2_000, 4.25, None));
    split.add(&priced(3_000, 0.0, Some(UsageCostSource::Unknown)));

    assert!((split.priced_cost_usd - 0.25).abs() < f64::EPSILON);
    assert_eq!(split.priced_source, Some(UsageCostSource::Charged));
    assert_eq!(split.unpriced_turns, 2);
    assert_eq!(split.unpriced_input_tokens, 5_000);
    assert_eq!(split.unpriced_output_tokens, 20);
}

/// An estimate anywhere makes the priced total an estimate.
#[test]
fn cost_split_source_is_the_least_certain_of_its_records() {
    let mut a = CostSplit::default();
    a.add(&priced(1, 0.1, Some(UsageCostSource::Charged)));
    let mut b = CostSplit::default();
    b.add(&priced(1, 0.2, Some(UsageCostSource::Estimated)));
    a.merge(&b);
    assert_eq!(a.priced_source, Some(UsageCostSource::Estimated));
    assert!((a.priced_cost_usd - 0.3).abs() < 1e-12);
}

/// `cost_source` round-trips and is absent from (and defaults on) old lines.
#[test]
fn cost_source_round_trips_and_is_omitted_when_unset() {
    let usage = MessageUsage {
        cost_usd: 0.01,
        cost_source: Some(UsageCostSource::Charged),
        ..Default::default()
    };
    let json = serde_json::to_value(&usage).expect("serialize");
    assert_eq!(json["cost_source"], "charged");
    let back: MessageUsage = serde_json::from_value(json).expect("parse");
    assert_eq!(back, usage);

    let unset = serde_json::to_value(MessageUsage::default()).expect("serialize");
    assert!(unset.get("cost_source").is_none());
}

fn write_cost_turns(
    workspace: &Path,
    stem: &str,
    header: &TranscriptMeta,
    usages: &[TurnUsage],
) -> PathBuf {
    let path = workspace.join("session_raw").join(format!("{stem}.jsonl"));
    std::fs::create_dir_all(path.parent().unwrap()).expect("create session_raw");
    let mut persisted = Vec::new();
    for (i, usage) in usages.iter().enumerate() {
        let mut next = persisted.clone();
        next.push(TranscriptMessage::new("user", format!("q{i}")));
        next.push(TranscriptMessage::assistant(format!("a{i}")));
        append_transcript_turn(
            &path,
            &persisted,
            &next,
            header,
            Some(usage),
            Some(&format!("{stem}-req-{i}")),
        )
        .expect("append cost turn");
        persisted = read_transcript(&path)
            .expect("read persisted turn")
            .messages;
    }
    path
}

fn mixed_cost_usages() -> Vec<TurnUsage> {
    [
        (100, 0.25, Some(UsageCostSource::Charged)),
        (200, 0.5, Some(UsageCostSource::Estimated)),
        (300, 1.0, Some(UsageCostSource::Unknown)),
        (400, 4.0, None),
    ]
    .into_iter()
    .map(|(input, cost, source)| {
        let mut usage = priced(input, cost, source);
        usage.usage.cached_input = input / 10;
        usage
    })
    .collect()
}

fn expected_mixed_split() -> CostSplit {
    CostSplit {
        priced_cost_usd: 0.75,
        priced_source: Some(UsageCostSource::Estimated),
        unpriced_turns: 2,
        unpriced_input_tokens: 700,
        unpriced_output_tokens: 20,
        unpriced_cached_input_tokens: 70,
    }
}

#[test]
fn transcript_spend_preserves_mixed_cost_sources_after_a_durable_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = write_cost_turns(
        tmp.path(),
        "1790000000_root",
        &meta("orchestrator", "root", Some("cost-thread")),
        &mixed_cost_usages(),
    );
    let transcript = read_transcript(&path).expect("read transcript");
    let spend = transcript_spend(&transcript);
    assert_eq!(spend.cost_split, expected_mixed_split());
    assert_eq!(spend.cost_usd, 5.75, "legacy total remains available");
    assert_eq!(spend.turns, 4);
    assert_eq!(spend.input_tokens, 1_000);
}

#[test]
fn thread_spend_merges_cost_sources_across_roots_and_subagent_runs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let thread = "cost-thread";
    let usages = mixed_cost_usages();
    let roots = ["1790000000_root", "1790000100_root"];
    for (i, root) in roots.iter().enumerate() {
        let turns = &usages[i * 2..i * 2 + 2];
        write_cost_turns(
            tmp.path(),
            root,
            &meta("orchestrator", "root", Some(thread)),
            turns,
        );
        write_cost_turns(
            tmp.path(),
            &format!("{root}__1790000200_researcher"),
            &meta("researcher", "subagent", Some("worker-thread")),
            turns,
        );
    }
    let spend = thread_spend(tmp.path(), thread);
    assert!(spend.found_transcript);
    assert_eq!(spend.root.cost_split, expected_mixed_split());
    assert_eq!(spend.root.turns, 4);
    assert_eq!(spend.root.cost_usd, 5.75);
    let (child, runs) = &spend.subagents["researcher"];
    assert_eq!(*runs, 2);
    assert_eq!(child.cost_split, expected_mixed_split());
    assert_eq!(child.turns, 4);
    assert_eq!(child.cost_usd, 5.75);
}
