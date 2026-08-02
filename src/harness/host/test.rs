//! Bundle-level tests. Each capability's own behaviour is tested in its module;
//! these cover the required/optional split and the hand-written `Clone`.

use super::*;

/// The bundle is generic over `State` only because `ModelResolver` is. Nothing
/// in these tests needs a real state, so use unit.
type S = ();

fn required() -> (
    Arc<dyn ContextComposer>,
    Arc<dyn DefinitionRegistry>,
    Arc<dyn SecurityGate>,
    Arc<dyn ModelResolver<S>>,
) {
    (
        Arc::new(StaticContextComposer::new("you are a test agent")),
        Arc::new(InMemoryDefinitionRegistry::new(Vec::new())),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(Arc::new(
            crate::harness::testkit::ScriptedModel::replies(vec!["ok"]),
        ))),
    )
}

fn bundle() -> HostCapabilities<S> {
    let (context, definitions, security, models) = required();
    HostCapabilities::new(context, definitions, security, models)
}

#[test]
fn new_leaves_every_optional_capability_absent() {
    // Absence is the contract: an unconfigured optional capability must be
    // `None`, not a no-op stub silently standing in for one.
    let caps = bundle();
    assert!(caps.memory.is_none());
    assert!(caps.budget.is_none());
    assert!(caps.progress.is_none());
    assert!(caps.learning.is_none());
    assert!(caps.tool_outcomes.is_none());
    assert!(caps.experience.is_none());
}

#[test]
fn with_methods_add_one_capability_without_disturbing_the_others() {
    let caps = bundle().with_progress(Arc::new(NoopProgressSink));
    assert!(caps.progress.is_some());
    assert!(
        caps.memory.is_none(),
        "unrelated capability must stay absent"
    );
    assert!(caps.budget.is_none());
}

#[test]
fn with_methods_chain_to_a_fully_populated_bundle() {
    let caps = bundle()
        .with_memory(Arc::new(InMemoryAgentMemory::default()))
        .with_budget(Arc::new(UnlimitedBudgetGate))
        .with_progress(Arc::new(NoopProgressSink))
        .with_learning(Arc::new(NoopLearningSink))
        .with_tool_outcomes(Arc::new(ErrorFieldClassifier))
        .with_experience(Arc::new(InMemoryExperienceStore::default()));

    assert!(caps.memory.is_some());
    assert!(caps.budget.is_some());
    assert!(caps.progress.is_some());
    assert!(caps.learning.is_some());
    assert!(caps.tool_outcomes.is_some());
    assert!(caps.experience.is_some());
}

#[test]
fn clone_shares_capabilities_rather_than_duplicating_them() {
    // The hand-written `Clone` exists so the bundle stays cloneable for any
    // `State`, including one that is not `Clone`. Cloning must share the
    // `Arc`s — a deep copy would give two halves of the system different
    // memory/budget state.
    let caps = bundle().with_memory(Arc::new(InMemoryAgentMemory::default()));
    let copy = caps.clone();

    assert!(Arc::ptr_eq(&caps.context, &copy.context));
    assert!(Arc::ptr_eq(&caps.definitions, &copy.definitions));
    assert!(Arc::ptr_eq(&caps.security, &copy.security));
    assert!(Arc::ptr_eq(
        caps.memory.as_ref().expect("memory set"),
        copy.memory.as_ref().expect("memory set"),
    ));
}
