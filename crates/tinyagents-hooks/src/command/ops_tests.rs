use super::*;

#[tokio::test]
async fn lifecycle_entrypoints_are_noops_without_registered_hooks() {
    let engine = HookEngine::default();
    let identity = TurnIdentity::default();
    assert_eq!(engine.session_started(identity.clone(), None).await, None);
    assert_eq!(
        engine
            .prompt_submitted(identity.clone(), "hello", Vec::new())
            .await,
        PromptVerdict::Submit {
            additional_context: None
        }
    );
    assert_eq!(
        engine
            .pre_compact(identity.clone(), "auto", None, None)
            .await,
        None
    );
    assert_eq!(
        engine
            .subagent_starting(identity.clone(), "worker", "task")
            .await,
        Ok(())
    );
    assert_eq!(
        engine
            .subagent_stopped(identity.clone(), "worker", "task", "ok", Some(1))
            .await,
        None
    );
    engine
        .agent_thought(identity.clone(), "thinking", Some(1))
        .await;
    engine.session_ended(identity, "completed", Some(1)).await;
}
