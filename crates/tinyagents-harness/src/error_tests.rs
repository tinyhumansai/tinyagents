use super::*;

#[test]
fn rerank_errors_convert_to_embedding_errors_with_context() {
    let source = tinyinference_embeddings::Error::Rerank(
        tinyinference_embeddings::rerank::RerankError::ResponseTooLarge { limit: 17 },
    );

    let converted = TinyAgentsError::from(source);

    assert!(matches!(
        converted,
        TinyAgentsError::Embedding(message) if message.contains("17 bytes")
    ));
}

#[test]
fn tinyinference_budget_refusals_convert_to_terminal_limits() {
    let source = tinyinference_llm::model::budget::BudgetExceeded {
        snapshot: tinyinference_llm::model::budget::BudgetSnapshot::default(),
        requested: tinyinference_llm::model::budget::Spend::default(),
        limits: tinyinference_llm::model::budget::SpendLimits::default(),
    };
    let converted = TinyAgentsError::from(tinyinference_llm::Error::BudgetExceeded(source));

    assert!(converted.is_terminal_limit());
}

#[test]
fn refilling_rate_limit_errors_are_not_terminal_limits() {
    let error = TinyAgentsError::LimitExceeded("rate limit: could not acquire 1 token".into());
    assert!(!error.is_terminal_limit());
}

#[test]
fn provider_budget_refusals_convert_to_terminal_run_limits() {
    let source = tinyinference_llm::model::budget::BudgetExceeded {
        snapshot: tinyinference_llm::model::budget::BudgetSnapshot::default(),
        requested: tinyinference_llm::model::budget::Spend::default(),
        limits: tinyinference_llm::model::budget::SpendLimits::default(),
    };

    let converted = TinyAgentsError::from(tinyinference_llm::Error::BudgetExceeded(source));

    assert!(
        matches!(converted, TinyAgentsError::LimitExceeded(message) if message.contains("budget exceeded"))
    );
}
