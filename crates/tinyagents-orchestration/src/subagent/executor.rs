use async_trait::async_trait;

use super::{SubagentError, SubagentExecution, SubagentOutcome};

/// Host boundary that executes a prepared subagent through its chosen model,
/// tools and runtime services.
#[async_trait]
pub trait SubagentExecutor<C: Send + 'static = ()>: Send + Sync {
    /// Executes exactly the prepared plan and returns lossless neutral data.
    async fn execute(
        &self,
        execution: SubagentExecution<C>,
    ) -> Result<SubagentOutcome, SubagentError>;
}
