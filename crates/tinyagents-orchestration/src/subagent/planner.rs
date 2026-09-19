use async_trait::async_trait;

use super::{PreparedSubagent, SubagentError, SubagentRequest};

/// Host boundary that resolves a request into a complete execution plan.
///
/// It owns agent selection, prompts, limits, workspace metadata, context
/// lineage and tool policy. Returning a plan is intentionally all-or-nothing:
/// the generic driver cannot fill missing host data with defaults.
#[async_trait]
pub trait SubagentPlanner<C: Send + 'static = (), H: Send + 'static = ()>: Send + Sync {
    /// Resolves one request before any execution or terminal persistence.
    async fn prepare(
        &self,
        request: SubagentRequest<C, H>,
    ) -> Result<PreparedSubagent<C>, SubagentError>;
}
