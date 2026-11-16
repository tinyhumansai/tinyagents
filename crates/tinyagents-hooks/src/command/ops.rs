//! Lifecycle entry points for command hooks outside tool execution.

use super::{
    context::{TurnIdentity, build_input},
    engine::HookEngine,
    types::*,
};

/// Result of a prompt-submission hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptVerdict {
    /// Submit the prompt and optionally append hook-provided context.
    Submit { additional_context: Option<String> },
    /// Refuse submission with a user-facing explanation.
    Block(String),
}

impl HookEngine {
    /// Fire `sessionStart` and return context for the system prompt.
    pub async fn session_started(
        &self,
        identity: TurnIdentity,
        entrypoint: Option<String>,
    ) -> Option<String> {
        if !self.has_hooks(HookEvent::SessionStart).await {
            return None;
        }
        let input = build_input(
            HookEvent::SessionStart,
            identity,
            HookPayload::Session(SessionPayload {
                entrypoint,
                reason: None,
                duration_ms: None,
            }),
        );
        self.dispatch(HookEvent::SessionStart, input)
            .await
            .output
            .additional_context
    }

    /// Fire `sessionEnd` and release session-scoped hook state.
    pub async fn session_ended(
        &self,
        identity: TurnIdentity,
        reason: &str,
        duration_ms: Option<u64>,
    ) {
        let session_id = identity.session_id.clone();
        if self.has_hooks(HookEvent::SessionEnd).await {
            let input = build_input(
                HookEvent::SessionEnd,
                identity,
                HookPayload::Session(SessionPayload {
                    entrypoint: None,
                    reason: Some(reason.to_owned()),
                    duration_ms,
                }),
            );
            self.dispatch(HookEvent::SessionEnd, input).await;
        }
        if let Some(session_id) = session_id {
            self.forget_session(&session_id).await;
            super::followup::forget(&session_id).await;
        }
    }

    /// Fire `beforeSubmitPrompt`, returning a block or added context.
    pub async fn prompt_submitted(
        &self,
        identity: TurnIdentity,
        prompt: &str,
        attachments: Vec<String>,
    ) -> PromptVerdict {
        if !self.has_hooks(HookEvent::BeforeSubmitPrompt).await {
            return PromptVerdict::Submit {
                additional_context: None,
            };
        }
        let input = build_input(
            HookEvent::BeforeSubmitPrompt,
            identity,
            HookPayload::Prompt(PromptPayload {
                prompt: prompt.to_owned(),
                attachments,
            }),
        );
        let output = self
            .dispatch(HookEvent::BeforeSubmitPrompt, input)
            .await
            .output;
        if output.is_deny() || output.continue_ == Some(false) {
            PromptVerdict::Block(
                output
                    .user_message
                    .or(output.agent_message)
                    .unwrap_or_else(|| "a configured hook blocked this prompt".into()),
            )
        } else {
            PromptVerdict::Submit {
                additional_context: output.additional_context,
            }
        }
    }

    /// Fire `preCompact` and return an optional user-facing message.
    pub async fn pre_compact(
        &self,
        identity: TurnIdentity,
        trigger: &str,
        context_usage_percent: Option<f64>,
        message_count: Option<usize>,
    ) -> Option<String> {
        if !self.has_hooks(HookEvent::PreCompact).await {
            return None;
        }
        let input = build_input(
            HookEvent::PreCompact,
            identity,
            HookPayload::Compact(CompactPayload {
                trigger: trigger.to_owned(),
                context_usage_percent,
                message_count,
            }),
        );
        self.dispatch(HookEvent::PreCompact, input)
            .await
            .output
            .user_message
    }

    /// Fire the gating `subagentStart` event.
    pub async fn subagent_starting(
        &self,
        identity: TurnIdentity,
        subagent_type: &str,
        task: &str,
    ) -> Result<(), String> {
        if !self.has_hooks(HookEvent::SubagentStart).await {
            return Ok(());
        }
        let input = build_input(
            HookEvent::SubagentStart,
            identity.clone(),
            HookPayload::Subagent(SubagentPayload {
                subagent_type: subagent_type.to_owned(),
                task: task.to_owned(),
                parent_conversation_id: identity.conversation_id,
                status: None,
                duration_ms: None,
            }),
        );
        match self
            .dispatch(HookEvent::SubagentStart, input)
            .await
            .denial_reason()
        {
            Some(reason) => Err(reason.to_owned()),
            None => Ok(()),
        }
    }

    /// Fire `subagentStop` and return any follow-up requested for the parent.
    pub async fn subagent_stopped(
        &self,
        identity: TurnIdentity,
        subagent_type: &str,
        task: &str,
        status: &str,
        duration_ms: Option<u64>,
    ) -> Option<String> {
        if !self.has_hooks(HookEvent::SubagentStop).await {
            return None;
        }
        let input = build_input(
            HookEvent::SubagentStop,
            identity.clone(),
            HookPayload::Subagent(SubagentPayload {
                subagent_type: subagent_type.to_owned(),
                task: task.to_owned(),
                parent_conversation_id: identity.conversation_id,
                status: Some(status.to_owned()),
                duration_ms,
            }),
        );
        self.dispatch(HookEvent::SubagentStop, input)
            .await
            .output
            .followup_message
    }

    /// Fire the observational `afterAgentThought` event for a reasoning block.
    ///
    /// The hooks run on a background task, so this returns as soon as they are
    /// scheduled and nothing they print can change the turn. Matchers select on
    /// the thought text.
    pub async fn agent_thought(
        &self,
        identity: TurnIdentity,
        text: &str,
        duration_ms: Option<u64>,
    ) {
        if !self.has_hooks(HookEvent::AfterAgentThought).await {
            return;
        }
        let input = build_input(
            HookEvent::AfterAgentThought,
            identity,
            HookPayload::Text(TextPayload {
                text: text.to_owned(),
                duration_ms,
            }),
        );
        self.dispatch(HookEvent::AfterAgentThought, input).await;
    }
}

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
