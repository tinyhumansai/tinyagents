//! Subprocess lifecycle for the Claude Agent SDK provider.
//!
//! [`ClaudeAgentSdkProvider`] implements `ChatModel<()>` by shelling out to
//! `claude -p --output-format stream-json` once per [`invoke`][ChatModel::invoke]
//! call: the harness never links against the Claude Agent SDK directly, it
//! only speaks the CLI's stdin/stdout contract. This is the "prompt-guided"
//! sibling of [`crate::providers::claude_code`], which drives the same CLI in
//! full agentic (multi-turn, tool-using) mode via a long-lived session
//! instead of a single stateless invocation; use this module when a plain
//! one-shot completion is enough. Wire message shapes for the NDJSON stream
//! live in `protocol`.

mod protocol;

use crate::tool::{coalesce_prompt_tool_results, with_prompt_tool_instructions};
use anyhow::Context;
use async_trait::async_trait;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelProfile, ModelRequest, ModelResponse};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use protocol::SdkMessage;

/// Configuration for the Claude CLI subprocess adapter.
#[derive(Debug, Clone)]
pub struct ClaudeAgentSdkConfig {
    /// Whether the embedding host enables this provider.
    pub enabled: bool,
    /// Claude CLI binary or path.
    pub binary: String,
    /// Model used when a request does not override it.
    pub default_model: String,
    /// Optional maximum request budget in US dollars.
    pub max_budget_usd: Option<f64>,
}

impl Default for ClaudeAgentSdkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            binary: "claude".to_string(),
            default_model: "claude-sonnet-4-6".to_string(),
            max_budget_usd: None,
        }
    }
}

/// Prompt-guided chat model backed by `claude -p`.
pub struct ClaudeAgentSdkProvider {
    pub(super) config: ClaudeAgentSdkConfig,
    profile: ModelProfile,
}

/// Fully-formed CLI arguments and stdin payload for one `claude -p` call.
struct ClaudeInvocation {
    args: Vec<String>,
    stdin: String,
}

/// Builds the argv and stdin payload for one `claude -p` invocation.
///
/// The system prompt, when present, is wrapped in `[SYSTEM]...[/SYSTEM]`
/// tags and prepended to stdin rather than passed as a CLI flag, keeping the
/// full request off argv (see [`ClaudeAgentSdkProvider::invoke_cli`]). A
/// `max_budget_usd` also pins `--max-turns 10` so a budget-capped run cannot
/// wander indefinitely before the budget check kicks in.
fn build_invocation(
    system_prompt: Option<&str>,
    message: &str,
    model: &str,
    max_budget_usd: Option<f64>,
) -> ClaudeInvocation {
    let stdin = match system_prompt {
        Some(system) if !system.trim().is_empty() => {
            format!("[SYSTEM]\n{system}\n[/SYSTEM]\n\n{message}")
        }
        _ => message.to_string(),
    };
    let mut args = vec![
        "-p".to_string(),
        "--model".to_string(),
        model.to_string(),
        "--output-format".to_string(),
        "stream-json".to_string(),
        "--no-color".to_string(),
    ];
    if let Some(budget) = max_budget_usd {
        args.push("--max-turns".to_string());
        args.push("10".to_string());
        args.push("--budget".to_string());
        args.push(format!("{budget:.4}"));
    }
    ClaudeInvocation { args, stdin }
}

/// Render the complete non-system transcript for the stateless CLI process.
///
/// Every `claude -p` invocation starts a fresh process, so passing only the
/// final user turn loses the question and any assistant/tool turns that led to
/// it. Keep the common one-user request compact, but label every turn when a
/// transcript is present so the model can distinguish its own prior output
/// from the next user turn.
fn render_transcript(messages: &[Message]) -> String {
    let non_system: Vec<&Message> = messages
        .iter()
        .filter(|message| !matches!(message, Message::System(_)))
        .collect();
    if non_system.len() == 1 {
        return non_system[0].text();
    }

    non_system
        .into_iter()
        .map(|message| {
            let role = match message {
                Message::User(_) => "USER",
                Message::Assistant(_) => "ASSISTANT",
                Message::Tool(_) => "TOOL",
                Message::System(_) => unreachable!("system messages were filtered"),
            };
            format!("[{role}]\n{}\n[/{role}]", message.text())
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Wraps a subprocess spawn failure with the binary name for a legible error.
fn spawn_error(binary: &str, source: std::io::Error) -> anyhow::Error {
    let message = format!("failed to spawn claude binary '{binary}': {source}");
    anyhow::Error::new(source).context(message)
}

impl ClaudeAgentSdkProvider {
    /// Creates a provider that defaults to `config.default_model` for every
    /// call.
    pub fn new(config: ClaudeAgentSdkConfig) -> Self {
        let model = config.default_model.clone();
        Self::for_model(config, model)
    }

    /// Creates a provider pinned to `model`, overriding `config.default_model`
    /// for this instance's [`ModelProfile`]. A `model` set explicitly on a
    /// given [`ModelRequest`] still takes precedence over both.
    pub fn for_model(config: ClaudeAgentSdkConfig, model: impl Into<String>) -> Self {
        Self {
            config,
            profile: ModelProfile {
                provider: Some("claude-agent-sdk".to_string()),
                model: Some(model.into()),
                ..Default::default()
            },
        }
    }

    /// Spawns `claude -p`, streams and decodes its NDJSON stdout, and
    /// returns the assembled response text.
    ///
    /// The request body goes over stdin rather than argv (see
    /// [`build_invocation`]) so large prompts do not hit OS argv-length
    /// limits. Stderr is drained on a concurrent task while stdout is read,
    /// because leaving either pipe unread while the other fills can deadlock
    /// the child process. Reading stdout is bounded by a 120s timeout and
    /// waiting for process exit by a separate 30s timeout; either firing
    /// kills the child and returns an error. When the CLI streams no final
    /// `Result` message, the joined `Text` chunks are used as a fallback.
    async fn invoke_cli(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
    ) -> anyhow::Result<String> {
        let model = if model.is_empty() {
            &self.config.default_model
        } else {
            model
        };

        // `claude -p` reads stdin in non-interactive mode. Keep the full
        // request out of argv so large harness prompts can spawn on Windows.
        let invocation =
            build_invocation(system_prompt, message, model, self.config.max_budget_usd);

        let mut cmd = Command::new(&self.config.binary);
        cmd.args(&invocation.args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::piped())
            .kill_on_drop(true);

        tracing::debug!(
            "[claude_agent_sdk] spawning claude binary={} model={} message_len={}",
            self.config.binary,
            model,
            invocation.stdin.len()
        );

        let mut child = cmd.spawn().map_err(|source| {
            tracing::warn!(
                error = %source,
                binary = %self.config.binary,
                "[claude_agent_sdk] failed to spawn claude binary"
            );
            spawn_error(&self.config.binary, source)
        })?;

        let mut stdin = child
            .stdin
            .take()
            .context("claude subprocess has no stdin")?;
        stdin
            .write_all(invocation.stdin.as_bytes())
            .await
            .context("failed to write claude request to stdin")?;
        stdin
            .shutdown()
            .await
            .context("failed to close claude subprocess stdin")?;
        drop(stdin);

        let stdout = child
            .stdout
            .take()
            .context("claude subprocess has no stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("claude subprocess has no stderr")?;

        // Drain stderr concurrently to prevent pipe-buffer stalls and capture failure context.
        let stderr_task = tokio::spawn(async move {
            let mut err_lines = BufReader::new(stderr).lines();
            let mut buf = String::new();
            while let Ok(Some(line)) = err_lines.next_line().await {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(&line);
            }
            buf
        });

        let mut lines = BufReader::new(stdout).lines();
        let mut text_parts: Vec<String> = Vec::new();
        let mut result_text: Option<String> = None;
        let mut error_message: Option<String> = None;

        let read_result = timeout(Duration::from_secs(120), async {
            while let Some(line) = lines.next_line().await? {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                tracing::trace!(
                    "[claude_agent_sdk] ndjson line received line_len={}",
                    line.len()
                );
                match serde_json::from_str::<SdkMessage>(&line) {
                    Ok(SdkMessage::Text { text }) => {
                        text_parts.push(text);
                    }
                    Ok(SdkMessage::Result {
                        result,
                        is_error,
                        total_cost_usd,
                    }) => {
                        if let Some(cost) = total_cost_usd {
                            tracing::debug!(
                                "[claude_agent_sdk] request completed total_cost_usd={:.6}",
                                cost
                            );
                        }
                        if is_error {
                            error_message = Some(result.unwrap_or_else(|| {
                                "claude subprocess returned an error".to_string()
                            }));
                        } else {
                            result_text = result;
                        }
                    }
                    Ok(SdkMessage::Error { error }) => {
                        error_message = Some(error.message);
                    }
                    Ok(SdkMessage::Unknown) => {
                        tracing::trace!("[claude_agent_sdk] unknown ndjson message type, skipping");
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            line_len = line.len(),
                            "[claude_agent_sdk] failed to parse ndjson line"
                        );
                    }
                }
            }
            anyhow::Ok(())
        })
        .await;

        match read_result {
            Ok(inner) => inner?,
            Err(_) => {
                let _ = child.kill().await;
                anyhow::bail!("[claude_agent_sdk] subprocess timed out while reading output");
            }
        }

        let status = timeout(Duration::from_secs(30), child.wait())
            .await
            .map_err(|_| {
                anyhow::anyhow!("[claude_agent_sdk] subprocess timed out while waiting for exit")
            })??;
        let stderr_output = stderr_task.await.unwrap_or_default();
        tracing::debug!("[claude_agent_sdk] subprocess exited status={}", status);

        if !status.success() {
            anyhow::bail!(
                "[claude_agent_sdk] claude subprocess exited with non-zero status {}; stderr={}",
                status,
                stderr_output
            );
        }

        if let Some(err) = error_message {
            anyhow::bail!("[claude_agent_sdk] error from claude CLI: {err}");
        }

        // Use the final result message if present; otherwise join streaming text parts.
        let output = result_text
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| text_parts.join(""));

        tracing::debug!(
            "[claude_agent_sdk] response collected output_len={}",
            output.len()
        );

        Ok(output)
    }
}

#[async_trait]
impl ChatModel<()> for ClaudeAgentSdkProvider {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    /// Identity for harness response-cache scoping: the SDK binary and the
    /// selected model. No credential is involved on this path.
    fn cache_identity(&self) -> Option<String> {
        Some(format!(
            "claude_agent_sdk:{}:{}",
            self.config.binary,
            self.profile
                .model
                .as_deref()
                .unwrap_or(&self.config.default_model)
        ))
    }

    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let messages = coalesce_prompt_tool_results(&request.messages);
        let messages = with_prompt_tool_instructions(&messages, &request.tools);
        let system = coalesce_system_prompt(&messages);
        let transcript = render_transcript(&messages);
        let model = request
            .model
            .as_deref()
            .or(self.profile.model.as_deref())
            .unwrap_or(&self.config.default_model);
        let output = self
            .invoke_cli(system.as_deref(), &transcript, model)
            .await
            .map_err(|error| tinyinference_llm::Error::Model(error.to_string()))?;

        let response = ModelResponse::assistant(output);
        Ok(if request.tools.is_empty() {
            response
        } else {
            crate::tool::apply_prompt_tool_calls(response)
        })
    }
}

/// Join every system message into the one system prompt the CLI accepts.
///
/// The CLI takes a single `--system-prompt`, and this used to `find_map` the
/// first system message and drop the rest (codex on #6068). What gets dropped is
/// not incidental: the artifact contents list and the turn-cap wrap-up are both
/// appended as system messages *precisely because* a system message is the one
/// thing compression and the trim will not remove. Taking only the first put
/// them back to being removable — on this provider alone, and silently.
fn coalesce_system_prompt(messages: &[Message]) -> Option<String> {
    let parts: Vec<String> = messages
        .iter()
        .filter(|message| matches!(message, Message::System(_)))
        .map(|message| message.text())
        .filter(|text| !text.trim().is_empty())
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

#[cfg(test)]
#[path = "test.rs"]
mod tests;
