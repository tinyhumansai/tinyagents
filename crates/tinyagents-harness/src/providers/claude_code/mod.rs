//! Claude Code CLI model adapter.
//!
//! [`ClaudeCodeProvider`] implements `ChatModel<()>` by driving the
//! `claude` CLI as a long-lived, session-resuming subprocess (one turn per
//! [`invoke`][ChatModel::invoke]/[`stream`][ChatModel::stream] call, sharing a
//! `--session-id`/`--resume` UUID across a conversation). This is the
//! agentic sibling of [`crate::providers::claude_agent_sdk`]: Claude Code
//! runs its own built-in tools internally and never returns them to this
//! harness as `ToolCall`s (see `event_mapper`), so from the harness's
//! point of view it behaves as a prompt-guided chat model even though the
//! CLI itself is doing real multi-step tool use.
//!
//! ## How the pieces fit together
//! - This file wires request/response conversion (`request_messages`,
//!   `model_response`) and owns per-thread concurrency
//!   ([`MAX_CONCURRENT_TURNS`] semaphore + per-thread mutex in `run_chat`).
//! - [`driver`] spawns the CLI once per turn and drives its stdin/stdout.
//! - `input_builder` renders the JSONL stdin payload; `stream_parser`
//!   parses the JSONL stdout back into typed events; `event_mapper` folds
//!   those events into deltas and a final response.
//! - `session_store` persists the thread-key → CC session UUID mapping so
//!   a conversation resumes instead of restarting.
//! - [`auth`] resolves which credential the spawned CLI uses;
//!   [`auth_status`] separately probes the CLI's own sign-in state for UIs.
//! - [`settings`] persists the user's full-access opt-in;
//!   [`version_check`] locates and version-gates the CLI binary; [`types`]
//!   holds the small shared value types.
//! - `bridge` holds the crate-private normalized message/response shapes
//!   these modules pass between each other.
//!
//! See `README.md` in this directory for the CLI invocation shape, the
//! permission/sandbox model, and the auth resolution order in full detail.

pub mod auth;
pub mod auth_status;
mod bridge;
pub mod driver;
mod event_mapper;
mod input_builder;
mod session_store;
pub mod settings;
mod stream_parser;
pub mod types;
pub mod version_check;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::tool::{
    ToolCallStreamScrubber, apply_prompt_tool_calls, coalesce_prompt_tool_results,
    with_prompt_tool_instructions,
};
use async_trait::async_trait;
use bridge::{ChatMessage, ChatResponse, ProviderDelta};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use tinyinference_llm::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
    ResponseFormat,
};
use tinyinference_llm::usage::Usage;
use tokio::sync::Semaphore;

/// Aborts the wrapped task when dropped, so a caller that stops polling the
/// stream returned by [`ChatModel::stream`] does not leave the spawned CLI
/// driver task running past the caller's interest in it.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Provider string prefix used by host routing grammars.
pub const PROVIDER_PREFIX: &str = "claude-code:";
/// Maximum concurrent Claude subprocess turns per provider.
pub const MAX_CONCURRENT_TURNS: usize = 4;

/// Renders the JSONL stdin payload Claude Code receives for a model request.
///
/// This is exposed for hosts that need to verify their typed multimodal
/// conversion without depending on the adapter's internal bridge types.
pub fn render_request_stdin(request: &ModelRequest, is_new_session: bool) -> Vec<u8> {
    input_builder::build_stdin(&request_messages(request), is_new_session)
}

#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
#[allow(unsafe_code)]
pub(crate) fn test_set_env(key: impl AsRef<std::ffi::OsStr>, value: impl AsRef<std::ffi::OsStr>) {
    // SAFETY: every moved environment-mutating test serializes access through
    // `ENV_TEST_LOCK`; no provider work runs concurrently in those tests.
    unsafe { std::env::set_var(key, value) }
}

#[cfg(test)]
#[allow(unsafe_code)]
pub(crate) fn test_remove_env(key: impl AsRef<std::ffi::OsStr>) {
    // SAFETY: see `test_set_env`.
    unsafe { std::env::remove_var(key) }
}

/// Claude Code CLI-backed inference model.
#[derive(Clone)]
pub struct ClaudeCodeProvider {
    /// Model name passed to the CLI.
    pub model: String,
    bin_path: PathBuf,
    workspace_dir: PathBuf,
    project_dir: PathBuf,
    anthropic_api_key: Option<String>,
    semaphore: Arc<Semaphore>,
    thread_locks: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    session_store: Arc<session_store::SessionStore>,
    profile: ModelProfile,
    mcp_provider: Option<Arc<dyn driver::McpEndpointProvider>>,
}

impl std::fmt::Debug for ClaudeCodeProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClaudeCodeProvider")
            .field("model", &self.model)
            .field("bin_path", &self.bin_path)
            .field("workspace_dir", &self.workspace_dir)
            .field("project_dir", &self.project_dir)
            .finish_non_exhaustive()
    }
}

impl ClaudeCodeProvider {
    /// Construct a provider with an already-resolved CLI binary.
    pub fn new(
        model: impl Into<String>,
        bin_path: PathBuf,
        workspace_dir: PathBuf,
        project_dir: PathBuf,
        anthropic_api_key: Option<String>,
    ) -> Self {
        let model = model.into();
        Self {
            profile: ModelProfile {
                provider: Some("claude-code".into()),
                model: Some(model.clone()),
                // Claude Code executes its own native tools internally and
                // intentionally never returns them to the harness. Advertise
                // prompt-guided tool handling so the harness does not wait for
                // tool calls that this adapter cannot surface.
                tool_calling: false,
                parallel_tool_calls: false,
                streaming: true,
                streaming_tool_chunks: false,
                ..Default::default()
            },
            model,
            bin_path,
            project_dir,
            session_store: Arc::new(session_store::SessionStore::open(&workspace_dir)),
            workspace_dir,
            anthropic_api_key,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_TURNS)),
            thread_locks: Arc::new(Mutex::new(HashMap::new())),
            mcp_provider: None,
        }
    }

    /// Resolve the installed CLI and environment authentication.
    pub fn from_env(
        model: impl Into<String>,
        workspace_dir: PathBuf,
        project_dir: PathBuf,
    ) -> anyhow::Result<Self> {
        match version_check::probe() {
            types::CliStatus::Ok { path, .. } => {
                let (_, key) = auth::resolve();
                Ok(Self::new(
                    model,
                    path.into(),
                    workspace_dir,
                    project_dir,
                    key,
                ))
            }
            types::CliStatus::NotInstalled => anyhow::bail!(
                "[claude-code] `claude` CLI not installed; require >= {}",
                types::MIN_CLI_VERSION
            ),
            types::CliStatus::Outdated {
                version,
                min_required,
                path,
            } => anyhow::bail!(
                "[claude-code] `claude` CLI at {path} is version {version}; require >= {min_required}"
            ),
            types::CliStatus::Unusable { path, reason } => {
                anyhow::bail!("[claude-code] `claude` CLI at {path} unusable: {reason}")
            }
        }
    }

    /// Attach a host-provided authenticated MCP endpoint resolver.
    pub fn with_mcp_provider(mut self, provider: Arc<dyn driver::McpEndpointProvider>) -> Self {
        self.mcp_provider = Some(provider);
        self
    }

    /// Runs one turn: acquires a concurrency permit, serializes against any
    /// other in-flight turn on the same `thread_id`, and delegates to
    /// [`driver::run_turn`].
    ///
    /// The per-thread lock (not just the global semaphore) matters because
    /// two overlapping calls for the same conversation would otherwise race
    /// on the same CC `--session-id`/`--resume` UUID. The lock map entry for
    /// an idle thread is opportunistically removed once its only other
    /// holder is this method's own now-dropped guard, so the map does not
    /// grow unboundedly across many distinct threads.
    async fn run_chat(
        &self,
        messages: &[ChatMessage],
        stream: Option<&tokio::sync::mpsc::Sender<ProviderDelta>>,
        model_override: Option<&str>,
        thread_id: String,
    ) -> anyhow::Result<ChatResponse> {
        // Acquire the per-thread mutex *before* the global concurrency
        // semaphore (M-14). Reversed, N callers on one busy thread each hold
        // a global permit while blocked on the same thread lock — that is
        // head-of-line blocking for every *other* thread's turns, which the
        // semaphore exists to admit. Waiting on the free, per-thread lock
        // first means a caller only claims a global permit once it can
        // actually make progress.
        let lock_key = thread_id.clone();
        let thread_lock = {
            let mut locks = self
                .thread_locks
                .lock()
                .expect("claude-code thread lock map poisoned");
            locks
                .entry(thread_id.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        let _thread_guard = thread_lock.lock().await;
        let _permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| anyhow::anyhow!("claude-code semaphore closed: {error}"))?;
        let append_system_prompt = coalesce_system_prompt(messages);
        let result = driver::run_turn(driver::TurnContext {
            bin_path: self.bin_path.clone(),
            workspace_dir: self.workspace_dir.clone(),
            project_dir: self.project_dir.clone(),
            thread_id,
            model: model_override.unwrap_or(&self.model).to_string(),
            append_system_prompt,
            messages,
            session_store: self.session_store.clone(),
            stream,
            anthropic_api_key: self.anthropic_api_key.clone(),
            mcp_provider: self.mcp_provider.clone(),
        })
        .await;
        drop(_thread_guard);

        // Remove idle entries without disrupting a waiter that already holds
        // a clone of the same lock. A new caller racing after removal creates
        // a fresh lock only after this turn has released the old one.
        if Arc::strong_count(&thread_lock) == 2 {
            let mut locks = self
                .thread_locks
                .lock()
                .expect("claude-code thread lock map poisoned");
            if locks
                .get(&lock_key)
                .is_some_and(|lock| Arc::ptr_eq(lock, &thread_lock))
                && Arc::strong_count(locks.get(&lock_key).expect("lock checked above")) == 2
            {
                locks.remove(&lock_key);
            }
        }
        result
    }
}

/// Claude Code accepts one appended system prompt, so preserve every system
/// message in order instead of silently dropping middleware additions after the
/// first one.
fn coalesce_system_prompt(messages: &[ChatMessage]) -> Option<String> {
    let parts: Vec<&str> = messages
        .iter()
        .filter(|message| message.role == "system")
        .map(|message| message.content.as_str())
        .filter(|content| !content.trim().is_empty())
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// Resolve the caller's logical conversation id. A content hash is not a
/// conversation id: two independent threads can begin with the same text.
/// Callers should provide `metadata.thread_id` (or the equivalent
/// `conversation_id`/`session_id`); `continuation_id` is also accepted for
/// integrations that already carry a provider-neutral continuation handle.
/// Requests without an id get an isolated session rather than risking a
/// cross-conversation resume.
fn thread_key_from_request(request: &ModelRequest) -> String {
    const METADATA_KEYS: &[&str] = &["thread_id", "conversation_id", "session_id"];
    for key in METADATA_KEYS {
        if let Some(value) = request
            .metadata
            .get(*key)
            .and_then(serde_json::Value::as_str)
            && !value.trim().is_empty()
        {
            return value.to_string();
        }
    }
    if let Some(value) = request
        .continuation_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return value.to_string();
    }
    format!("ephemeral_{}", uuid::Uuid::new_v4())
}

/// Converts a `ModelRequest` into the flattened [`ChatMessage`] list this
/// provider sends downstream: prompt-tool coalescing/instructions applied
/// first, then any structured [`ResponseFormat`] is appended as a trailing
/// system instruction (see [`response_format_instruction`]).
fn request_messages(request: &ModelRequest) -> Vec<ChatMessage> {
    let mut messages = coalesce_prompt_tool_results(&request.messages);
    if !request.tools.is_empty() {
        messages = with_prompt_tool_instructions(&messages, &request.tools);
    }
    if let Some(instruction) = response_format_instruction(request.response_format.as_ref()) {
        messages.push(Message::system(instruction));
    }
    messages
        .iter()
        .map(|message| {
            let role = match message {
                Message::System(_) => "system",
                Message::User(_) => "user",
                Message::Assistant(_) => "assistant",
                Message::Tool(_) => "tool",
            };
            let content = match message {
                Message::System(value) => render_content(&value.content),
                Message::User(value) => render_content(&value.content),
                Message::Assistant(value) => render_content(&value.content),
                Message::Tool(value) => render_content(&value.content),
            };
            ChatMessage::new(role, content)
        })
        .collect()
}

/// Claude Code does not expose a provider-native JSON-schema switch. Carry the
/// requested schema in the system prompt so the provider-schema strategy still
/// has a concrete wire-level instruction and the normal extractor can validate
/// the returned JSON.
fn response_format_instruction(format: Option<&ResponseFormat>) -> Option<String> {
    let schema = match format? {
        ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
            Some((name, schema))
        }
        ResponseFormat::Text | ResponseFormat::JsonObject => None,
    }?;
    Some(format!(
        "Return only valid JSON for the `{}` response. Do not include markdown fences or prose.\nJSON Schema:\n{}",
        schema.0,
        serde_json::to_string_pretty(schema.1).unwrap_or_else(|_| schema.1.to_string())
    ))
}

/// Flattens a message's content blocks to plain text: text and reasoning
/// blocks pass through, JSON/extension blocks are stringified, an image
/// becomes an `[OH_IMAGE:<url>]` marker `input_builder` later rehydrates,
/// and redacted-thinking blocks are dropped (nothing to show).
fn render_content(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.clone()),
            ContentBlock::Image(image) => Some(format!("[OH_IMAGE:{}]", image.url)),
            ContentBlock::Json(value) | ContentBlock::ProviderExtension(value) => {
                Some(value.to_string())
            }
            ContentBlock::Thinking { text, .. } => Some(text.clone()),
            ContentBlock::RedactedThinking { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Converts an internal [`ChatResponse`] into the harness's `ModelResponse`.
/// `tool_calls` is always empty (see `event_mapper`); cost is surfaced via
/// `raw` only when the CLI reported a non-zero charge, since `finish_reason`
/// is always `"stop"` — the CLI does not distinguish other reasons.
fn model_response(response: ChatResponse) -> ModelResponse {
    let usage = response.usage.map(|value| Usage {
        input_tokens: value.input_tokens,
        output_tokens: value.output_tokens,
        total_tokens: value.input_tokens.saturating_add(value.output_tokens),
        cache_read_tokens: value.cached_input_tokens,
        cache_creation_tokens: value.cache_creation_tokens,
        reasoning_tokens: value.reasoning_tokens,
        charged_amount: None,
        context_window_tokens: None,
    });
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: response.text.into_iter().map(ContentBlock::Text).collect(),
            tool_calls: Vec::new(),
            usage,
        },
        usage,
        finish_reason: Some("stop".into()),
        raw: response
            .usage
            .filter(|value| value.charged_amount_usd > 0.0)
            .map(|value| serde_json::json!({"total_cost_usd": value.charged_amount_usd})),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// [`model_response`] plus prompt-tool-call extraction when the request
/// declared tools, since this provider never returns native `ToolCall`s.
fn model_response_with_tools(response: ChatResponse, has_tools: bool) -> ModelResponse {
    let response = model_response(response);
    if has_tools {
        apply_prompt_tool_calls(response)
    } else {
        response
    }
}

/// Classifies a driver failure into `Model` (retryable) or `Validation`
/// (not) by running its message through the shared provider-failure
/// classifier, so a CLI spawn/timeout/exit failure gets the same retry
/// treatment as other providers' errors.
fn map_error(error: anyhow::Error) -> tinyinference_llm::Error {
    let message = format!("claude-code model call failed: {error}");
    if !matches!(
        tinyinference_llm::classify_provider_failure(None, None, &message),
        tinyinference_llm::ProviderFailureClass::NonRetryable
            | tinyinference_llm::ProviderFailureClass::NonRetryableRateLimit
    ) {
        tinyinference_llm::Error::Model(message)
    } else {
        tinyinference_llm::Error::Validation(message)
    }
}

#[async_trait]
impl ChatModel<()> for ClaudeCodeProvider {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }
    fn cache_identity(&self) -> Option<String> {
        Some(format!(
            "claude_code:{}:{}:{}",
            self.bin_path.display(),
            self.model,
            self.project_dir.display()
        ))
    }
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let thread_id = thread_key_from_request(&request);
        let has_tools = !request.tools.is_empty();
        let messages = request_messages(&request);
        self.run_chat(&messages, None, request.model.as_deref(), thread_id)
            .await
            .map(|response| model_response_with_tools(response, has_tools))
            .map_err(map_error)
    }
    async fn stream(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelStream> {
        let provider = self.clone();
        let thread_id = thread_key_from_request(&request);
        let model_override = request.model.clone();
        let has_tools = !request.tools.is_empty();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = AbortOnDrop(tokio::spawn(async move {
            let _ = tx.send(ModelStreamItem::Started);
            // Claude Code is prompt-guided whenever tools are present, so its
            // text deltas can split `<tool_call>` markup across arbitrary CLI
            // chunks. Hold that markup back from live consumers; terminal
            // parsing below still turns the complete block into a ToolCall.
            let mut tool_call_scrubber = has_tools.then(ToolCallStreamScrubber::new);
            let messages = request_messages(&request);
            let (delta_tx, mut delta_rx) = tokio::sync::mpsc::channel(64);
            let call = provider.run_chat(
                &messages,
                Some(&delta_tx),
                model_override.as_deref(),
                thread_id,
            );
            tokio::pin!(call);
            let response = loop {
                tokio::select! { delta = delta_rx.recv() => if let Some(delta) = delta { forward_delta(&tx, delta, tool_call_scrubber.as_mut()); }, response = &mut call => break response }
            };
            while let Ok(delta) = delta_rx.try_recv() {
                forward_delta(&tx, delta, tool_call_scrubber.as_mut());
            }
            flush_tool_call_scrubber(&tx, tool_call_scrubber.as_mut());
            let terminal = response
                .map(|response| model_response_with_tools(response, has_tools))
                .map(ModelStreamItem::Completed)
                .unwrap_or_else(|error| ModelStreamItem::Failed(map_error(error).to_string()));
            let _ = tx.send(terminal);
        }));
        let stream =
            futures::stream::unfold((rx, Some(handle)), |(mut receiver, handle)| async move {
                receiver.recv().await.map(|item| (item, (receiver, handle)))
            });
        Ok(ModelStream::new(Box::pin(stream)))
    }
}

/// Converts one [`ProviderDelta`] into a `ModelStreamItem::MessageDelta`,
/// running text through `tool_call_scrubber` (when the request has tools)
/// so streamed `<tool_call>` markup does not reach the live consumer.
fn forward_delta(
    sender: &tokio::sync::mpsc::UnboundedSender<ModelStreamItem>,
    delta: ProviderDelta,
    tool_call_scrubber: Option<&mut ToolCallStreamScrubber>,
) {
    let item = match delta {
        ProviderDelta::TextDelta { delta } => {
            let text = match tool_call_scrubber {
                Some(scrubber) => scrubber.feed(&delta),
                None => delta,
            };
            (!text.is_empty()).then(|| MessageDelta::text(text))
        }
        ProviderDelta::ThinkingDelta { delta } => Some(MessageDelta::reasoning(delta)),
    };
    if let Some(item) = item {
        let _ = sender.send(ModelStreamItem::MessageDelta(item));
    }
}

/// Emits any text the scrubber is still holding once the stream ends, so a
/// turn that finished mid-buffer does not silently drop trailing text.
fn flush_tool_call_scrubber(
    sender: &tokio::sync::mpsc::UnboundedSender<ModelStreamItem>,
    tool_call_scrubber: Option<&mut ToolCallStreamScrubber>,
) {
    let Some(scrubber) = tool_call_scrubber else {
        return;
    };
    let text = scrubber.flush();
    if !text.is_empty() {
        let _ = sender.send(ModelStreamItem::MessageDelta(MessageDelta::text(text)));
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "pipeline_test.rs"]
mod pipeline_test;
