use super::{ArtifactFiling, GeneratedArtifact, MediaArtifactKind};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::Duration;
use std::{path::PathBuf, sync::Arc};
use tinytools::{
    PermissionLevel, Tool, ToolCallOptions, ToolCategory, ToolContent, ToolPolicy, ToolProgress,
    ToolResult, ToolRunContext, ToolScope, ToolTimeout,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MediaJobProgress {
    pub id: String,
    pub state: String,
    pub outputs: usize,
    pub output_indices: Vec<usize>,
    pub cost_usd: Option<f64>,
    pub error: Option<String>,
}
pub type MediaProgressFn = Arc<dyn Fn(&MediaJobProgress) + Send + Sync>;
pub type MediaInnerFactory<T> =
    Arc<dyn for<'a> Fn(MediaProgressFn, Option<&'a dyn ToolRunContext>) -> T + Send + Sync>;
/// Host-supplied extraction of its provider-assigned call identifier.
pub type ToolCallIdExtractor =
    Arc<dyn for<'a> Fn(Option<&'a dyn ToolRunContext>) -> Option<String> + Send + Sync>;

fn canonical_tool_call_id(context: Option<&dyn ToolRunContext>) -> Option<String> {
    tinyagents_harness::tool::ToolExecutionContext::call_id_of(context)
}

fn progress_label(kind: &MediaArtifactKind) -> &'static str {
    match kind {
        MediaArtifactKind::Image => "Generating image",
        MediaArtifactKind::Video => "Generating video",
    }
}

fn heartbeat_message(label: &str, elapsed: Duration, job_state: Option<&str>) -> String {
    let seconds = elapsed.as_secs();
    let elapsed = if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    };
    match job_state {
        Some(state) => format!("{label} — {elapsed} elapsed (job {state})"),
        None => format!("{label} — {elapsed} elapsed"),
    }
}

pub struct MediaArtifactTool<T: Tool> {
    inner: T,
    kind: MediaArtifactKind,
    per_call: Option<MediaInnerFactory<T>>,
    call_id: Option<ToolCallIdExtractor>,
    filing: Arc<dyn ArtifactFiling>,
}
impl<T: Tool> MediaArtifactTool<T> {
    #[must_use]
    pub fn new(inner: T, kind: MediaArtifactKind, filing: Arc<dyn ArtifactFiling>) -> Self {
        Self {
            inner,
            kind,
            per_call: None,
            call_id: None,
            filing,
        }
    }
    #[must_use]
    pub fn with_per_call_inner(mut self, factory: MediaInnerFactory<T>) -> Self {
        self.per_call = Some(factory);
        self
    }
    /// Let a host extract its call id from its own erased context extension.
    #[must_use]
    pub fn with_tool_call_id(mut self, extract: ToolCallIdExtractor) -> Self {
        self.call_id = Some(extract);
        self
    }
    fn title(args: &Value, index: usize, total: usize) -> String {
        let base = args
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.chars().take(60).collect())
            .unwrap_or_else(|| "Generated media".into());
        if total > 1 {
            format!("{base} ({}/{total})", index + 1)
        } else {
            base
        }
    }

    async fn run_with_heartbeat<F: std::future::Future<Output = anyhow::Result<ToolResult>>>(
        &self,
        context: Option<&dyn ToolRunContext>,
        status: Arc<std::sync::Mutex<Option<MediaJobProgress>>>,
        run: F,
    ) -> anyhow::Result<ToolResult> {
        let Some(context) = context else {
            return run.await;
        };
        let label = progress_label(&self.kind);
        let start = tokio::time::Instant::now();
        let mut run = Box::pin(run);
        let mut ticker = tokio::time::interval_at(
            tokio::time::Instant::now() + Duration::from_secs(15),
            Duration::from_secs(15),
        );
        loop {
            tokio::select! {
                result=&mut run => return result,
                _=ticker.tick()=>{
                    let state=status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
                    let message=heartbeat_message(label,start.elapsed(),state.as_ref().map(|state|state.state.as_str()));
                    context.report_progress(ToolProgress::message(message));
                }
            }
        }
    }
    async fn file_entries(
        &self,
        mut result: ToolResult,
        args: &Value,
        context: Option<&dyn ToolRunContext>,
        call_id: Option<&str>,
    ) -> ToolResult {
        let Some(data) = result.content.iter_mut().find_map(|block| {
            if let ToolContent::Json { data } = block {
                Some(data)
            } else {
                None
            }
        }) else {
            return result;
        };
        let Some(entries) = data.get_mut("artifacts").and_then(Value::as_array_mut) else {
            return result;
        };
        let total = entries.len();
        let mut replacements = Vec::new();
        for (index, entry) in entries.iter_mut().enumerate() {
            let Some(path) = entry.get("path").and_then(Value::as_str).map(PathBuf::from) else {
                continue;
            };
            let generated = GeneratedArtifact {
                path: path.clone(),
                artifact_kind: self.kind.clone(),
                kind: entry
                    .get("type")
                    .or_else(|| entry.get("kind"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                media_type: entry
                    .get("media_type")
                    .or_else(|| entry.get("format"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                bytes: entry.get("bytes").and_then(Value::as_u64),
                title: Self::title(args, index, total),
                tool_call_id: call_id.map(ToOwned::to_owned),
            };
            match self.filing.file(generated, context).await {
                Ok(filed) => {
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert("path".into(), json!(filed.path));
                        obj.insert("artifact_id".into(), json!(filed.id));
                    }
                    replacements.push((path.to_string_lossy().into_owned(), filed.path));
                }
                Err(error) => {
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert("artifact_error".into(), json!(error));
                    }
                }
            }
        }
        if let Some(markdown) = result.markdown_formatted.as_mut() {
            for (source, destination) in replacements {
                *markdown = markdown.replace(&source, &destination);
            }
        }
        result
    }
}

#[async_trait]
impl<T: Tool> Tool for MediaArtifactTool<T> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn parameters_schema(&self) -> Value {
        self.inner.parameters_schema()
    }
    fn policy(&self) -> ToolPolicy {
        self.inner.policy()
    }
    fn permission_level(&self) -> PermissionLevel {
        self.inner.permission_level()
    }
    fn permission_level_with_args(&self, a: &Value) -> PermissionLevel {
        self.inner.permission_level_with_args(a)
    }
    fn scope(&self) -> ToolScope {
        self.inner.scope()
    }
    fn category(&self) -> ToolCategory {
        self.inner.category()
    }
    fn external_effect(&self) -> bool {
        self.inner.external_effect()
    }
    fn external_effect_with_args(&self, a: &Value) -> bool {
        self.inner.external_effect_with_args(a)
    }
    fn timeout_policy(&self, a: &Value) -> ToolTimeout {
        self.inner.timeout_policy(a)
    }
    fn supports_markdown(&self) -> bool {
        self.inner.supports_markdown()
    }
    fn max_result_size_chars(&self) -> Option<usize> {
        self.inner.max_result_size_chars()
    }
    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        self.execute_with_context(args, ToolCallOptions::default(), None)
            .await
    }
    async fn execute_with_options(
        &self,
        args: Value,
        options: ToolCallOptions,
    ) -> anyhow::Result<ToolResult> {
        self.execute_with_context(args, options, None).await
    }
    async fn execute_with_context(
        &self,
        args: Value,
        options: ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let status = Arc::new(std::sync::Mutex::new(None));
        let callback_status = Arc::clone(&status);
        let progress: MediaProgressFn = Arc::new(move |update| {
            *callback_status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(update.clone());
        });
        let inner = self
            .per_call
            .as_ref()
            .map(|factory| factory(progress, context));
        let run = async {
            match inner {
                Some(tool) => {
                    tool.execute_with_context(args.clone(), options, context)
                        .await
                }
                None => {
                    self.inner
                        .execute_with_context(args.clone(), options, context)
                        .await
                }
            }
        };
        let result = self.run_with_heartbeat(context, status, run).await?;
        if result.is_error {
            return Ok(result);
        }
        let call_id = self
            .call_id
            .as_ref()
            .and_then(|extract| extract(context))
            .or_else(|| canonical_tool_call_id(context));
        Ok(self
            .file_entries(result, &args, context, call_id.as_deref())
            .await)
    }
}

#[cfg(test)]
#[path = "artifact_tool_tests.rs"]
mod tests;
