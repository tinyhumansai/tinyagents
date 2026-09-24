//! [`GenerateVideoTool`]: asynchronous video generation as a harness tool.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyinference_video::{VideoGenerator, VideoRequest, WaitPolicy, wait_for_job};
use tinytools::{
    PermissionLevel, Tool, ToolCallOptions, ToolCategory, ToolPolicy, ToolResult, ToolRunContext,
    ToolTimeout,
};

use super::types::{
    MediaOutput, arg_bool, arg_i64, arg_list, arg_str, arg_u64, check_option_types,
};
use super::{artifact_stem, media_policy};

/// Default model-visible name.
pub const GENERATE_VIDEO_TOOL_NAME: &str = "generate_video";

/// Generates videos with a [`VideoGenerator`], waits for the job, and saves
/// the clips to the run's workspace.
///
/// A job that is still running when the wait budget runs out is reported with
/// its id; calling the tool again with `resume_job_id` collects it without
/// paying for a new generation.
pub struct GenerateVideoTool {
    generator: Arc<dyn VideoGenerator>,
    output: MediaOutput,
    wait: WaitPolicy,
    name: String,
    description: String,
    permission: PermissionLevel,
    category: ToolCategory,
}

impl GenerateVideoTool {
    /// Creates the tool with the default name, description and wait policy.
    #[must_use]
    pub fn new(generator: Arc<dyn VideoGenerator>, output: MediaOutput) -> Self {
        Self {
            generator,
            output,
            wait: WaitPolicy::default(),
            name: GENERATE_VIDEO_TOOL_NAME.to_owned(),
            description: "Generate a short video clip from a text prompt, optionally starting \
                          from (or ending on) an image, and save it into the workspace. Takes \
                          minutes. Billed per call: if a job times out, call again with \
                          `resume_job_id` instead of submitting a new one."
                .to_owned(),
            permission: PermissionLevel::Write,
            category: ToolCategory::System,
        }
    }

    /// Overrides the model-visible name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Overrides the model-visible description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Overrides the permission level the host gates the call on
    /// (default [`PermissionLevel::Write`]).
    #[must_use]
    pub fn with_permission_level(mut self, permission: PermissionLevel) -> Self {
        self.permission = permission;
        self
    }

    /// Overrides the tool's category (default [`ToolCategory::System`]).
    #[must_use]
    pub fn with_category(mut self, category: ToolCategory) -> Self {
        self.category = category;
        self
    }

    /// Overrides how long and how often the tool waits for a job.
    #[must_use]
    pub fn with_wait_policy(mut self, wait: WaitPolicy) -> Self {
        self.wait = wait;
        self
    }

    fn request(
        &self,
        args: &Value,
        workspace: Option<&std::path::Path>,
    ) -> Result<VideoRequest, String> {
        check_option_types(
            args,
            &["duration", "duration_seconds", "durationSeconds"],
            &["seed"],
            &["generate_audio", "audio"],
        )?;
        let prompt = arg_str(args, &["prompt"]).map(str::to_owned);
        let duration_s = match arg_u64(args, &["duration", "duration_seconds", "durationSeconds"]) {
            Some(duration) => Some(
                u32::try_from(duration)
                    .map_err(|_| format!("`duration` must not exceed {}", u32::MAX))?,
            ),
            None => None,
        };
        let mut request = VideoRequest {
            prompt: prompt.clone(),
            model: arg_str(args, &["model"]).map(str::to_owned),
            duration_s,
            resolution: arg_str(args, &["resolution"]).map(str::to_owned),
            aspect_ratio: arg_str(args, &["aspect_ratio", "aspectRatio"]).map(str::to_owned),
            size: arg_str(args, &["size"]).map(str::to_owned),
            generate_audio: arg_bool(args, &["generate_audio", "audio"]),
            seed: arg_i64(args, &["seed"]),
            ..VideoRequest::default()
        };
        if let Some(raw) = arg_str(args, &["first_frame", "inputImage", "input_image"]) {
            request.first_frame = Some(self.output.reference(raw, workspace)?);
        }
        if let Some(raw) = arg_str(args, &["last_frame"]) {
            request.last_frame = Some(self.output.reference(raw, workspace)?);
        }
        for raw in arg_list(args, &["references", "reference_images"]) {
            request
                .references
                .push(self.output.reference(&raw, workspace)?);
        }
        // Reject requests without a prompt or first frame: prompt is optional only with
        // first_frame, and first_frame is optional only with prompt.
        if prompt.is_none() && request.first_frame.is_none() {
            return Err(
                "either `prompt` or `first_frame` is required for video generation".to_string(),
            );
        }
        Ok(request)
    }

    async fn run(&self, args: &Value, context: Option<&dyn ToolRunContext>) -> ToolResult {
        let workspace = context.and_then(ToolRunContext::workspace_root);
        let dir = match self.output.dir(workspace) {
            Ok(dir) => dir,
            Err(message) => return ToolResult::error(message),
        };
        let outcome = if let Some(job_id) = arg_str(args, &["resume_job_id"]) {
            let model = arg_str(args, &["model"]).unwrap_or(self.generator.default_model());
            tracing::info!(tool = %self.name, job_id, "[media] resuming video job");
            wait_for_job(self.generator.as_ref(), job_id, model, &self.wait).await
        } else {
            let request = match self.request(args, workspace) {
                Ok(request) => request,
                Err(message) => return ToolResult::error(message),
            };
            tracing::info!(
                tool = %self.name,
                provider = self.generator.name(),
                model = request.model.as_deref().unwrap_or(self.generator.default_model()),
                duration_s = request.duration_s,
                first_frame = request.first_frame.is_some(),
                "[media] generate_video"
            );
            self.generator.generate(request, &self.wait).await
        };
        let response = match outcome {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(tool = %self.name, %error, job_id = ?error.job_id(), "[media] video generation failed");
                return ToolResult::error(format!("Video generation failed: {error}"));
            }
        };
        if response.videos.is_empty() {
            return ToolResult::error(format!(
                "Video job {} succeeded and was billed, but returned no videos. \
                 Do not generate again; report this error to the user.",
                response.job_id
            ));
        }

        let stem = artifact_stem("video");
        let mut artifacts = Vec::with_capacity(response.videos.len());
        let mut lines = vec![format!(
            "Generated {} video(s) with {} (job {}):",
            response.videos.len(),
            response.model,
            response.job_id
        )];
        for (index, video) in response.videos.iter().enumerate() {
            match self
                .output
                .persist(&dir, &format!("{stem}-{index}"), "mp4", &video.data)
            {
                Ok(path) => {
                    lines.push(format!("- {}", path.display()));
                    artifacts.push(json!({
                        "type": "video",
                        "path": path.display().to_string(),
                        "media_type": video.media_type,
                        "bytes": video.data.len(),
                    }));
                }
                Err(error) => {
                    return ToolResult::error(format!(
                        "Video job {} delivered and was billed, but saving clip {index} failed: \
                         {error}. Retry with resume_job_id={} to download it again.",
                        response.job_id, response.job_id
                    ));
                }
            }
        }
        if let Some(cost) = response.cost_usd {
            lines.push(format!("Cost: ${cost:.4}"));
        }
        ToolResult::success_with_markdown(
            json!({
                "job_id": response.job_id,
                "model": response.model,
                "cost_usd": response.cost_usd,
                "artifacts": artifacts,
            }),
            lines.join("\n"),
        )
    }
}

impl std::fmt::Debug for GenerateVideoTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GenerateVideoTool")
            .field("name", &self.name)
            .field("provider", &self.generator.name())
            .field("output", &self.output)
            .field("wait", &self.wait)
            .finish()
    }
}

#[async_trait]
impl Tool for GenerateVideoTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string", "description": "What happens in the clip. Optional only with first_frame." },
                "model": { "type": "string", "description": format!("Model id. Default: {}.", self.generator.default_model()) },
                "duration": { "type": ["integer", "string"], "description": "Seconds (integer or numeric string; the default model accepts 4-15)." },
                "resolution": { "type": "string", "description": "480p, 720p or 1080p (model-dependent)." },
                "aspect_ratio": { "type": "string", "description": "e.g. 16:9, 9:16, 1:1, landscape, portrait." },
                "generate_audio": { "type": ["boolean", "string"], "description": "Add an audio track (boolean or string \"true\"/\"false\"), where supported." },
                "seed": { "type": ["integer", "string"], "description": "Deterministic seed (integer or numeric string)." },
                "size": { "type": "string", "description": "Exact pixels such as 1280x720 (interchangeable with resolution + aspect_ratio)." },
                "first_frame": { "type": "string", "description": "Image to start from: https URL, data: URL or workspace path. Local paths are canonicalized and must remain inside the workspace, including after symlink resolution." },
                "last_frame": { "type": "string", "description": "Image to end on; local paths follow the first_frame workspace policy." },
                "references": {
                    "type": ["array", "string"],
                    "items": { "type": "string" },
                    "description": "Reference images/clips guiding subject or style. Local paths are canonicalized and must remain inside the workspace, including after symlink resolution."
                },
                "resume_job_id": { "type": "string", "description": "Collect an earlier job that timed out, without paying again." }
            }
        })
    }

    fn policy(&self) -> ToolPolicy {
        media_policy(u64::try_from(self.wait.timeout.as_millis()).unwrap_or(u64::MAX))
    }

    fn permission_level(&self) -> PermissionLevel {
        self.permission
    }

    fn permission_level_with_args(&self, _args: &Value) -> PermissionLevel {
        self.permission
    }

    fn category(&self) -> ToolCategory {
        self.category
    }

    fn external_effect(&self) -> bool {
        true
    }

    fn external_effect_with_args(&self, _args: &Value) -> bool {
        true
    }

    /// The job loop enforces its own [`WaitPolicy`] deadline, so the harness
    /// must not cut the call short.
    fn timeout_policy(&self, _args: &Value) -> ToolTimeout {
        ToolTimeout::Unbounded
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        Ok(self.run(&args, None).await)
    }

    async fn execute_with_context(
        &self,
        args: Value,
        _options: ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        Ok(self.run(&args, context).await)
    }
}
