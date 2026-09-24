//! [`GenerateImageTool`]: image generation as a harness tool.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyinference_image::{ImageGenerator, ImageRequest, MAX_IMAGES_PER_REQUEST};
use tinytools::{
    PermissionLevel, Tool, ToolCallOptions, ToolCategory, ToolPolicy, ToolResult, ToolRunContext,
    ToolTimeout,
};

use super::types::{MediaOutput, arg_i64, arg_list, arg_str, arg_u64, check_option_types};
use super::{artifact_stem, media_policy};

/// Default model-visible name.
pub const GENERATE_IMAGE_TOOL_NAME: &str = "generate_image";

/// Upper bound on one image call; slow models take ~90 s at high quality.
const IMAGE_TIMEOUT_MS: u64 = 300_000;

/// Generates images with an [`ImageGenerator`] and saves them to the run's
/// workspace.
///
/// The result lists the saved file paths. A billed call that produced nothing
/// is reported as an error that tells the model not to retry, because a retry
/// is a new, separately billed generation.
pub struct GenerateImageTool {
    generator: Arc<dyn ImageGenerator>,
    output: MediaOutput,
    name: String,
    description: String,
    permission: PermissionLevel,
    category: ToolCategory,
}

impl GenerateImageTool {
    /// Creates the tool with the default name and description.
    #[must_use]
    pub fn new(generator: Arc<dyn ImageGenerator>, output: MediaOutput) -> Self {
        Self {
            generator,
            output,
            name: GENERATE_IMAGE_TOOL_NAME.to_owned(),
            description: "Generate or edit images from a text prompt. Pass reference images \
                          (URLs or workspace file paths) to edit, restyle, or keep a subject \
                          consistent. Saves each image into the workspace and returns its path. \
                          Billed per call: do not call again after an error that says it was billed."
                .to_owned(),
            permission: PermissionLevel::Write,
            category: ToolCategory::System,
        }
    }

    /// Overrides the model-visible name (for hosts with an existing name).
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

    async fn run(&self, args: &Value, context: Option<&dyn ToolRunContext>) -> ToolResult {
        let workspace = context.and_then(ToolRunContext::workspace_root);
        let Some(prompt) = arg_str(args, &["prompt"]) else {
            return ToolResult::error("`prompt` is required");
        };
        if let Err(message) = check_option_types(args, &["n", "count"], &["seed"], &[]) {
            return ToolResult::error(message);
        }
        let mut request = ImageRequest::new(prompt);
        request.model = arg_str(args, &["model"]).map(str::to_owned);
        // Enforce the maximum image count at runtime
        if let Some(n) = arg_u64(args, &["n", "count"]) {
            if n > u64::from(MAX_IMAGES_PER_REQUEST) {
                return ToolResult::error(format!(
                    "image count {} exceeds maximum of {}",
                    n, MAX_IMAGES_PER_REQUEST
                ));
            }
            request.n = u32::try_from(n).ok();
        }
        request.size = arg_str(args, &["size"]).map(str::to_owned);
        request.resolution = arg_str(args, &["resolution"]).map(str::to_owned);
        request.aspect_ratio = arg_str(args, &["aspect_ratio", "aspectRatio"]).map(str::to_owned);
        request.quality = arg_str(args, &["quality"]).map(str::to_owned);
        request.output_format = arg_str(args, &["output_format", "format"]).map(str::to_owned);
        request.background = arg_str(args, &["background"]).map(str::to_owned);
        request.seed = arg_i64(args, &["seed"]);
        for raw in arg_list(
            args,
            &[
                "references",
                "reference_images",
                "input_images",
                "inputImages",
            ],
        ) {
            match self.output.reference(&raw, workspace) {
                Ok(reference) => request.references.push(reference),
                Err(message) => return ToolResult::error(message),
            }
        }
        let dir = match self.output.dir(workspace) {
            Ok(dir) => dir,
            Err(message) => return ToolResult::error(message),
        };

        let model_label = request
            .model
            .clone()
            .unwrap_or_else(|| self.generator.default_model().to_owned());
        tracing::info!(
            tool = %self.name,
            provider = self.generator.name(),
            model = %model_label,
            references = request.references.len(),
            "[media] generate_image"
        );
        let response = match self.generator.generate(request).await {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(tool = %self.name, %error, "[media] image generation failed");
                return ToolResult::error(format!("Image generation failed: {error}"));
            }
        };

        // Reject empty responses as billed failures, since the call was billed
        // but produced nothing to save.
        if response.images.is_empty() {
            return ToolResult::error(
                "Image generation succeeded and was billed, but returned no images. \
                 Do not generate again; report this error to the user."
                    .to_string(),
            );
        }

        let stem = artifact_stem("image");
        let mut artifacts = Vec::with_capacity(response.images.len());
        let mut lines = vec![format!(
            "Generated {} image(s) with {}:",
            response.images.len(),
            response.model
        )];
        for (index, image) in response.images.iter().enumerate() {
            // Preserve the generated image format in the artifact extension
            let ext = match image.media_type.split(';').next().map(str::trim) {
                Some("image/png") => "png",
                Some("image/jpeg") | Some("image/jpg") => "jpeg",
                Some("image/webp") => "webp",
                Some("image/gif") => "gif",
                other => {
                    return ToolResult::error(format!(
                        "Image generation succeeded and was billed, but image {index} had unsupported media type {}. \
                         Do not generate again; report this to the user.",
                        other.unwrap_or("<empty>")
                    ));
                }
            };
            match self
                .output
                .persist(&dir, &format!("{stem}-{index}"), ext, &image.data)
            {
                Ok(path) => {
                    lines.push(format!("- {}", path.display()));
                    artifacts.push(json!({
                        "type": "image",
                        "path": path.display().to_string(),
                        "media_type": image.media_type,
                        "bytes": image.data.len(),
                    }));
                }
                Err(error) => {
                    return ToolResult::error(format!(
                        "Image generation succeeded and was billed, but saving image {index} failed: \
                         {error}. Do not generate again; report this to the user."
                    ));
                }
            }
        }
        if let Some(cost) = response.cost_usd {
            lines.push(format!("Cost: ${cost:.4}"));
        }
        ToolResult::success_with_markdown(
            json!({
                "model": response.model,
                "cost_usd": response.cost_usd,
                "artifacts": artifacts,
            }),
            lines.join("\n"),
        )
    }
}

impl std::fmt::Debug for GenerateImageTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GenerateImageTool")
            .field("name", &self.name)
            .field("provider", &self.generator.name())
            .field("output", &self.output)
            .finish()
    }
}

#[async_trait]
impl Tool for GenerateImageTool {
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
                "prompt": { "type": "string", "description": "What to draw, or the edit to apply to the reference images." },
                "model": { "type": "string", "description": format!("Model id. Default: {}.", self.generator.default_model()) },
                "n": { "type": ["integer", "string"], "description": "Number of images, integer or numeric string (default 1)." },
                "aspect_ratio": { "type": "string", "description": "e.g. 1:1, 16:9, 9:16, 4:3, landscape, portrait, square, auto." },
                "resolution": { "type": "string", "description": "Resolution tier: 1K, 2K or 4K." },
                "size": { "type": "string", "description": "Exact pixels such as 1536x1024 (overrides resolution/aspect_ratio)." },
                "quality": { "type": "string", "enum": ["auto", "low", "medium", "high"] },
                "output_format": { "type": "string", "enum": ["png", "jpeg", "webp"] },
                "background": { "type": "string", "enum": ["auto", "transparent", "opaque"] },
                "seed": { "type": ["integer", "string"], "description": "Deterministic seed, integer or numeric string, where supported." },
                "references": {
                    "type": ["array", "string"],
                    "items": { "type": "string" },
                    "description": "Reference images: https URLs, data: URLs, or workspace file paths. Local paths are canonicalized and must remain inside the workspace, including after symlink resolution."
                }
            },
            "required": ["prompt"]
        })
    }

    fn policy(&self) -> ToolPolicy {
        media_policy(IMAGE_TIMEOUT_MS)
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

    fn timeout_policy(&self, _args: &Value) -> ToolTimeout {
        ToolTimeout::Millis(IMAGE_TIMEOUT_MS)
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
