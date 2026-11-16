//! The always-on tool that stands in for every packed tool.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyagents_harness::tool::{BoundToolCall, ToolCallBinding};
use tinytools::{PermissionLevel, Tool, ToolCallOptions, ToolResult, ToolRunContext};

use super::catalog::PackCatalog;
use super::handle::{PackRegistryHandle, ToolVec};
use super::render::{named_tool, render_pack};
use super::types::{PackTarget, PinnedPackCall, ResolvedPackCall, resolve_call};

/// Name of the disclosure-and-dispatch tool.
pub const USE_SKILL: &str = "use_skill";

/// The one always-on tool that stands in for every packed tool.
///
/// It is both halves of the pack seam. Called with a `skill` alone it renders
/// that pack's tool schemas into the conversation; called with a `skill` and a
/// `tool` it executes that tool. These were two tools — `load_skill` and
/// `use_skill` — until the pack index that each carried in its own description
/// made the pair spend 3.3 kB of every single turn saying one list twice, and
/// made the first call of any packed tool a mandatory two-call round trip.
/// A model that learned the retired name gets the harness's "unknown tool"
/// recovery result (#4249) and retries against the schema it can actually see,
/// so no alias is carried for it.
///
/// **Permission forwarding is load-bearing.** The harness gates a call on the
/// tool's `permission_level_with_args`, so a proxy reporting its own level would
/// launder every packed tool's risk down to this one's — a crypto write would
/// be admitted on a channel that refuses crypto writes. The argument-aware
/// accessor resolves the selected tool and defers to it. The argument-less
/// accessor reports the minimum `ReadOnly` operation: schema disclosure. This
/// keeps the dispatcher visible in a read-only session even when a pack also
/// contains mutators. A named target that cannot be resolved reports the
/// `Dangerous` ceiling, including after the call is pinned for admission.
pub struct UseSkillTool {
    handle: PackRegistryHandle,
    catalog: PackCatalog,
    description: String,
    binding: ToolCallBinding,
}

impl PackRegistryHandle {
    pub(crate) fn bind_call(&self, outer: Arc<dyn Tool>, args: &Value) -> Arc<PinnedPackCall> {
        let answer_permission = if named_tool(args).is_some() {
            PermissionLevel::Dangerous
        } else {
            outer.permission_level_with_args(args)
        };
        let answer = |result| PackTarget::Answer {
            result,
            permission: answer_permission,
        };
        let target = match self.catalog_for_binding() {
            Some(catalog) => match resolve_call(self, &catalog, args) {
                ResolvedPackCall::Execute {
                    tools,
                    index,
                    arguments,
                } => PackTarget::Execute {
                    tools,
                    index,
                    arguments,
                },
                ResolvedPackCall::Answer(result) => answer(result),
            },
            None => answer(ToolResult::error(
                "packed tool catalog is unavailable for this registration",
            )),
        };
        Arc::new(PinnedPackCall { outer, target })
    }
}

impl PinnedPackCall {
    pub(crate) fn selection(&self) -> Option<super::types::PackSelection> {
        match &self.target {
            PackTarget::Execute {
                tools,
                index,
                arguments,
            } => Some(super::types::PackSelection {
                tools: Arc::clone(tools),
                index: *index,
                arguments: arguments.clone(),
            }),
            PackTarget::Answer { .. } => None,
        }
    }

    fn selected(&self) -> Option<&dyn Tool> {
        match &self.target {
            PackTarget::Execute { tools, index, .. } => Some(tools[*index].as_ref()),
            PackTarget::Answer { .. } => None,
        }
    }

    fn selected_arguments(&self) -> Option<&Value> {
        match &self.target {
            PackTarget::Execute { arguments, .. } => Some(arguments),
            PackTarget::Answer { .. } => None,
        }
    }

    pub(crate) fn admission_target(&self) -> Option<(&dyn Tool, &Value)> {
        Some((self.selected()?, self.selected_arguments()?))
    }
}

impl BoundToolCall for PinnedPackCall {
    fn tool(&self) -> Arc<dyn Tool> {
        // The bound executor is this wrapper so the selected target remains pinned.
        Arc::new(self.clone())
    }

    fn admission_target(&self) -> Option<(&dyn Tool, &Value)> {
        PinnedPackCall::admission_target(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl Tool for PinnedPackCall {
    fn name(&self) -> &str {
        self.outer.name()
    }
    fn description(&self) -> &str {
        self.outer.description()
    }
    fn parameters_schema(&self) -> Value {
        self.outer.parameters_schema()
    }

    fn policy(&self) -> tinytools::ToolPolicy {
        let Some(selected) = self.selected() else {
            return self.outer.policy();
        };
        let outer_policy = self.outer.policy();
        let mut selected_policy = selected.policy();
        // Dispatch-level wrappers such as ApprovalRequiredToolSet apply
        // policy to the packed declaration. Keep those stricter admission
        // constraints when projecting the selected implementation's policy.
        selected_policy.classified |= outer_policy.classified;
        selected_policy.access.approval_required |= outer_policy.access.approval_required;
        selected_policy
    }

    async fn execute(&self, _args: Value) -> anyhow::Result<ToolResult> {
        self.execute_with_options(Value::Null, ToolCallOptions::default())
            .await
    }

    async fn execute_with_options(
        &self,
        _args: Value,
        options: ToolCallOptions,
    ) -> anyhow::Result<ToolResult> {
        if let PackTarget::Answer { result, .. } = &self.target {
            return Ok(result.clone());
        }
        self.selected()
            .expect("execute target has a tool")
            .execute_with_options(
                self.selected_arguments()
                    .expect("execute target has args")
                    .clone(),
                options,
            )
            .await
    }

    async fn prepare_authorization(&self, _args: &Value) -> anyhow::Result<Option<Value>> {
        let Some(tool) = self.selected() else {
            return Ok(None);
        };
        tool.prepare_authorization(self.selected_arguments().expect("execute target has args"))
            .await
    }

    async fn execute_with_context(
        &self,
        _args: Value,
        options: ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        if let PackTarget::Answer { result, .. } = &self.target {
            return Ok(result.clone());
        }
        self.selected()
            .expect("execute target has a tool")
            .execute_with_context(
                self.selected_arguments()
                    .expect("execute target has args")
                    .clone(),
                options,
                context,
            )
            .await
    }

    fn external_effect(&self) -> bool {
        self.selected().is_some_and(|tool| tool.external_effect())
    }

    fn supports_markdown(&self) -> bool {
        self.outer.supports_markdown()
    }
    fn external_effect_with_args(&self, _args: &Value) -> bool {
        match (self.selected(), self.selected_arguments()) {
            (Some(tool), Some(args)) => tool.external_effect_with_args(args),
            _ => false,
        }
    }
    fn timeout_policy(&self, _args: &Value) -> tinytools::ToolTimeout {
        match (self.selected(), self.selected_arguments()) {
            (Some(tool), Some(args)) => tool.timeout_policy(args),
            _ => tinytools::ToolTimeout::Inherit,
        }
    }
    fn permission_level(&self) -> PermissionLevel {
        self.selected()
            .map_or_else(|| self.outer.permission_level(), Tool::permission_level)
    }
    fn permission_level_with_args(&self, _args: &Value) -> PermissionLevel {
        match &self.target {
            PackTarget::Execute {
                tools,
                index,
                arguments,
            } => tools[*index].permission_level_with_args(arguments),
            PackTarget::Answer { permission, .. } => *permission,
        }
    }
}

impl UseSkillTool {
    pub fn new(handle: PackRegistryHandle, catalog: PackCatalog) -> Self {
        handle.register_catalog(catalog);
        let description = format!(
            "Reach a skill's tools. Their names, descriptions and argument schemas are NOT in \
             your context until you ask for them: call this with `skill` alone to see them, then \
             again with `skill` + `tool` + `args` to run one.\n\nSkills:\n{}",
            catalog.pack_index_markdown()
        );
        let binding_handle = handle.clone();
        let binding =
            ToolCallBinding::new(move |outer, args| binding_handle.bind_call(outer, args))
                .with_metadata(handle.clone());
        Self {
            handle,
            catalog,
            description,
            binding,
        }
    }

    fn skill_enum(&self) -> Vec<&'static str> {
        self.catalog.packs().iter().map(|p| p.id).collect()
    }

    fn resolve(&self, args: &Value) -> Option<(ToolVec, usize)> {
        let skill = args.get("skill").and_then(Value::as_str)?;
        let tool = named_tool(args)?;
        self.handle.resolve(&self.catalog, skill, tool)
    }
}

/// Recovers the shared registry handle from an erased registered `use_skill` tool.
///
/// The handle lets a host bind durable and synthesized registries and render
/// their filtered pack listings. Calls still pass through the generic
/// [`ToolCallBinding`] so admission and execution use the same pinned target.
///
/// Equivalent to [`PackRegistryHandle::of`].
pub fn pack_registry_handle(tool: &dyn Tool) -> Option<PackRegistryHandle> {
    PackRegistryHandle::of(tool)
}

impl PackRegistryHandle {
    /// The shared registry handle of an erased registered `use_skill` tool.
    ///
    /// [`UseSkillTool::new`] attaches the handle as metadata on the tool's
    /// [`ToolCallBinding`], which the tool exposes as its host extension; this
    /// reads it back from there. `None` for any tool whose binding carries no
    /// `PackRegistryHandle` (every tool other than `use_skill`).
    #[must_use]
    pub fn of(tool: &dyn Tool) -> Option<Self> {
        tool.host_extension()
            .and_then(|extension| extension.downcast_ref::<ToolCallBinding>())
            .and_then(|binding| binding.metadata::<Self>())
            .cloned()
    }
}

#[async_trait]
impl Tool for UseSkillTool {
    fn name(&self) -> &str {
        USE_SKILL
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "skill": { "type": "string", "enum": self.skill_enum(), "description": "Skill to read or run a tool from." },
                "tool": { "type": "string", "description": "Tool to run. Omit to list the skill's tools and their arguments instead." },
                "args": {
                    "type": "object",
                    "description": "The tool's own arguments, as documented in the listing.",
                    "additionalProperties": true
                }
            },
            "required": ["skill"]
        })
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
        let Some(skill) = args.get("skill").and_then(Value::as_str) else {
            return Ok(ToolResult::error(format!(
                "`skill` is required.\n\nSkills:\n{}",
                self.catalog.pack_index_markdown()
            )));
        };

        // Disclosure half: no tool named, so render the pack's schemas.
        let Some(name) = named_tool(&args) else {
            return Ok(match render_pack(&self.catalog, skill, &self.handle) {
                Ok(text) => ToolResult::success(text),
                Err(message) => ToolResult::error(message),
            });
        };

        let Some((tools, idx)) = self.handle.resolve(&self.catalog, skill, name) else {
            return Ok(ToolResult::error(format!(
                "{} No tool `{name}` in skill `{skill}`. Call `use_skill {{ \"skill\": \"{skill}\" }}` \
                 to see what it contains.\n\nSkills:\n{}",
                self.catalog.not_found_marker(),
                self.catalog.pack_index_markdown()
            )));
        };
        let inner_args = args.get("args").cloned().unwrap_or_else(|| json!({}));
        tracing::debug!(tool = tools[idx].name(), "[toolpacks] use_skill dispatch");
        tools[idx]
            .execute_with_context(inner_args, options, context)
            .await
    }

    async fn prepare_authorization(&self, args: &Value) -> anyhow::Result<Option<Value>> {
        let Some((tools, idx)) = self.resolve(args) else {
            return Ok(None);
        };
        let inner_args = args.get("args").cloned().unwrap_or_else(|| json!({}));
        tools[idx].prepare_authorization(&inner_args).await
    }

    fn supports_markdown(&self) -> bool {
        // The inner result is forwarded verbatim, markdown rendering included,
        // so advertise the capability rather than suppressing a real saving.
        true
    }

    fn external_effect_with_args(&self, args: &Value) -> bool {
        match self.resolve(args) {
            Some((tools, idx)) => {
                let inner_args = args.get("args").cloned().unwrap_or_else(|| json!({}));
                tools[idx].external_effect_with_args(&inner_args)
            }
            None => false,
        }
    }

    fn timeout_policy(&self, args: &Value) -> tinytools::ToolTimeout {
        match self.resolve(args) {
            Some((tools, idx)) => {
                let inner_args = args.get("args").cloned().unwrap_or_else(|| json!({}));
                tools[idx].timeout_policy(&inner_args)
            }
            None => tinytools::ToolTimeout::Inherit,
        }
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn permission_level_with_args(&self, args: &Value) -> PermissionLevel {
        // Naming no tool renders a schema and does nothing else. Reporting the
        // packed ceiling here would put an approval prompt in front of reading
        // a tool list, which is the round trip this tool exists to remove.
        if named_tool(args).is_none() {
            return PermissionLevel::ReadOnly;
        }
        match self.resolve(args) {
            Some((tools, idx)) => {
                let inner = args.get("args").cloned().unwrap_or_else(|| json!({}));
                tools[idx].permission_level_with_args(&inner)
            }
            // Unresolvable: the call will fail anyway, but report the ceiling so
            // a malformed call can never be admitted on a channel that would
            // have refused the real tool.
            None => PermissionLevel::Dangerous,
        }
    }

    /// The generic call-binding contract rides on the vocabulary's erased host
    /// extension. It captures the selected packed target once so admission,
    /// authorization, and execution use the same tool.
    fn host_extension(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        Some(&self.binding)
    }
}
