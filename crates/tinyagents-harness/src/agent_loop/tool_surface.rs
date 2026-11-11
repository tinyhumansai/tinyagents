//! The run's tool surface: which tool schemas the loop advertises, and how that
//! set changes while the run is in flight.
//!
//! Split out of `run_loop.rs`. [`ToolSurface`] is built once at the top of a
//! run ([`AgentHarness::build_tool_surface`]) and then adjusted at the start of
//! every turn: a toolset chain whose live set changed is diffed and declared as
//! a transcript patch ([`ToolSurface::declare_toolset_changes`]), tools a
//! successful `tool_search` discovered are promoted
//! ([`ToolSurface::promote_discovered`]), and the wire list for the turn is
//! assembled ([`ToolSurface::assemble_turn_schemas`]).

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::tool_changes;
use super::*;
use crate::tool::ToolGate;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// The direct tool schemas: the registry's `Direct` schemas filtered by the
    /// host allow-list, extended by this run's toolset chain (B3), name-sorted,
    /// then provider-projected.
    ///
    /// Additive: a name the registry already advertises keeps the registry's
    /// declaration, so a registered tool always wins a collision. The sort is
    /// what every consumer (and the provider prompt cache) relies on for
    /// wire-byte stability.
    async fn direct_tool_schemas(
        &self,
        ctx: &RunContext<Ctx>,
        gate: &ToolGate,
    ) -> Result<Vec<ToolSchema>> {
        let mut schemas = self
            .tools
            .schemas()
            .into_iter()
            .filter(|schema| {
                self.tools.get(&schema.name).is_some_and(|tool| {
                    gate.exposure(tool.as_ref()) == tinytools::ToolExposure::Direct
                }) && gate.lists(
                    &schema.name,
                    self.tools.get(&schema.name).as_deref(),
                    tinytools::Surface::Catalog,
                )
            })
            .collect::<Vec<_>>();
        if let Some(toolset) = &self.toolset {
            // Every registered name, listed or not: a registered tool owns its
            // name even when the rules keep it off this catalogue, so a
            // toolset tool may never take its place on the wire.
            let registered = self.tools.names();
            let existing: HashSet<&str> = registered.iter().map(String::as_str).collect();
            let extra: Vec<_> = toolset
                .tools(ctx)
                .await?
                .into_iter()
                .filter(|tool| gate.exposure(tool.as_ref()) == tinytools::ToolExposure::Direct)
                .filter(|tool| {
                    gate.lists(
                        tool.name(),
                        Some(tool.as_ref()),
                        tinytools::Surface::Catalog,
                    )
                })
                .filter(|tool| !existing.contains(tool.name()))
                .map(|tool| crate::tool::provider_schema(tool.as_ref()))
                .collect();
            schemas.extend(extra);
            schemas.sort_by(|left, right| left.name.cmp(&right.name));
        }
        // Provider projection applies once, to the full combined set
        // (registry + toolset), so a toolset-supplied schema reaches the wire
        // cleaned exactly like a registered one.
        if let Some(preparation) = &self.policy.tool_schemas {
            schemas = crate::tool::prepare_tool_schemas(&schemas, preparation);
        }
        Ok(schemas)
    }

    /// Builds the run's tool surface once, at the top of the run.
    ///
    /// Only *direct* tools go on the initial wire request. Deferred tools are
    /// indexed into the run's catalogue and reached through the `tool_search`
    /// bridge; search matches are promoted on the next request. The same host
    /// allow-list gates both halves: deferral only ever subtracts from what the
    /// host admitted.
    pub(super) async fn build_tool_surface(
        &self,
        ctx: &RunContext<Ctx>,
        messages: &[Message],
        gate: &ToolGate,
    ) -> Result<ToolSurface> {
        let mut tool_schemas = self.direct_tool_schemas(ctx, gate).await?;
        // Captured before the bridge schemas are appended below, so
        // `ToolsAdvertised.direct` reports the actual `Direct`-exposure count.
        let direct_schema_count = tool_schemas.len();
        let direct_tool_schemas = tool_schemas.clone();
        let mut bridge_schemas: Vec<ToolSchema> = Vec::new();
        let deferred_catalog = self.deferred_catalog(gate);
        // A resumed transcript carries promoted declarations in SystemMessage
        // patches. Only restore names still admitted into this run's catalogue.
        let promoted_schemas: BTreeMap<String, ToolSchema> =
            tinyinference_llm::message::replay_system_state(messages)
                .1
                .into_iter()
                .filter(|schema| deferred_catalog.get(&schema.name).is_some())
                .map(|schema| (schema.name.clone(), schema))
                .collect();
        let promoted_names: BTreeSet<String> = promoted_schemas.keys().cloned().collect();
        let recorded_promotions = promoted_names.clone();
        // Listed only when a call to it would be answered: hidden or denied
        // on the catalogue, or approval-gated (the bridge is answered in place
        // and cannot be deferred to an approver), keeps it off the wire.
        let search = crate::tool::discover::TOOL_SEARCH_NAME;
        let answerable = |surface| {
            matches!(
                gate.admits_intrinsic(search, surface),
                crate::tool::CallGate::Admit(
                    tinytools::ApprovalDirective::Default | tinytools::ApprovalDirective::Waived
                )
            )
        };
        let bridge_listed =
            answerable(tinytools::Surface::Catalog) && answerable(tinytools::Surface::Call);
        if bridge_listed && !deferred_catalog.is_empty() {
            // A host-registered `tool_search` keeps its slot: the intrinsic
            // bridge only fills a name nobody registered. Check the full
            // registry (`self.tools.dispatch`), not just the direct set — a
            // `Hidden` or `Deferred` registration under that name must also
            // suppress the intrinsic schema, because admission's own collision
            // rule (`answer_discovery_bridge`) checks the same full registry.
            // A toolset tool under that name suppresses it too.
            let mut bridge: Vec<_> =
                crate::tool::discover::bridge_schemas(&deferred_catalog, &self.policy.discovery)
                    .into_iter()
                    .collect();
            if let Some(preparation) = &self.policy.tool_schemas {
                // The bridge schemas are generated here, after the direct set
                // was prepared, so they need the same provider projection (for
                // example Gemini's `minimum`/`maximum` removal) applied
                // individually or they reach the wire raw.
                bridge = bridge
                    .into_iter()
                    .map(|schema| crate::tool::prepare_tool_schema(&schema, preparation))
                    .collect();
            }
            for schema in bridge {
                // The direct set also holds toolset-supplied tools, which
                // the registry check alone misses.
                if self.tools.dispatch(&schema.name).is_none()
                    && !tool_schemas.iter().any(|direct| direct.name == schema.name)
                {
                    tool_schemas.push(schema.clone());
                    bridge_schemas.push(schema);
                }
            }
        }
        Ok(ToolSurface {
            tool_schemas,
            direct_schema_count,
            direct_tool_schemas,
            declared_tool_schemas: Vec::new(),
            bridge_schemas,
            deferred_catalog,
            promoted_schemas,
            promoted_names,
            recorded_promotions,
        })
    }

    /// Fails closed on a structured-output schema whose name collides with a
    /// registered tool *or* the intrinsic discovery bridge.
    ///
    /// Under the tool-call strategy the schema is sent as an extra `function`
    /// entry, so a collision puts two identically-named functions in one
    /// request — which OpenAI rejects outright — and makes "was this the schema
    /// or the real tool?" unanswerable for every returned call. Two checks,
    /// because neither alone covers every name that ends up on the wire:
    /// `self.tools.names()` covers every registered tool (Direct, Deferred,
    /// Hidden) but not the intrinsic bridge; `tool_schemas` covers the bridge
    /// (and the Direct set) but never contains a Deferred tool's own name.
    pub(super) fn check_structured_schema_name(&self, tool_schemas: &[ToolSchema]) -> Result<()> {
        if let Some(name) = self
            .policy
            .default_response_format
            .as_ref()
            .and_then(|format| match format {
                ResponseFormat::Auto { name, .. } | ResponseFormat::JsonSchema { name, .. } => {
                    Some(name)
                }
                _ => None,
            })
            && (self
                .tools
                .names()
                .iter()
                .any(|registered| registered == name)
                || tool_schemas.iter().any(|schema| &schema.name == name))
        {
            return Err(TinyAgentsError::Validation(format!(
                "structured-output schema name `{name}` collides with a registered tool (or the \
                 intrinsic discovery bridge) of the same name; rename one of them"
            )));
        }
        Ok(())
    }
}

impl ToolSurface {
    /// B6 (`docs/runtime-comparison/plan.md`, `declare_tool_changes`):
    /// re-consult the toolset chain (documented as "called once per turn",
    /// `ToolSet::tools`) and diff its live set against what this transcript has
    /// declared so far. A caller whose toolset never varies turn to turn sees no
    /// diff and pays nothing here — this only fires for a genuine mid-run
    /// change. Runs before the request/`ModelStarted`, so the patch (if any) is
    /// part of *this* turn's request.
    pub(super) async fn declare_toolset_changes<State: Send + Sync, Ctx: Send + Sync>(
        &mut self,
        harness: &AgentHarness<State, Ctx>,
        ctx: &RunContext<Ctx>,
        messages: &mut Vec<Message>,
        gate: &ToolGate,
        patch_profile: Option<&tinyinference_llm::model::ModelProfile>,
    ) -> Result<bool> {
        if harness.toolset.is_none() {
            return Ok(false);
        }
        let live_schemas = harness.direct_tool_schemas(ctx, gate).await?;
        let mut rewrote = false;
        if let Some(patch) = tool_changes::diff_tool_set(&self.declared_tool_schemas, &live_schemas)
        {
            let in_place = tool_changes::patch_inserts_in_place(patch_profile);
            tool_changes::apply_tool_change_patch(messages, patch, in_place);
            // A mid-conversation patch is an ordinary append; the folded or
            // front-inserted form rewrites the transcript in place.
            rewrote = !in_place;
            self.declared_tool_schemas = live_schemas.clone();
            self.direct_tool_schemas = live_schemas;
        }
        Ok(rewrote)
    }

    /// Promotes only names returned by a successful intrinsic search. The patch
    /// makes the declaration recoverable from the transcript; the provider
    /// receives its typed schema on this and later calls.
    pub(super) fn promote_discovered(
        &mut self,
        messages: &mut Vec<Message>,
        patch_profile: Option<&tinyinference_llm::model::ModelProfile>,
    ) -> bool {
        let newly_promoted: Vec<ToolSchema> = self
            .promoted_names
            .difference(&self.recorded_promotions)
            .filter_map(|name| self.deferred_catalog.get(name).cloned())
            .collect();
        if newly_promoted.is_empty() {
            return false;
        }
        let mut rewrote = false;
        if let Some(patch) = tool_changes::diff_tool_set(&[], &newly_promoted) {
            let in_place = tool_changes::patch_inserts_in_place(patch_profile);
            tool_changes::apply_tool_change_patch(messages, patch, in_place);
            rewrote = !in_place;
        }
        self.recorded_promotions
            .extend(newly_promoted.iter().map(|schema| schema.name.clone()));
        self.promoted_schemas.extend(
            newly_promoted
                .into_iter()
                .map(|schema| (schema.name.clone(), schema)),
        );
        rewrote
    }

    /// Rebuilds the turn's wire list: the direct set, then promoted tools not
    /// already direct, then the bridge schemas.
    pub(super) fn assemble_turn_schemas(&mut self) {
        self.tool_schemas = self.direct_tool_schemas.clone();
        self.tool_schemas.extend(
            self.promoted_schemas
                .values()
                .filter(|schema| {
                    !self
                        .direct_tool_schemas
                        .iter()
                        .any(|direct| direct.name == schema.name)
                })
                .cloned(),
        );
        self.tool_schemas.extend(self.bridge_schemas.clone());
    }
}
