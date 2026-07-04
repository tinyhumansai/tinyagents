//! Tool policy and selection middleware: allowlisting, classification-based
//! policy enforcement, dynamic/contextual tool selection, and human
//! approval.
//!
//! Split out of `library/mod.rs`; see that module's doc comment for the
//! full built-in middleware library overview.

use super::*;

// ── ToolAllowlistMiddleware ───────────────────────────────────────────────────

impl ToolAllowlistMiddleware {
    /// Creates an allowlist middleware permitting only the named tools.
    pub fn new(allowed: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            label: "tool_allowlist",
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns `true` if `name` is on the allowlist.
    pub fn allows(&self, name: &str) -> bool {
        self.allowed.contains(name)
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ToolAllowlistMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        if !self.allowed.contains(&call.name) {
            return Err(TinyAgentsError::Validation(format!(
                "tool `{}` is not on the allowlist",
                call.name
            )));
        }
        Ok(())
    }
}

// ── ToolPolicyMiddleware ──────────────────────────────────────────────────────

impl ToolPolicyMiddleware {
    /// Creates a policy middleware from a name→policy snapshot (typically
    /// [`ToolRegistry::policies`][crate::harness::tool::ToolRegistry::policies]).
    ///
    /// Defaults are permissive: nothing is required or denied until configured.
    /// Use [`strict`](Self::strict) for a fail-closed baseline.
    pub fn new(
        policies: std::collections::HashMap<String, crate::harness::tool::ToolPolicy>,
    ) -> Self {
        Self {
            label: "tool_policy",
            policies,
            require_classification: false,
            require_background_safe: false,
            deny: crate::harness::tool::ToolSideEffects::default(),
            require_sandbox: false,
            require_approval: false,
            approved: std::collections::HashSet::new(),
            enforce_result_bytes: false,
        }
    }

    /// Creates a fail-closed policy middleware: unclassified tools are rejected,
    /// and tools declaring `destructive` or `payment` side effects are denied.
    pub fn strict(
        policies: std::collections::HashMap<String, crate::harness::tool::ToolPolicy>,
    ) -> Self {
        Self {
            label: "tool_policy",
            policies,
            require_classification: true,
            require_background_safe: false,
            deny: crate::harness::tool::ToolSideEffects {
                destructive: true,
                payment: true,
                ..crate::harness::tool::ToolSideEffects::default()
            },
            require_sandbox: false,
            require_approval: false,
            approved: std::collections::HashSet::new(),
            enforce_result_bytes: false,
        }
    }

    /// Requires every tool to carry a classified policy (fail closed on
    /// unclassified or unknown tools).
    pub fn require_classification(mut self, require: bool) -> Self {
        self.require_classification = require;
        self
    }

    /// Requires every exposed/executed tool to be `background_safe`.
    pub fn require_background_safe(mut self, require: bool) -> Self {
        self.require_background_safe = require;
        self
    }

    /// Denies tools declaring any side effect present in `mask`.
    pub fn deny_side_effects(mut self, mask: crate::harness::tool::ToolSideEffects) -> Self {
        self.deny = mask;
        self
    }

    /// Enforces that a tool declaring
    /// [`SandboxMode::Required`][crate::harness::tool::SandboxMode::Required]
    /// only runs when the run carries a sandboxed workspace (fail closed
    /// otherwise). See [`RunContext::with_workspace`][crate::harness::context::RunContext::with_workspace].
    pub fn require_sandbox(mut self, require: bool) -> Self {
        self.require_sandbox = require;
        self
    }

    /// Blocks any tool declaring `approval_required` unless its name is in
    /// `approved`, turning the declarative approval flag into a fail-closed gate.
    pub fn require_approval(
        mut self,
        approved: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.require_approval = true;
        self.approved = approved.into_iter().map(Into::into).collect();
        self
    }

    /// Enforces each tool's declared `max_result_bytes` cap by truncating and
    /// flagging oversized results in `after_tool`.
    pub fn enforce_result_bytes(mut self, enforce: bool) -> Self {
        self.enforce_result_bytes = enforce;
        self
    }

    /// Returns `Ok(())` if the named tool is permitted, otherwise an explanation
    /// of why it is blocked. Used by both the exposure and execution hooks so a
    /// hidden tool cannot be executed by a divergent decision.
    fn evaluate(&self, name: &str) -> std::result::Result<(), String> {
        let Some(policy) = self.policies.get(name) else {
            if self.require_classification {
                return Err(format!("tool `{name}` has no declared policy"));
            }
            return Ok(());
        };
        if self.require_classification && !policy.classified {
            return Err(format!("tool `{name}` is unclassified"));
        }
        let s = &policy.side_effects;
        let d = &self.deny;
        let denied = (d.writes_files && s.writes_files)
            || (d.network && s.network)
            || (d.installs_dependencies && s.installs_dependencies)
            || (d.destructive && s.destructive)
            || (d.external_service && s.external_service)
            || (d.payment && s.payment);
        if denied {
            return Err(format!("tool `{name}` declares a denied side effect"));
        }
        if self.require_background_safe && !policy.access.background_safe {
            return Err(format!("tool `{name}` is not background-safe"));
        }
        if self.require_approval && policy.access.approval_required && !self.approved.contains(name)
        {
            return Err(format!(
                "tool `{name}` requires approval that was not granted"
            ));
        }
        Ok(())
    }

    /// The context-aware slice of policy enforcement: the sandbox requirement
    /// depends on the run's workspace, which `evaluate` (name-only) cannot see.
    fn evaluate_sandbox<Ctx>(
        &self,
        name: &str,
        ctx: &RunContext<Ctx>,
    ) -> std::result::Result<(), String> {
        if !self.require_sandbox {
            return Ok(());
        }
        let Some(policy) = self.policies.get(name) else {
            return Ok(());
        };
        if policy.runtime.sandbox != crate::harness::tool::SandboxMode::Required {
            return Ok(());
        }
        let sandboxed = ctx
            .workspace
            .as_ref()
            .is_some_and(|ws| ws.sandbox == crate::harness::tool::SandboxMode::Required);
        if sandboxed {
            Ok(())
        } else {
            Err(format!(
                "tool `{name}` requires a sandbox but the run has none"
            ))
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for ToolPolicyMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.tools.retain(|schema| {
            self.evaluate(&schema.name).is_ok() && self.evaluate_sandbox(&schema.name, ctx).is_ok()
        });
        Ok(())
    }

    async fn before_tool(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        self.evaluate(&call.name)
            .and_then(|_| self.evaluate_sandbox(&call.name, ctx))
            .map_err(TinyAgentsError::Validation)
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        result: &mut ToolResult,
    ) -> Result<()> {
        if !self.enforce_result_bytes {
            return Ok(());
        }
        if let Some(policy) = self.policies.get(&result.name)
            && let Some(limit) = policy.runtime.max_result_bytes
            && result.content.len() > limit
        {
            // Truncate on a char boundary at or below the byte limit so the
            // enforced payload is still valid UTF-8.
            let mut end = limit;
            while end > 0 && !result.content.is_char_boundary(end) {
                end -= 1;
            }
            result.content.truncate(end);
            let note = format!("tool result exceeded max_result_bytes ({limit}); truncated");
            result.error = Some(match result.error.take() {
                Some(existing) => format!("{existing}; {note}"),
                None => note,
            });
        }
        Ok(())
    }
}

// ── DynamicToolSelectionMiddleware ────────────────────────────────────────────

impl DynamicToolSelectionMiddleware {
    /// Creates a selection middleware exposing only tools for which `predicate`
    /// returns `true`.
    pub fn new(predicate: ToolPredicate) -> Self {
        Self {
            label: "dynamic_tool_selection",
            predicate,
        }
    }

    /// Creates a selection middleware exposing only the named tools.
    pub fn allowing(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let allowed: HashSet<String> = names.into_iter().map(Into::into).collect();
        Self::new(Arc::new(move |schema: &ToolSchema| {
            allowed.contains(&schema.name)
        }))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for DynamicToolSelectionMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        request.tools.retain(|schema| (self.predicate)(schema));
        Ok(())
    }
}

// ── ContextualToolSelectionMiddleware ─────────────────────────────────────────

impl ContextualToolSelectionMiddleware {
    /// Creates a selection middleware from a context-aware predicate.
    pub fn new(predicate: ContextualToolPredicate) -> Self {
        Self {
            label: "contextual_tool_selection",
            predicate,
        }
    }

    /// Builds a selection middleware from explicit allow/deny lists.
    ///
    /// Composition rules (fail-closed):
    /// - a tool named in `deny` is always hidden;
    /// - when `allow` is `Some`, a tool must be named in it to be exposed
    ///   (unknown tools are hidden);
    /// - when `allow` is `None`, everything not denied is exposed.
    pub fn from_lists(
        allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        deny: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let allow: Option<HashSet<String>> =
            allow.map(|names| names.into_iter().map(Into::into).collect());
        let deny: HashSet<String> = deny.into_iter().map(Into::into).collect();
        Self::from_resolved_lists(allow, deny)
    }

    /// Builds a selection middleware whose effective policy is a child allow/deny
    /// pair *composed with* an inherited parent policy, so a delegated sub-agent
    /// can only ever narrow — never widen — the tools its parent allowed.
    ///
    /// Inheritance rules:
    /// - **deny is additive**: the effective denylist is `parent_deny ∪ child_deny`
    ///   (a child cannot un-deny what the parent denied);
    /// - **allow is intersective**: if both parent and child restrict to an
    ///   allowlist, the effective allowlist is their intersection; if only one
    ///   restricts, that allowlist applies; if neither does, all-not-denied is
    ///   exposed.
    ///
    /// The result is fail-closed for the same reasons as [`Self::from_lists`].
    pub fn inheriting(
        parent_allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        parent_deny: impl IntoIterator<Item = impl Into<String>>,
        child_allow: Option<impl IntoIterator<Item = impl Into<String>>>,
        child_deny: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let parent_allow: Option<HashSet<String>> =
            parent_allow.map(|n| n.into_iter().map(Into::into).collect());
        let child_allow: Option<HashSet<String>> =
            child_allow.map(|n| n.into_iter().map(Into::into).collect());
        let allow = match (parent_allow, child_allow) {
            (Some(p), Some(c)) => Some(p.intersection(&c).cloned().collect()),
            (Some(p), None) => Some(p),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };
        let mut deny: HashSet<String> = parent_deny.into_iter().map(Into::into).collect();
        deny.extend(child_deny.into_iter().map(Into::into));
        Self::from_resolved_lists(allow, deny)
    }

    fn from_resolved_lists(allow: Option<HashSet<String>>, deny: HashSet<String>) -> Self {
        Self::new(Arc::new(move |schema: &ToolSchema, _ctx| {
            if deny.contains(&schema.name) {
                return false;
            }
            match &allow {
                Some(set) => set.contains(&schema.name),
                None => true,
            }
        }))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx>
    for ContextualToolSelectionMiddleware
{
    fn name(&self) -> &str {
        self.label
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> Result<()> {
        let selection = ToolSelectionContext {
            run_id: ctx.config.run_id.as_str().to_string(),
            depth: ctx.config.depth,
            tags: ctx.config.tags.clone(),
            requested_model: request.model.clone(),
        };
        let mut excluded = Vec::new();
        request.tools.retain(|schema| {
            let keep = (self.predicate)(schema, &selection);
            if !keep {
                excluded.push(schema.name.clone());
            }
            keep
        });
        // Make the exposure decision auditable when it actually withheld tools.
        if !excluded.is_empty() {
            ctx.emit(AgentEvent::ToolsFiltered {
                by: self.label.to_string(),
                excluded,
                remaining: request.tools.len(),
            });
        }
        Ok(())
    }
}

// ── HumanApprovalMiddleware ───────────────────────────────────────────────────

impl HumanApprovalMiddleware {
    /// Creates an approval middleware that interrupts when any flagged tool is
    /// called and no approval callback is configured.
    pub fn new(flagged: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            label: "human_approval",
            flagged: flagged.into_iter().map(Into::into).collect(),
            approve: None,
        }
    }

    /// Attaches an approval callback consulted for flagged tools. Returning
    /// `true` admits the call; `false` (or no callback) raises an interrupt.
    pub fn with_approval(mut self, approve: ApprovalFn) -> Self {
        self.approve = Some(approve);
        self
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for HumanApprovalMiddleware {
    fn name(&self) -> &str {
        self.label
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        call: &mut ToolCall,
    ) -> Result<()> {
        if self.flagged.contains(&call.name) {
            let approved = self
                .approve
                .as_ref()
                .map(|approve| approve(call))
                .unwrap_or(false);
            if !approved {
                return Err(TinyAgentsError::Interrupted {
                    node: "tool".to_string(),
                    message: format!("tool `{}` requires human approval", call.name),
                });
            }
        }
        Ok(())
    }
}
