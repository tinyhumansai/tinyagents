//! Graph-authoring implementations (`graph_define`, `graph_validate`,
//! `graph_compile`, `graph_diff`, `graph_register`) lowering through the
//! expressive-language compiler and capability resolver.
//!
//! Split out of `session/builtins/mod.rs`; see that module's doc comment
//! for the full built-in surface and the blocking-bridge design.

use super::*;

// ── Graph-authoring implementations ─────────────────────────────────────────

pub(super) fn graph_define_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    params: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let name =
        map_str(params, "name").ok_or_else(|| invalid(ctx, "graph_define: missing `name`"))?;
    let source =
        map_str(params, "source").ok_or_else(|| invalid(ctx, "graph_define: missing `source`"))?;

    // Check the limit up front (without consuming a slot) so a session that
    // has already hit the cap fails fast instead of paying for a parse and
    // compile it can't keep the result of anyway.
    if ctx.counters.lock().expect("counters poisoned").graph_def >= ctx.policy.max_graph_definitions
    {
        return Err(raise(
            ctx,
            TinyAgentsError::LimitExceeded(format!(
                "graph definition limit ({}) exceeded",
                ctx.policy.max_graph_definitions
            )),
        ));
    }
    if source.len() > ctx.policy.max_script_bytes {
        return Err(raise(
            ctx,
            TinyAgentsError::LimitExceeded(format!(
                "graph source is {} bytes, exceeding max_script_bytes ({})",
                source.len(),
                ctx.policy.max_script_bytes
            )),
        ));
    }

    let label = ctx
        .language
        .as_ref()
        .map(|l| l.provenance_label.clone())
        .unwrap_or_else(|| ctx.session_label.clone());
    let origin = Origin::generated_by(label);
    let program = parse_str(&source).map_err(|err| raise(ctx, err))?;
    let blueprints =
        compile_with_provenance(&program, origin.clone()).map_err(|err| raise(ctx, err))?;
    let blueprint = blueprints
        .into_iter()
        .find(|b| b.graph_id == name)
        .ok_or_else(|| {
            invalid(
                ctx,
                format!("graph_define: source has no graph named `{name}`"),
            )
        })?;

    // The draft is about to be recorded successfully; consume a slot now
    // (re-checking the limit under the same lock to guard against a
    // concurrent `graph_define` racing between the check above and here).
    {
        let mut counters = ctx.counters.lock().expect("counters poisoned");
        if counters.graph_def >= ctx.policy.max_graph_definitions {
            return Err(raise(
                ctx,
                TinyAgentsError::LimitExceeded(format!(
                    "graph definition limit ({}) exceeded",
                    ctx.policy.max_graph_definitions
                )),
            ));
        }
        counters.graph_def += 1;
    }

    let handle = GraphBlueprintHandle {
        name: blueprint.graph_id.clone(),
        source,
        blueprint: blueprint.clone(),
        origin,
        compiled: false,
        requires_review: ctx.policy.generated_graphs_require_review,
    };
    ctx.drafts
        .lock()
        .expect("drafts poisoned")
        .insert(handle.name.clone(), handle.clone());
    record(
        ctx,
        new_call_id(),
        ReplCallKind::Graph,
        "graph_define",
        json!({ "name": handle.name }),
        Duration::default(),
    );
    Ok(draft_descriptor(&handle))
}

/// Builds the script-visible descriptor map for a graph draft (carrying its
/// name, node count, and compile/review status). The opaque
/// [`GraphBlueprintHandle`] itself lives host-side in `ctx.drafts`.
fn draft_descriptor(handle: &GraphBlueprintHandle) -> Dynamic {
    let mut map = Map::new();
    map.insert("name".into(), Dynamic::from(handle.name.clone()));
    map.insert(
        "nodes".into(),
        Dynamic::from(handle.blueprint.nodes.len() as i64),
    );
    map.insert("compiled".into(), Dynamic::from(handle.compiled));
    map.insert(
        "requires_review".into(),
        Dynamic::from(handle.requires_review),
    );
    Dynamic::from_map(map)
}

/// Looks up a graph draft by the `name` field of a descriptor map.
pub(super) fn lookup_draft<State: Send + Sync>(
    ctx: &HostContext<State>,
    descriptor: &Map,
    func: &str,
) -> Result<GraphBlueprintHandle, Box<EvalAltResult>> {
    let name = map_str(descriptor, "name")
        .ok_or_else(|| invalid(ctx, format!("{func}: descriptor is missing `name`")))?;
    ctx.drafts
        .lock()
        .expect("drafts poisoned")
        .get(&name)
        .cloned()
        .ok_or_else(|| invalid(ctx, format!("{func}: no graph draft named `{name}`")))
}

pub(super) fn graph_validate_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    descriptor: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let handle = lookup_draft(ctx, descriptor, "graph_validate")?;
    let program = parse_str(&handle.source).map_err(|err| raise(ctx, err))?;
    let diagnostics = Resolver::from_registry(&*ctx.registry).resolve_program(&program);
    let array: Array = diagnostics
        .iter()
        .map(|d| Dynamic::from(d.message.clone()))
        .collect();
    Ok(Dynamic::from_array(array))
}

pub(super) fn graph_compile_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    descriptor: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let mut handle = lookup_draft(ctx, descriptor, "graph_compile")?;
    // Bind the blueprint through the same resolver gate file-backed `.rag`
    // source passes — generated topology is never trusted blindly.
    Resolver::from_registry(&*ctx.registry)
        .resolve_blueprint(&handle.blueprint)
        .map_err(|err| raise(ctx, err))?;
    handle.compiled = true;
    handle.requires_review = ctx.policy.generated_graphs_require_review;
    ctx.drafts
        .lock()
        .expect("drafts poisoned")
        .insert(handle.name.clone(), handle.clone());
    record(
        ctx,
        new_call_id(),
        ReplCallKind::Graph,
        "graph_compile",
        json!({ "name": handle.name, "requires_review": handle.requires_review }),
        Duration::default(),
    );
    Ok(draft_descriptor(&handle))
}

pub(super) fn graph_diff_handles<State: Send + Sync>(
    ctx: &HostContext<State>,
    old: &Blueprint,
    new: &Blueprint,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let diff = blueprint_diff(old, new);
    let value = serde_json::to_value(&diff)
        .map_err(|err| raise(ctx, TinyAgentsError::Validation(err.to_string())))?;
    Ok(repl_value_to_dynamic(&json_to_repl_value(&value)))
}

pub(super) fn graph_register_impl<State: Send + Sync + 'static>(
    ctx: &HostContext<State>,
    params: &Map,
) -> Result<Dynamic, Box<EvalAltResult>> {
    let graph = params
        .get("graph")
        .and_then(|d| d.read_lock::<Map>().map(|m| m.clone()))
        .ok_or_else(|| {
            invalid(
                ctx,
                "graph_register: `graph` must be a compiled graph descriptor",
            )
        })?;
    let handle = lookup_draft(ctx, &graph, "graph_register")?;
    if !handle.compiled {
        return Err(raise(
            ctx,
            TinyAgentsError::Validation(
                "graph_register: graph must be compiled via graph_compile first".to_string(),
            ),
        ));
    }
    let review_id = map_str(params, "review_id").filter(|s| !s.is_empty());
    if handle.requires_review && review_id.is_none() {
        return Err(raise(
            ctx,
            TinyAgentsError::Validation(format!(
                "graph_register: generated graph `{}` requires review (no review_id)",
                handle.name
            )),
        ));
    }
    // Enforce the review gate and emit a registry intent. The compiled topology
    // is handed to the host for installation through the registry resolver —
    // the REPL never installs generated topology directly.
    record(
        ctx,
        new_call_id(),
        ReplCallKind::Graph,
        "graph_register",
        json!({ "name": handle.name, "review_id": review_id }),
        Duration::default(),
    );
    Ok(Dynamic::from(handle.name))
}
