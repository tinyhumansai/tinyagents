//! Built-in prompt templates for the RLM driver model, and the placeholder
//! renderer.
//!
//! A template is a plain [`RlmTemplate`] document, so external harnesses can
//! ship their own as JSON and select them through
//! [`TemplateSpec::Inline`](super::types::TemplateSpec). The built-ins cover
//! the three recurring RLM shapes:
//!
//! - **`general`** — solve a task with code, calling capabilities as needed.
//! - **`context-explorer`** — the recursive-LM pattern from the RLM
//!   literature: a context too large to read at once is injected as the
//!   `context` variable, and the model probes it programmatically (slice,
//!   search, summarize slices via sub-LLM calls) instead of reading it all.
//! - **`orchestrator`** — decompose the task and delegate the pieces to
//!   registered sub-agents, then synthesize.

use super::host::CapabilityListing;
use super::types::{RlmPolicy, RlmTemplate, TemplateSpec};
use crate::error::{Result, TinyAgentsError};

/// Shared preamble describing the cell loop contract.
const LOOP_CONTRACT: &str = r#"You are operating a sandboxed code notebook. Each of your replies must contain
exactly one fenced code block (```{{language}} ... ```). The host executes it and
shows you the captured output, the cell's value, and any error; variables
persist between cells. Keep cells small and observe intermediate results
instead of writing one giant script. When you have the answer, call
final_answer("...") from code. Never fabricate outputs you have not observed.

{{usage}}

Available capabilities:
{{capabilities}}

Resource limits (exceeding them aborts the run):
{{limits}}"#;

/// The built-in `general` template.
pub fn general() -> RlmTemplate {
    RlmTemplate {
        name: "general".to_string(),
        system_prompt: format!(
            "{LOOP_CONTRACT}\n\nSolve the user's task with code. Use sub-LLM calls for fuzzy \
             subproblems (summarization, extraction, judgment) and plain code for exact ones \
             (counting, filtering, arithmetic)."
        ),
    }
}

/// The built-in `context-explorer` template (recursive-LM context probing).
pub fn context_explorer() -> RlmTemplate {
    RlmTemplate {
        name: "context-explorer".to_string(),
        system_prompt: format!(
            "{LOOP_CONTRACT}\n\nA variable named `context` holds material that is too large to \
             read in one glance. NEVER print all of it. Probe it programmatically: inspect its \
             length and structure first, then slice/search it, and delegate fuzzy analysis of \
             individual chunks to sub-LLM calls (`llm`). Combine the per-chunk findings with \
             code, then answer."
        ),
    }
}

/// The built-in `orchestrator` template (sub-agent delegation).
pub fn orchestrator() -> RlmTemplate {
    RlmTemplate {
        name: "orchestrator".to_string(),
        system_prompt: format!(
            "{LOOP_CONTRACT}\n\nYou are an orchestrator. Decompose the task into independent \
             pieces, delegate each to the most suitable registered agent with `agent(name, \
             input)`, inspect their replies, iterate if a piece came back weak, and synthesize \
             the final answer yourself."
        ),
    }
}

/// Resolves a [`TemplateSpec`] to a concrete template.
pub fn resolve(spec: &TemplateSpec) -> Result<RlmTemplate> {
    match spec {
        TemplateSpec::Inline(template) => Ok(template.clone()),
        TemplateSpec::Named(name) => match name.as_str() {
            "general" => Ok(general()),
            "context-explorer" => Ok(context_explorer()),
            "orchestrator" => Ok(orchestrator()),
            other => Err(TinyAgentsError::Validation(format!(
                "unknown rlm template `{other}` (built-ins: general, context-explorer, \
                 orchestrator)"
            ))),
        },
    }
}

/// Renders a template's system prompt, substituting the documented
/// placeholders.
pub fn render_system_prompt(
    template: &RlmTemplate,
    language: &str,
    usage: &str,
    capabilities: &CapabilityListing,
    policy: &RlmPolicy,
) -> String {
    template
        .system_prompt
        .replace("{{language}}", language)
        .replace("{{usage}}", usage)
        .replace("{{capabilities}}", &render_capabilities(capabilities))
        .replace("{{limits}}", &render_limits(policy))
}

fn render_capabilities(listing: &CapabilityListing) -> String {
    let mut out = String::new();
    if listing.models.is_empty() {
        out.push_str("- models: (none registered)\n");
    } else {
        out.push_str(&format!("- models: {}\n", listing.models.join(", ")));
    }
    if listing.tools.is_empty() {
        out.push_str("- tools: (none registered)\n");
    } else {
        out.push_str("- tools:\n");
        for (name, description) in &listing.tools {
            if description.is_empty() {
                out.push_str(&format!("    - {name}\n"));
            } else {
                out.push_str(&format!("    - {name}: {description}\n"));
            }
        }
    }
    if listing.agents.is_empty() {
        out.push_str("- agents: (none registered)");
    } else {
        out.push_str(&format!("- agents: {}", listing.agents.join(", ")));
    }
    out
}

fn render_limits(policy: &RlmPolicy) -> String {
    let timeout = policy
        .cell_timeout
        .map(|t| format!("{}s", t.as_secs()))
        .unwrap_or_else(|| "none".to_string());
    format!(
        "- max cells: {}\n- max sub-LLM calls: {}\n- max tool calls: {}\n- max agent calls: \
         {}\n- per-cell timeout: {timeout}",
        policy.max_cells, policy.max_llm_calls, policy.max_tool_calls, policy.max_agent_calls
    )
}
