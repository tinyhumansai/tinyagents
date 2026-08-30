//! Injected (hidden) tool arguments: values the *host* supplies, never the
//! model.
//!
//! # What this is for
//!
//! Some tool arguments are not the model's business: the caller's thread id,
//! the recursion depth, the id of the tool call being answered, a database
//! handle, an authenticated user. Today a tool that needs them reaches around
//! the argument schema entirely — `SubAgentTool` reads `context.thread_id` and
//! `context.depth` off [`ToolExecutionContext`][super::ToolExecutionContext]
//! because there is no declarative way to *receive* them — which means every
//! such tool re-invents the plumbing and none of it is visible in the tool's
//! declared shape.
//!
//! LangChain models this as `InjectedToolArg` / `InjectedToolCallId`
//! annotations, strips them from the model-facing `tool_call_schema`, and
//! LangGraph's `ToolNode` re-injects the real values at execution time.
//!
//! # The ordering rule (security-critical)
//!
//! At execution time the sequence must be, in this order:
//!
//! 1. **Strip** every injected key from the model-supplied arguments.
//! 2. **Validate** the remaining arguments against the model-facing schema.
//! 3. **Inject** the host's real values.
//! 4. Invoke the tool.
//!
//! Step 1 must come first and must be unconditional. A model that has seen an
//! injected key named in a prompt, a log, or an error message can put that key
//! in its own `arguments` object; if the host merges its value in *after*, a
//! well-formed merge might still let the model's value win, and if the host
//! merges *before* validating, the forged key rides along. LangGraph's
//! `ToolNode` strips first for exactly this reason, with the comment that it
//! "prevents an LLM from forging hidden InjectedToolArg fields via
//! ToolCall.args".
//!
//! [`strip_injected_arguments`] performs step 1 and reports what it removed, so
//! a forgery attempt is visible in the log rather than silent.
//!
//! # Status
//!
//! The declaration side is live: [`Tool::injected_arguments`][super::Tool::injected_arguments]
//! declares the keys and [`ToolRegistry::schemas`][super::ToolRegistry::schemas]
//! already projects them out of what the model sees. The enforcement side (the
//! four-step sequence above) belongs to the agent loop's tool-execution path.

use serde_json::Value;

use super::types::ToolSchema;

/// Removes every key in `injected` from a model-supplied argument object,
/// returning the names that were actually present.
///
/// A non-empty return value means the model emitted a key it was never shown —
/// either because it inferred one, or because it was told one. Callers should
/// log it; the value itself is discarded either way.
///
/// Non-object arguments (including the raw string preserved on an
/// [`invalid`][super::ToolCall::invalid] call) are left untouched: there is no
/// key to forge in a scalar.
pub fn strip_injected_arguments(arguments: &mut Value, injected: &[&str]) -> Vec<String> {
    if injected.is_empty() {
        return Vec::new();
    }
    let Some(object) = arguments.as_object_mut() else {
        return Vec::new();
    };

    let mut removed = Vec::new();
    for key in injected {
        if object.remove(*key).is_some() {
            removed.push((*key).to_string());
        }
    }

    if !removed.is_empty() {
        tinyagents_tracing::warn!(
            "[tool::injected] discarded model-supplied value(s) for host-injected argument(s): {}",
            removed.join(", ")
        );
    }
    removed
}

/// Removes `injected` keys from a schema's `properties` **and** its `required`
/// list, producing the model-facing projection of the declaration.
///
/// Dropping a key from `properties` alone is not enough: leaving it in
/// `required` tells the model to supply an argument it cannot see, which is
/// either a validation failure or an invitation to invent the value.
pub fn project_injected_arguments(mut schema: ToolSchema, injected: &[&str]) -> ToolSchema {
    if injected.is_empty() {
        return schema;
    }
    let Some(parameters) = schema.parameters.as_object_mut() else {
        return schema;
    };

    if let Some(Value::Object(properties)) = parameters.get_mut("properties") {
        for key in injected {
            properties.remove(*key);
        }
    }

    if let Some(Value::Array(required)) = parameters.get_mut("required") {
        required.retain(|value| value.as_str().is_none_or(|name| !injected.contains(&name)));
    }

    tinyagents_tracing::trace!(
        "[tool::injected] projected {} hidden argument(s) out of `{}`",
        injected.len(),
        schema.name
    );
    schema
}
