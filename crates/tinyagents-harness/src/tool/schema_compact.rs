//! Byte-budgeted compaction of tool schemas.
//!
//! A tool schema is paid on every request, so an oversized one is a standing
//! tax. This module bounds it the way Codex bounds MCP schemas: a description
//! cap plus a graduated ladder that removes the least load-bearing parts of
//! the parameter schema first and stops as soon as the serialised size fits.
//!
//! The ladder, in order:
//!
//! 1. prune `$defs`/`definitions` entries nothing references (always safe —
//!    after [`super::SchemaCleanr`] has inlined refs they are dead weight);
//! 2. strip `description` from every property nested below the top level;
//! 3. strip `description` from top-level properties too;
//! 4. drop the definition tables outright, rewriting any surviving `$ref` to
//!    an open object;
//! 5. collapse objects nested deeper than [`COLLAPSE_DEPTH`] to
//!    `{"type": "object"}`;
//! 6. drop `anyOf` / `oneOf` / `allOf` compositions.
//!
//! Each rung keeps the top-level argument surface (property names, types,
//! `required`) intact, which is what a model needs to produce a valid call.

use serde_json::{Map, Value, json};
use tinyinference_llm::tool::ToolSchema;

/// Objects nested deeper than this are collapsed by rung 5.
pub const COLLAPSE_DEPTH: usize = 3;

/// Size limits applied to one tool schema after provider cleaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchemaCompaction {
    /// Serialised-parameter budget in bytes; `None` disables the ladder.
    /// Codex's default of 5,000 bytes is a cheap proxy for ~1k tokens.
    pub max_schema_bytes: Option<usize>,
    /// Cap on the tool description in bytes (cut at a char boundary, marked
    /// with `…`); `None` leaves descriptions alone.
    pub max_description_bytes: Option<usize>,
}

impl SchemaCompaction {
    /// No compaction at all.
    pub const NONE: Self = Self {
        max_schema_bytes: None,
        max_description_bytes: None,
    };

    /// Codex's budget for third-party tools: 5,000-byte parameters and a
    /// 1,000-byte description.
    pub const THIRD_PARTY: Self = Self {
        max_schema_bytes: Some(5_000),
        max_description_bytes: Some(1_000),
    };

    /// `true` when neither limit is set.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        self.max_schema_bytes.is_none() && self.max_description_bytes.is_none()
    }
}

/// Applies `compaction` to one schema, leaving name and format untouched.
///
/// `max_schema_bytes` is enforced as a hard ceiling here, not just a target
/// for the ladder: when even the maximally-compacted rung (which keeps the
/// top-level argument surface intact by design — see [`compact_parameters`])
/// still exceeds the budget, the parameters are replaced with an open
/// `{"type":"object","properties":{}}` schema rather than shipping a request
/// over the configured limit. This only widens what is *advertised*; it does
/// not weaken admission, which always validates against the tool's canonical
/// declared schema, never the wire-projected one (see
/// `docs/modules/harness/tool-discovery.md`).
#[must_use]
pub fn compact_tool_schema(schema: &ToolSchema, compaction: &SchemaCompaction) -> ToolSchema {
    let description = match compaction.max_description_bytes {
        Some(max) => clip_bytes(&schema.description, max),
        None => schema.description.clone(),
    };
    let parameters = match compaction.max_schema_bytes {
        Some(max) => {
            let compacted = compact_parameters(schema.parameters.clone(), max);
            if serialized_len(&compacted) > max {
                tracing::warn!(
                    "[tool::schema] `{}`'s parameters still exceed the {max}-byte compaction \
                     budget after the full ladder; advertising an open object schema instead of \
                     sending an over-budget request",
                    schema.name
                );
                json!({"type": "object", "properties": {}})
            } else {
                compacted
            }
        }
        None => schema.parameters.clone(),
    };
    ToolSchema {
        name: schema.name.clone(),
        description,
        parameters,
        format: schema.format.clone(),
    }
}

/// Runs the compaction ladder until `schema` serialises within `max_bytes`.
///
/// This function itself is best-effort: a schema whose top-level property
/// *surface* (names, types, `required`) alone exceeds `max_bytes` cannot
/// shrink further without dropping arguments the model needs to see, which
/// every rung refuses to do (see the module docs — each rung keeps the
/// top-level argument surface intact). When the final rung
/// (`drop_compositions`) still exceeds the budget, that maximally-compacted
/// value is returned rather than an error. [`compact_tool_schema`] is the
/// caller that turns this into a hard ceiling: it checks the result against
/// `max_bytes` and falls back to an open object schema when this function
/// could not get under budget, so `max_schema_bytes` is always actually
/// enforced on the wire.
#[must_use]
pub fn compact_parameters(schema: Value, max_bytes: usize) -> Value {
    let rungs: [fn(Value) -> Value; 6] = [
        prune_unreachable_definitions,
        |value| strip_descriptions(value, 1),
        |value| strip_descriptions(value, 0),
        drop_definitions,
        |value| collapse_deep_objects(value, 0),
        drop_compositions,
    ];
    let mut current = schema;
    for rung in rungs {
        if serialized_len(&current) <= max_bytes {
            return current;
        }
        current = rung(current);
    }
    current
}

fn serialized_len(value: &Value) -> usize {
    serde_json::to_vec(value).map_or(usize::MAX, |bytes| bytes.len())
}

/// Rung 1: keeps only the definitions some `$ref` still reaches.
#[must_use]
pub fn prune_unreachable_definitions(mut schema: Value) -> Value {
    let Some(object) = schema.as_object_mut() else {
        return schema;
    };
    for table in ["$defs", "definitions"] {
        let Some(Value::Object(defs)) = object.get(table).cloned() else {
            continue;
        };
        // Reachability is computed against the schema *without* the table, then
        // transitively through the kept definitions themselves.
        let mut body = object.clone();
        body.remove(table);
        let mut reachable = std::collections::HashSet::new();
        let mut frontier: Vec<String> = Vec::new();
        collect_refs(&Value::Object(body), table, &mut frontier);
        while let Some(name) = frontier.pop() {
            if reachable.insert(name.clone())
                && let Some(definition) = defs.get(&name)
            {
                collect_refs(definition, table, &mut frontier);
            }
        }
        let kept: Map<String, Value> = defs
            .into_iter()
            .filter(|(name, _)| reachable.contains(name))
            .collect();
        if kept.is_empty() {
            object.remove(table);
        } else {
            object.insert(table.to_string(), Value::Object(kept));
        }
    }
    schema
}

fn collect_refs(value: &Value, table: &str, out: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            if let Some(Value::String(target)) = object.get("$ref")
                && let Some(name) = target.strip_prefix(&format!("#/{table}/"))
            {
                // `$ref` fragments are RFC 6901 JSON Pointer tokens: `/` and
                // `~` in the original definition name are escaped as `~1`
                // and `~0`. Decoding here (order matters — `~1` before
                // `~0`, matching the standard decode algorithm) keeps this
                // reachability scan in the same key space as `$defs`'s own
                // (unescaped) keys; skipping it would treat a definition
                // named e.g. `"a/b"` as unreachable and prune it even though
                // a `$ref: "#/$defs/a~1b"` still points at it.
                out.push(name.replace("~1", "/").replace("~0", "~"));
            }
            for child in object.values() {
                collect_refs(child, table, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_refs(item, table, out);
            }
        }
        _ => {}
    }
}

/// Rungs 2–3: removes `description` from properties nested deeper than
/// `keep_depth` levels of properties (depth 0 = every property).
#[must_use]
pub fn strip_descriptions(schema: Value, keep_depth: usize) -> Value {
    strip_descriptions_at(schema, keep_depth, 0)
}

fn strip_descriptions_at(schema: Value, keep_depth: usize, depth: usize) -> Value {
    let Value::Object(mut object) = schema else {
        return schema;
    };
    if depth > keep_depth {
        object.remove("description");
    }
    for (key, value) in object.iter_mut() {
        match key.as_str() {
            "properties" | "$defs" | "definitions" => {
                if let Value::Object(entries) = value {
                    for entry in entries.values_mut() {
                        let taken = std::mem::take(entry);
                        *entry = strip_descriptions_at(taken, keep_depth, depth + 1);
                    }
                }
            }
            "items" | "additionalProperties" => {
                let taken = std::mem::take(value);
                *value = strip_descriptions_at(taken, keep_depth, depth + 1);
            }
            "anyOf" | "oneOf" | "allOf" => {
                if let Value::Array(variants) = value {
                    for variant in variants.iter_mut() {
                        let taken = std::mem::take(variant);
                        *variant = strip_descriptions_at(taken, keep_depth, depth);
                    }
                }
            }
            _ => {}
        }
    }
    Value::Object(object)
}

/// Rung 4: drops the definition tables and opens any surviving `$ref`.
#[must_use]
pub fn drop_definitions(schema: Value) -> Value {
    match schema {
        Value::Object(mut object) => {
            object.remove("$defs");
            object.remove("definitions");
            if object.remove("$ref").is_some() {
                // A `$ref` can resolve to any JSON type, not only an object
                // (a string enum, a number, a union). Replacing it with
                // `{"type": "object"}` would advertise a type the referenced
                // schema never promised: the model could then produce an
                // object while admission still validates the call against
                // the original (unreachable) definition, and a
                // schema-conformant call fails. Drop the reference and leave
                // the schema unconstrained instead — no `type` means "any
                // JSON value", which only widens what is advertised, never
                // narrows it.
                return json!({});
            }
            let rebuilt: Map<String, Value> = object
                .into_iter()
                .map(|(key, value)| (key, drop_definitions(value)))
                .collect();
            Value::Object(rebuilt)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(drop_definitions).collect()),
        other => other,
    }
}

/// Rung 5: collapses object schemas nested deeper than [`COLLAPSE_DEPTH`].
#[must_use]
pub fn collapse_deep_objects(schema: Value, depth: usize) -> Value {
    let Value::Object(mut object) = schema else {
        return schema;
    };
    let is_object_schema = matches!(object.get("type"), Some(Value::String(t)) if t == "object")
        || object.contains_key("properties");
    if is_object_schema && depth >= COLLAPSE_DEPTH {
        let mut collapsed = Map::new();
        collapsed.insert("type".to_string(), Value::String("object".to_string()));
        if let Some(description) = object.remove("description") {
            collapsed.insert("description".to_string(), description);
        }
        return Value::Object(collapsed);
    }
    for (key, value) in object.iter_mut() {
        match key.as_str() {
            "properties" | "$defs" | "definitions" => {
                if let Value::Object(entries) = value {
                    for entry in entries.values_mut() {
                        let taken = std::mem::take(entry);
                        *entry = collapse_deep_objects(taken, depth + 1);
                    }
                }
            }
            "items" | "additionalProperties" => {
                let taken = std::mem::take(value);
                *value = collapse_deep_objects(taken, depth + 1);
            }
            "anyOf" | "oneOf" | "allOf" => {
                if let Value::Array(variants) = value {
                    for variant in variants.iter_mut() {
                        let taken = std::mem::take(variant);
                        *variant = collapse_deep_objects(taken, depth);
                    }
                }
            }
            _ => {}
        }
    }
    Value::Object(object)
}

/// Rung 6: removes `anyOf` / `oneOf` / `allOf` everywhere. A level that held
/// nothing but a composition becomes an open object.
#[must_use]
pub fn drop_compositions(schema: Value) -> Value {
    match schema {
        Value::Object(mut object) => {
            // As with `$ref` above, a composition (`anyOf`/`oneOf`/`allOf`)
            // can describe a primitive or a union of types, not only an
            // object. Dropping it must not force `type: "object"` — leave
            // the schema unconstrained (whatever `type`/`properties` remain
            // after the composition keys are gone, or nothing at all) so the
            // model is never told to send a type the schema no longer
            // constrains it to.
            for key in ["anyOf", "oneOf", "allOf"] {
                object.remove(key);
            }
            let rebuilt: Map<String, Value> = object
                .into_iter()
                .map(|(key, value)| (key, drop_compositions(value)))
                .collect();
            Value::Object(rebuilt)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(drop_compositions).collect()),
        other => other,
    }
}

/// Clips `text` to at most `max_bytes` bytes on a char boundary, appending `…`
/// (which itself fits inside the budget) when anything was removed.
#[must_use]
pub fn clip_bytes(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    const MARK: &str = "…";
    // The `…` marker alone is `MARK.len()` (3) bytes. A budget smaller than
    // that cannot contain even the marker, so the only value that respects
    // `max_bytes` is the empty string — returning the marker anyway (as
    // `saturating_sub` plus a bare `format!` would) silently exceeds the
    // configured `max_description_bytes`.
    if max_bytes < MARK.len() {
        return String::new();
    }
    let budget = max_bytes - MARK.len();
    let mut end = budget.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{MARK}", &text[..end])
}

#[cfg(test)]
#[path = "schema_compact_test.rs"]
mod test;
