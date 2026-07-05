//! The embedded Rhai interpreter backend.
//!
//! The engine is built fresh for every cell over a **persistent scope**, so
//! variables survive across cells while the capability closures always see
//! the current cell's deadline. The engine has no filesystem, network, or
//! process access — the closures registered here are its entire host
//! surface, which is what makes this backend the hermetic default.
//!
//! Capability calls block the evaluating thread through the fail-closed
//! [`bridge_block_on`](super::bridge_block_on) adapter; the session drives
//! `eval_cell` inside [`tokio::task::spawn_blocking`], so blocking here never
//! starves the async runtime driving the underlying provider I/O.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use rhai::{Array, Dynamic, Engine, EvalAltResult, Map, Position, Scope};
use serde_json::Value;

use super::{CellEval, RlmInterpreter, bridge_block_on};
use crate::error::{Result, TinyAgentsError};
use crate::rlm::host::{RlmHostApi, is_fatal};
use crate::rlm::types::HostCall;

/// Sentinel runtime-error text `on_progress` terminates a cell with when the
/// wall-clock deadline elapses mid-script.
const DEADLINE_TOKEN: &str = "rlm cell exceeded its wall-clock timeout";

/// Sentinel runtime-error text `on_progress` terminates a cell with when the
/// session cancel flag trips mid-script.
const CANCELLED_TOKEN: &str = "rlm cell cancelled by host";

/// The embedded Rhai backend. See the [module docs](self).
pub struct RhaiInterpreter {
    max_operations: u64,
    scope: Scope<'static>,
}

impl RhaiInterpreter {
    /// Creates a backend bounded by `max_operations` Rhai operations per cell
    /// (`0` means unlimited).
    pub fn new(max_operations: u64) -> Self {
        Self {
            max_operations,
            scope: Scope::new(),
        }
    }
}

/// Shared per-cell buffers the capability closures write into.
#[derive(Default)]
struct CellState {
    stdout: String,
    /// A fatal host error (limit/timeout/cancel) stashed so `eval_cell` can
    /// surface the precise crate error instead of its stringified form.
    fatal: Option<TinyAgentsError>,
}

type SharedCellState = Arc<Mutex<CellState>>;

/// Dispatches one host call from a synchronous capability closure, blocking
/// through the bridge and splitting fatal from script-visible failures.
fn dispatch(
    host: &Arc<dyn RlmHostApi>,
    cell: &SharedCellState,
    call: HostCall,
) -> std::result::Result<Value, Box<EvalAltResult>> {
    let deadline = host.deadline();
    let cancel = host.cancel_flag();
    match bridge_block_on(deadline, &cancel, host.handle(call)) {
        Ok(value) => Ok(value),
        Err(err) => {
            let message = err.to_string();
            if is_fatal(&err) {
                cell.lock().expect("cell state poisoned").fatal = Some(err);
            }
            Err(Box::new(EvalAltResult::ErrorRuntime(
                Dynamic::from(message),
                Position::NONE,
            )))
        }
    }
}

// ── Dynamic ⇄ JSON conversion ───────────────────────────────────────────────

/// Converts a Rhai value into JSON. Opaque host types are stringified rather
/// than leaked.
pub(crate) fn dynamic_to_json(value: &Dynamic) -> Value {
    if value.is_unit() {
        Value::Null
    } else if let Ok(b) = value.as_bool() {
        Value::Bool(b)
    } else if let Ok(i) = value.as_int() {
        Value::from(i)
    } else if let Ok(f) = value.as_float() {
        serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    } else if let Some(s) = value.read_lock::<rhai::ImmutableString>() {
        Value::String(s.to_string())
    } else if let Some(array) = value.read_lock::<Array>() {
        Value::Array(array.iter().map(dynamic_to_json).collect())
    } else if let Some(map) = value.read_lock::<Map>() {
        Value::Object(
            map.iter()
                .map(|(k, v)| (k.to_string(), dynamic_to_json(v)))
                .collect(),
        )
    } else {
        Value::String(value.to_string())
    }
}

/// Converts JSON into a Rhai value.
pub(crate) fn json_to_dynamic(value: &Value) -> Dynamic {
    match value {
        Value::Null => Dynamic::UNIT,
        Value::Bool(b) => Dynamic::from(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Dynamic::from(i)
            } else {
                Dynamic::from(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => Dynamic::from(s.clone()),
        Value::Array(items) => {
            Dynamic::from_array(items.iter().map(json_to_dynamic).collect::<Array>())
        }
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                out.insert(k.clone().into(), json_to_dynamic(v));
            }
            Dynamic::from_map(out)
        }
    }
}

/// Builds the sandboxed engine for one cell, registering the capability
/// closures against `host` and the shared cell buffers.
fn build_engine(host: Arc<dyn RlmHostApi>, cell: SharedCellState, max_operations: u64) -> Engine {
    let mut engine = Engine::new();
    engine.set_max_operations(max_operations);

    // Fail-closed mid-script enforcement: `on_progress` fires between Rhai
    // statements/operations, catching runaway script loops with no host
    // calls. In-flight capability calls are bounded separately by the
    // blocking bridge.
    let progress_host = host.clone();
    let progress_cell = cell.clone();
    engine.on_progress(move |_ops| {
        if progress_host.cancel_flag().is_cancelled() {
            return Some(Dynamic::from(CANCELLED_TOKEN.to_string()));
        }
        if progress_cell
            .lock()
            .expect("cell state poisoned")
            .fatal
            .is_some()
        {
            return Some(Dynamic::from(DEADLINE_TOKEN.to_string()));
        }
        match progress_host.deadline() {
            Some(deadline) if Instant::now() >= deadline => {
                Some(Dynamic::from(DEADLINE_TOKEN.to_string()))
            }
            _ => None,
        }
    });

    // ── print / debug capture ──
    let print_cell = cell.clone();
    engine.on_print(move |text| {
        let mut state = print_cell.lock().expect("cell state poisoned");
        state.stdout.push_str(text);
        state.stdout.push('\n');
    });
    let debug_cell = cell.clone();
    engine.on_debug(move |text, _source, _pos| {
        let mut state = debug_cell.lock().expect("cell state poisoned");
        state.stdout.push_str(text);
        state.stdout.push('\n');
    });

    // ── llm(prompt) / llm(#{ model, prompt, system }) ──
    let llm_host = host.clone();
    let llm_cell = cell.clone();
    engine.register_fn(
        "llm",
        move |prompt: &str| -> std::result::Result<String, Box<EvalAltResult>> {
            let value = dispatch(
                &llm_host,
                &llm_cell,
                HostCall::Llm {
                    model: None,
                    prompt: prompt.to_string(),
                    system: None,
                },
            )?;
            Ok(value.as_str().unwrap_or_default().to_string())
        },
    );
    let llm_map_host = host.clone();
    let llm_map_cell = cell.clone();
    engine.register_fn(
        "llm",
        move |params: Map| -> std::result::Result<String, Box<EvalAltResult>> {
            let get = |key: &str| params.get(key).and_then(|d| d.clone().into_string().ok());
            let prompt = get("prompt").ok_or_else(|| {
                Box::new(EvalAltResult::ErrorRuntime(
                    Dynamic::from("llm: missing `prompt`".to_string()),
                    Position::NONE,
                ))
            })?;
            let value = dispatch(
                &llm_map_host,
                &llm_map_cell,
                HostCall::Llm {
                    model: get("model"),
                    prompt,
                    system: get("system"),
                },
            )?;
            Ok(value.as_str().unwrap_or_default().to_string())
        },
    );

    // ── tool(name) / tool(name, #{ ... }) ──
    let tool_host = host.clone();
    let tool_cell = cell.clone();
    engine.register_fn(
        "tool",
        move |name: &str| -> std::result::Result<Dynamic, Box<EvalAltResult>> {
            let value = dispatch(
                &tool_host,
                &tool_cell,
                HostCall::Tool {
                    tool: name.to_string(),
                    arguments: Value::Null,
                },
            )?;
            Ok(json_to_dynamic(&value))
        },
    );
    let tool_args_host = host.clone();
    let tool_args_cell = cell.clone();
    engine.register_fn(
        "tool",
        move |name: &str, args: Map| -> std::result::Result<Dynamic, Box<EvalAltResult>> {
            let arguments = dynamic_to_json(&Dynamic::from_map(args));
            let value = dispatch(
                &tool_args_host,
                &tool_args_cell,
                HostCall::Tool {
                    tool: name.to_string(),
                    arguments,
                },
            )?;
            Ok(json_to_dynamic(&value))
        },
    );

    // ── agent(name, input) ──
    let agent_host = host.clone();
    let agent_cell = cell.clone();
    engine.register_fn(
        "agent",
        move |name: &str, input: &str| -> std::result::Result<String, Box<EvalAltResult>> {
            let value = dispatch(
                &agent_host,
                &agent_cell,
                HostCall::Agent {
                    agent: name.to_string(),
                    input: input.to_string(),
                    data: None,
                },
            )?;
            Ok(value.as_str().unwrap_or_default().to_string())
        },
    );

    // ── final_answer(text) ──
    let answer_host = host.clone();
    let answer_cell = cell.clone();
    engine.register_fn(
        "final_answer",
        move |text: &str| -> std::result::Result<(), Box<EvalAltResult>> {
            dispatch(
                &answer_host,
                &answer_cell,
                HostCall::FinalAnswer {
                    answer: text.to_string(),
                },
            )?;
            Ok(())
        },
    );

    engine
}

#[async_trait]
impl RlmInterpreter for RhaiInterpreter {
    fn language(&self) -> &str {
        "rhai"
    }

    fn usage_guide(&self) -> String {
        r#"Write Rhai. Variables persist across cells. Host functions:
- llm(prompt)                          -> string   // ask the default sub-LLM
- llm(#{ model: "name", prompt: "...", system: "..." }) -> string
- tool("name", #{ arg: value, ... })   -> result   // call a registered tool
- agent("name", "input")               -> string   // delegate to a sub-agent
- final_answer("...")                             // end the task with this answer
- print(x)                                        // observe a value in the next turn
Errors raised by capabilities can be caught with try/catch. The last
expression of a cell is echoed back to you as its value.

Rhai syntax notes (Rhai is NOT JavaScript or Rust):
- There are NO tuples: return an array `[a, b]` or an object map `#{ k: v }`.
- Object maps are written `#{ name: value }` and indexed with `m.name` or `m["name"]`.
- Sub-arrays: `arr.extract(0..5)`; also `arr.len()`, `arr.push(x)`, `arr.filter(|x| ...)`,
  `arr.map(|x| ...)`; loops: `for x in arr { ... }` and `for i in 0..n { ... }`.
- Strings concatenate with `+`; convert with `x.to_string()`; interpolate with `` `${x}` ``.
- `let` declares a mutable variable; statements end with `;`."#
            .to_string()
    }

    async fn set_variable(&mut self, name: &str, value: Value) -> Result<()> {
        self.scope
            .set_value(name.to_string(), json_to_dynamic(&value));
        Ok(())
    }

    async fn eval_cell(&mut self, code: &str, host: Arc<dyn RlmHostApi>) -> Result<CellEval> {
        let cell: SharedCellState = Arc::new(Mutex::new(CellState::default()));
        let engine = build_engine(host, cell.clone(), self.max_operations);
        let mut scope = std::mem::take(&mut self.scope);
        let code = code.to_string();

        // Rhai is synchronous and the capability closures block through the
        // bridge, so evaluate on the blocking pool to keep the async runtime
        // (which drives the actual provider I/O) responsive.
        let (scope_back, eval) = tokio::task::spawn_blocking(move || {
            let result = engine.eval_with_scope::<Dynamic>(&mut scope, &code);
            (scope, result)
        })
        .await
        .map_err(|err| TinyAgentsError::Model(format!("rlm rhai eval task failed: {err}")))?;
        self.scope = scope_back;

        let mut state = cell.lock().expect("cell state poisoned");
        if let Some(fatal) = state.fatal.take() {
            return Err(fatal);
        }
        let stdout = std::mem::take(&mut state.stdout);
        drop(state);

        match eval {
            Ok(value) => {
                let json = dynamic_to_json(&value);
                Ok(CellEval {
                    stdout,
                    value: (!json.is_null()).then_some(json),
                    error: None,
                })
            }
            Err(err) => {
                let message = err.to_string();
                if message.contains(DEADLINE_TOKEN) {
                    return Err(TinyAgentsError::Timeout(DEADLINE_TOKEN.to_string()));
                }
                if message.contains(CANCELLED_TOKEN) {
                    return Err(TinyAgentsError::Cancelled);
                }
                if matches!(*err, EvalAltResult::ErrorTooManyOperations(_)) {
                    return Err(TinyAgentsError::LimitExceeded(
                        "rlm cell exceeded the operation limit".to_string(),
                    ));
                }
                Ok(CellEval {
                    stdout,
                    value: None,
                    error: Some(message),
                })
            }
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
