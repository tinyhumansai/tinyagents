//! The external-process interpreter backend (Python, Node.js, or any command
//! speaking the wire protocol).
//!
//! ## Wire protocol
//!
//! Line-delimited JSON over the child's stdin/stdout. The child is a
//! long-lived REPL: globals persist across cells. Frames, host → child:
//!
//! - `{"op":"eval","code":"..."}` — evaluate one cell.
//! - `{"op":"set_var","name":"...","value":<json>}` — set a global.
//! - `{"op":"call_result","ok":true,"value":<json>}` /
//!   `{"op":"call_result","ok":false,"error":"..."}` — the reply to a
//!   capability call (script-visible failures arrive as `ok:false`; *fatal*
//!   failures never get a reply — the host kills the child instead).
//! - `{"op":"shutdown"}` — exit cleanly.
//!
//! Child → host:
//!
//! - `{"op":"ready"}` — emitted once after bootstrap.
//! - `{"op":"call","call":{"capability":"llm"|"tool"|"agent"|"final_answer",...}}`
//!   — a capability call (the `call` payload is a serialized
//!   [`HostCall`]); the child blocks until the matching `call_result`.
//! - `{"op":"result","stdout":"...","value":<json>,"error":<string|null>}` —
//!   the cell outcome.
//! - `{"op":"var_set"}` — acknowledges `set_var`.
//!
//! Calls are strictly sequential (one cell evaluates at a time and blocks on
//! each capability call), so frames need no correlation ids.
//!
//! ## Sandboxing honesty
//!
//! Unlike the embedded Rhai backend, a child process has whatever OS access
//! the embedder's environment grants it. The host still enforces every
//! [`RlmPolicy`](crate::rlm::RlmPolicy) limit fail-closed — a cell that
//! exceeds its deadline or trips a policy bound gets its child **killed**,
//! not asked nicely — but filesystem/network isolation for untrusted models
//! must come from the embedder (container, jail, seccomp, a locked-down
//! `InterpreterSpec::Command` runner).

use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};

use super::{CellEval, RlmInterpreter};
use crate::error::{Result, TinyAgentsError};
use crate::rlm::host::{RlmHostApi, is_fatal};
use crate::rlm::types::HostCall;

/// The Python bootstrap program (`python3 -u -c <this>`), a minimal REPL
/// speaking the wire protocol. User prints are captured per cell; the
/// protocol channel is the real stdout, saved before any redirection.
const PYTHON_PRELUDE: &str = r#"
import ast, contextlib, io, json, sys, traceback
_out, _in = sys.stdout, sys.stdin

def _send(obj):
    _out.write(json.dumps(obj) + "\n"); _out.flush()

class RlmError(Exception):
    pass

def _call(call):
    _send({"op": "call", "call": call})
    while True:
        line = _in.readline()
        if not line:
            sys.exit(0)
        msg = json.loads(line)
        if msg.get("op") == "call_result":
            if msg.get("ok"):
                return msg.get("value")
            raise RlmError(msg.get("error") or "capability call failed")

def llm(prompt, model=None, system=None):
    if isinstance(prompt, dict):
        model, system, prompt = prompt.get("model"), prompt.get("system"), prompt.get("prompt")
    return _call({"capability": "llm", "prompt": prompt, "model": model, "system": system})

def tool(name, arguments=None):
    return _call({"capability": "tool", "tool": name, "arguments": arguments})

def agent(name, input, data=None):
    return _call({"capability": "agent", "agent": name, "input": str(input), "data": data})

def final_answer(answer):
    _call({"capability": "final_answer", "answer": str(answer)})

_g = {"__name__": "__rlm__", "llm": llm, "tool": tool, "agent": agent,
      "final_answer": final_answer, "RlmError": RlmError}

def _eval(code):
    buf = io.StringIO()
    value, error = None, None
    try:
        with contextlib.redirect_stdout(buf):
            tree = ast.parse(code, mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body[-1].value)
                body = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(body, "<rlm>", "exec"), _g)
                value = eval(compile(last, "<rlm>", "eval"), _g)
            else:
                exec(compile(tree, "<rlm>", "exec"), _g)
    except Exception:
        error = traceback.format_exc(limit=4)
    try:
        json.dumps(value)
    except Exception:
        value = repr(value)
    _send({"op": "result", "stdout": buf.getvalue(), "value": value, "error": error})

_send({"op": "ready"})
for _line in _in:
    _msg = json.loads(_line)
    _op = _msg.get("op")
    if _op == "eval":
        _eval(_msg.get("code") or "")
    elif _op == "set_var":
        _g[_msg["name"]] = _msg.get("value")
        _send({"op": "var_set"})
    elif _op == "shutdown":
        break
"#;

/// The Node.js bootstrap program (`node -e <this>`). Cells run in a
/// persistent `vm` context; `console.log` is captured per cell; capability
/// calls block on stdin with `fs.readSync`.
const JAVASCRIPT_PRELUDE: &str = r#"
const fs = require('fs');
const vm = require('vm');
let inbuf = Buffer.alloc(0);
function readLine() {
  for (;;) {
    const idx = inbuf.indexOf(10);
    if (idx >= 0) {
      const line = inbuf.slice(0, idx).toString('utf8');
      inbuf = inbuf.slice(idx + 1);
      return line;
    }
    const chunk = Buffer.alloc(65536);
    let n;
    try { n = fs.readSync(0, chunk, 0, chunk.length, null); }
    catch (e) { if (e.code === 'EAGAIN') continue; throw e; }
    if (n === 0) process.exit(0);
    inbuf = Buffer.concat([inbuf, chunk.slice(0, n)]);
  }
}
function send(obj) { fs.writeSync(1, JSON.stringify(obj) + '\n'); }
class RlmError extends Error {}
function call(c) {
  send({ op: 'call', call: c });
  for (;;) {
    const msg = JSON.parse(readLine());
    if (msg.op === 'call_result') {
      if (msg.ok) return msg.value === undefined ? null : msg.value;
      throw new RlmError(msg.error || 'capability call failed');
    }
  }
}
let stdoutBuf = '';
function logLine(args) {
  stdoutBuf += args.map(x => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ') + '\n';
}
const sandbox = {
  llm: p => (typeof p === 'string'
    ? call({ capability: 'llm', prompt: p, model: null, system: null })
    : call({ capability: 'llm', prompt: p.prompt, model: p.model || null, system: p.system || null })),
  tool: (name, args) => call({ capability: 'tool', tool: name, arguments: args === undefined ? null : args }),
  agent: (name, input) => call({ capability: 'agent', agent: name, input: String(input), data: null }),
  final_answer: a => { call({ capability: 'final_answer', answer: String(a) }); },
  console: { log: (...xs) => logLine(xs), error: (...xs) => logLine(xs), info: (...xs) => logLine(xs) },
  RlmError,
  JSON, Math,
};
const ctx = vm.createContext(sandbox);
send({ op: 'ready' });
for (;;) {
  const msg = JSON.parse(readLine());
  if (msg.op === 'eval') {
    stdoutBuf = '';
    let value = null, error = null;
    try {
      value = vm.runInContext(msg.code, ctx, { filename: '<rlm>' });
      if (value === undefined) value = null;
      if (typeof value === 'function') value = String(value);
      try { JSON.stringify(value); } catch { value = String(value); }
    } catch (e) {
      error = e && e.stack ? String(e.stack).split('\n').slice(0, 4).join('\n') : String(e);
    }
    send({ op: 'result', stdout: stdoutBuf, value, error });
  } else if (msg.op === 'set_var') {
    sandbox[msg.name] = msg.value;
    send({ op: 'var_set' });
  } else if (msg.op === 'shutdown') {
    process.exit(0);
  }
}
"#;

/// How long the child gets to print its `ready` frame after spawning.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);

/// The fallback bound for protocol exchanges when the session armed no cell
/// deadline (`RlmPolicy::cell_timeout: None`).
const DEFAULT_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(300);

/// A live child process with its protocol streams.
struct ChildHandle {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Rolling capture of the child's stderr, included in error messages.
    stderr: Arc<Mutex<String>>,
}

/// The external-process backend. See the [module docs](self).
pub struct ExternalInterpreter {
    language: String,
    binary: String,
    args: Vec<String>,
    child: Option<ChildHandle>,
}

impl ExternalInterpreter {
    /// A CPython child (`binary` defaults to `python3`).
    pub fn python(binary: Option<&str>, extra_args: Vec<String>) -> Self {
        let mut args = extra_args;
        args.extend(["-u".to_string(), "-c".to_string(), PYTHON_PRELUDE.to_string()]);
        Self {
            language: "python".to_string(),
            binary: binary.unwrap_or("python3").to_string(),
            args,
            child: None,
        }
    }

    /// A Node.js child (`binary` defaults to `node`).
    pub fn javascript(binary: Option<&str>, extra_args: Vec<String>) -> Self {
        let mut args = extra_args;
        args.extend(["-e".to_string(), JAVASCRIPT_PRELUDE.to_string()]);
        Self {
            language: "javascript".to_string(),
            binary: binary.unwrap_or("node").to_string(),
            args,
            child: None,
        }
    }

    /// An arbitrary command that speaks the wire protocol itself. Scripts are
    /// assumed to be Python-flavored for prompt purposes.
    pub fn command(binary: String, args: Vec<String>) -> Self {
        Self {
            language: "python".to_string(),
            binary,
            args,
            child: None,
        }
    }

    fn stderr_tail(&self) -> String {
        self.child
            .as_ref()
            .map(|c| {
                let text = c.stderr.lock().expect("stderr buffer poisoned");
                let tail: String = text.chars().rev().take(2000).collect();
                tail.chars().rev().collect()
            })
            .unwrap_or_default()
    }

    fn broken(&mut self, context: &str) -> TinyAgentsError {
        let stderr = self.stderr_tail();
        self.child = None;
        let mut message = format!("rlm external interpreter ({}): {context}", self.binary);
        if !stderr.is_empty() {
            message.push_str(&format!("; stderr tail: {stderr}"));
        }
        TinyAgentsError::Capability(message)
    }

    async fn ensure_started(&mut self) -> Result<()> {
        if self.child.is_some() {
            return Ok(());
        }
        let mut child = tokio::process::Command::new(&self.binary)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|err| {
                TinyAgentsError::Capability(format!(
                    "rlm external interpreter: failed to spawn `{}`: {err}",
                    self.binary
                ))
            })?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        let stderr_pipe = child.stderr.take().expect("piped stderr");
        let stderr = Arc::new(Mutex::new(String::new()));
        let stderr_buf = stderr.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr_pipe);
            let mut buffer = [0u8; 4096];
            loop {
                match reader.read(&mut buffer).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let mut text = stderr_buf.lock().expect("stderr buffer poisoned");
                        text.push_str(&String::from_utf8_lossy(&buffer[..n]));
                        // Keep only a bounded tail.
                        if text.len() > 16 * 1024 {
                            let cut = text.len() - 8 * 1024;
                            *text = text[cut..].to_string();
                        }
                    }
                }
            }
        });
        self.child = Some(ChildHandle {
            child,
            stdin,
            stdout,
            stderr,
        });

        // Wait for the bootstrap's `ready` frame.
        match self.read_frame(Instant::now() + STARTUP_TIMEOUT).await {
            Ok(frame) if frame.get("op").and_then(Value::as_str) == Some("ready") => Ok(()),
            Ok(frame) => Err(self.broken(&format!("unexpected startup frame: {frame}"))),
            Err(err) => {
                self.kill().await;
                Err(self.broken(&format!("did not become ready: {err}")))
            }
        }
    }

    async fn send_frame(&mut self, frame: Value) -> Result<()> {
        let handle = self
            .child
            .as_mut()
            .ok_or_else(|| TinyAgentsError::Capability("rlm interpreter not running".into()))?;
        let mut line = frame.to_string();
        line.push('\n');
        if handle.stdin.write_all(line.as_bytes()).await.is_err() {
            return Err(self.broken("stdin closed"));
        }
        let _ = handle.stdin.flush().await;
        Ok(())
    }

    async fn read_frame(&mut self, deadline: Instant) -> Result<Value> {
        let handle = self
            .child
            .as_mut()
            .ok_or_else(|| TinyAgentsError::Capability("rlm interpreter not running".into()))?;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(TinyAgentsError::Timeout(
                    "rlm cell exceeded its wall-clock timeout".to_string(),
                ));
            }
            let mut line = String::new();
            let read = tokio::time::timeout(remaining, handle.stdout.read_line(&mut line)).await;
            match read {
                Err(_) => {
                    return Err(TinyAgentsError::Timeout(
                        "rlm cell exceeded its wall-clock timeout".to_string(),
                    ));
                }
                Ok(Err(err)) => return Err(self.broken(&format!("stdout read failed: {err}"))),
                Ok(Ok(0)) => return Err(self.broken("exited unexpectedly")),
                Ok(Ok(_)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(frame) => return Ok(frame),
                        // Non-protocol noise on stdout (a library printing
                        // directly to fd 1) is ignored rather than fatal.
                        Err(_) => continue,
                    }
                }
            }
        }
    }

    async fn kill(&mut self) {
        if let Some(mut handle) = self.child.take() {
            let _ = handle.child.kill().await;
        }
    }
}

#[async_trait]
impl RlmInterpreter for ExternalInterpreter {
    fn language(&self) -> &str {
        &self.language
    }

    fn usage_guide(&self) -> String {
        match self.language.as_str() {
            "javascript" => r#"Write JavaScript (Node vm context; globals persist across cells). Host functions:
- llm("prompt")  or  llm({model, prompt, system})   -> string  // ask a sub-LLM
- tool("name", {arg: value, ...})                   -> result  // call a registered tool
- agent("name", "input")                            -> string  // delegate to a sub-agent
- final_answer("...")                                          // end the task with this answer
- console.log(x)                                               // observe a value next turn
Capability failures throw RlmError; catch them with try/catch. The completion
value of the cell (its last expression) is echoed back to you."#
                .to_string(),
            _ => r#"Write Python (a persistent exec namespace; globals survive across cells). Host functions:
- llm("prompt")  or  llm(prompt, model=..., system=...)  -> str   # ask a sub-LLM
- tool("name", {"arg": value, ...})                      -> result # call a registered tool
- agent("name", "input")                                 -> str    # delegate to a sub-agent
- final_answer("...")                                              # end the task with this answer
- print(x)                                                         # observe a value next turn
Capability failures raise RlmError; catch them with try/except. If the last
statement of a cell is an expression, its value is echoed back to you."#
                .to_string(),
        }
    }

    async fn set_variable(&mut self, name: &str, value: Value) -> Result<()> {
        self.ensure_started().await?;
        self.send_frame(json!({"op": "set_var", "name": name, "value": value}))
            .await?;
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            let frame = self.read_frame(deadline).await?;
            if frame.get("op").and_then(Value::as_str) == Some("var_set") {
                return Ok(());
            }
        }
    }

    async fn eval_cell(&mut self, code: &str, host: Arc<dyn RlmHostApi>) -> Result<CellEval> {
        self.ensure_started().await?;
        self.send_frame(json!({"op": "eval", "code": code})).await?;
        let deadline = host
            .deadline()
            .unwrap_or_else(|| Instant::now() + DEFAULT_EXCHANGE_TIMEOUT);

        loop {
            let frame = match self.read_frame(deadline).await {
                Ok(frame) => frame,
                Err(err) => {
                    // Fail closed: a cell that timed out (or broke the
                    // protocol) leaves the child in an unknown state.
                    self.kill().await;
                    return Err(err);
                }
            };
            match frame.get("op").and_then(Value::as_str) {
                Some("call") => {
                    let call: HostCall =
                        match serde_json::from_value(frame.get("call").cloned().unwrap_or_default())
                        {
                            Ok(call) => call,
                            Err(err) => {
                                self.send_frame(json!({
                                    "op": "call_result",
                                    "ok": false,
                                    "error": format!("malformed capability call: {err}"),
                                }))
                                .await?;
                                continue;
                            }
                        };
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    let outcome = tokio::time::timeout(remaining, host.handle(call)).await;
                    match outcome {
                        Err(_) => {
                            self.kill().await;
                            return Err(TinyAgentsError::Timeout(
                                "rlm cell exceeded its wall-clock timeout".to_string(),
                            ));
                        }
                        Ok(Ok(value)) => {
                            self.send_frame(
                                json!({"op": "call_result", "ok": true, "value": value}),
                            )
                            .await?;
                        }
                        Ok(Err(err)) if is_fatal(&err) => {
                            // Policy bound tripped: kill the child rather than
                            // letting the script observe its own limits.
                            self.kill().await;
                            return Err(err);
                        }
                        Ok(Err(err)) => {
                            self.send_frame(json!({
                                "op": "call_result",
                                "ok": false,
                                "error": err.to_string(),
                            }))
                            .await?;
                        }
                    }
                }
                Some("result") => {
                    let value = frame.get("value").cloned().unwrap_or(Value::Null);
                    return Ok(CellEval {
                        stdout: frame
                            .get("stdout")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        value: (!value.is_null()).then_some(value),
                        error: frame
                            .get("error")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    });
                }
                _ => continue,
            }
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        if self.child.is_some() {
            let _ = self.send_frame(json!({"op": "shutdown"})).await;
            self.kill().await;
        }
        Ok(())
    }
}

impl Drop for ExternalInterpreter {
    fn drop(&mut self) {
        // `kill_on_drop(true)` reaps the child if shutdown was never called.
        self.child = None;
    }
}
