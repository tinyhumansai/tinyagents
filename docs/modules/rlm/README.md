# RLM — the recursive-language-model runtime (`src/rlm/`)

The `rlm` module (Cargo feature `rlm`) turns a code sandbox into a
recursive-language-model harness: a **driver model** writes code cells, the
cells execute in a **sandboxed interpreter**, and the only host surface the
scripts see is a set of **capability calls back into the registry** — sub-LLM
queries, tools, and sub-agent delegation. Because scripts can call models
(and agents that call models), the loop is recursive by construction, which
is the execution pattern of the RLM literature this crate is architected
around (see the crate-level docs in `src/lib.rs`).

```text
             ┌──────────────────────────────────────────────┐
   task ───► │ RlmRunner: driver model ⇄ code cells ⇄ obs.  │ ───► answer
             └──────────────┬───────────────────────────────┘
                            │ eval(code)
             ┌──────────────▼───────────────┐
             │ RlmSession                   │
             │  ┌──────────┐   HostCall     │
             │  │ RlmInter-│ ─────────────► │  RlmHost ──► CapabilityRegistry
             │  │ preter   │ ◄───────────── │   (policy, counters, depth)
             │  └──────────┘   Value/error  │     ├─ llm    → ChatModel::invoke
             └──────────────────────────────┘     ├─ tool   → Tool::call
                                                  └─ agent  → HarnessAgent::run
```

## The three layers

| Layer | Type | What it is |
|---|---|---|
| Interpreter | `RlmInterpreter` (trait) | Pluggable cell execution: `eval_cell`, `set_variable`, `usage_guide`, `shutdown`. State persists across cells like a notebook. |
| Session | `RlmSession` | One interpreter bound to one `RlmHost`; the programmatic "interpreter as an API" surface. Enforces per-cell policy fail-closed. |
| Runner | `RlmRunner` | The model-driven loop: template → system prompt, fenced code cells in, observations out, stop on `final_answer(...)`. |

Every layer is usable on its own: an embedder that wants its own loop (or a
notebook UI) drives `RlmSession::eval` directly and never touches the runner.

## Interpreter backends

- **`InterpreterSpec::Rhai`** (default) — the embedded Rhai engine. This is
  the only *hermetic* sandbox: no filesystem, network, or process access
  exists inside the engine; the registered capability closures are its whole
  world. Bounded by `max_operations`, the cell deadline (`on_progress`
  hook), and the blocking bridge around every capability call (the same
  fail-closed adapter as `repl::session`).
- **`InterpreterSpec::Python { binary, args }`** — an external CPython child
  process (`python3` by default). A bootstrap prelude is injected via `-c`;
  cells run in a persistent exec namespace.
- **`InterpreterSpec::Javascript { binary, args }`** — a Node.js child
  (`node` by default, prelude via `-e`); cells run in a persistent `vm`
  context.
- **`InterpreterSpec::Command { binary, args }`** — any command that speaks
  the wire protocol itself (a containerized runner, a jailed interpreter, a
  different language).

### The wire protocol (external backends)

Line-delimited JSON on the child's stdin/stdout; calls are strictly
sequential so no correlation ids are needed. Child → host: `ready`,
`call {call: HostCall}`, `result {stdout, value, error}`, `var_set`.
Host → child: `eval {code}`, `set_var {name, value}`,
`call_result {ok, value|error}`, `shutdown`. `HostCall` is the shared,
serde-stable call shape (`{"capability": "llm"|"tool"|"agent"|"final_answer", ...}`).
Any runtime that speaks this protocol is a valid backend — that is the
extension point for other harnesses (e.g. openhuman) to bring their own
interpreter.

### Sandboxing honesty

The host enforces every policy limit fail-closed for **all** backends (a
child that exceeds its deadline or trips a bound is killed, not asked). But
an external child process has whatever OS access the environment grants it.
For untrusted driver models, either stay on the embedded Rhai backend or run
the external interpreter inside real isolation (container/jail/seccomp) via
`InterpreterSpec::Command`.

## The capability surface inside scripts

Identical semantics across languages (spelling varies per usage guide):

- `llm(prompt)` / `llm({model, prompt, system})` → string — a sub-LLM call;
  the unnamed form uses the session's default sub-model.
- `tool(name, args)` → tool result (raw JSON when the tool provides it).
  Arguments are validated against the tool's schema at the host boundary.
- `agent(name, input)` → string — delegates to a registered
  `HarnessAgent`; a full nested agent run with event fan-out and the shared
  recursion-depth guard (`RunConfig::checked_child_depth`).
- `final_answer(text)` — ends the run with this answer.
- `print(...)` / `console.log(...)` — captured and echoed back to the
  driver next turn, bounded by `max_output_bytes` (explicit truncation
  marker).

### Error contract

Script-visible failures (unknown tool, schema mismatch, tool error, provider
error) surface *inside* the script as catchable exceptions (`RlmError` in
Python/JS, `try`/`catch` in Rhai) so the driving model can adapt — that
feedback loop is the point of an RLM. Policy violations (`LimitExceeded`,
`Timeout`, `Cancelled`, `SubAgentDepth`) are **fatal**: the cell aborts (and
an external child is killed) so scripts can never observe and route around
their own resource limits. `rlm::is_fatal` is the classifier.

## Config-driven runs

Everything a run needs is one serde document — the integration surface for
external harnesses:

```json
{
  "interpreter": {"kind": "python", "binary": "/opt/venv/bin/python3"},
  "driver_model": "openai",
  "sub_model": "openai",
  "template": "context-explorer",
  "policy": {
    "max_cells": 8, "max_llm_calls": 16, "max_tool_calls": 64,
    "max_agent_calls": 8, "max_depth": 8,
    "max_script_bytes": 65536, "max_output_bytes": 262144,
    "cell_timeout": 90000, "max_operations": 5000000
  }
}
```

`RlmConfig::from_json` → `RlmRunner::from_config(config, registry, state)` →
`runner.set_context(json)` (optional) → `runner.run(task)`.

### Templates

`TemplateSpec::Named` selects a built-in; `TemplateSpec::Inline` carries a
custom `RlmTemplate` in the config document. Placeholders `{{language}}`,
`{{usage}}`, `{{capabilities}}`, `{{limits}}` are substituted at run time
from the live session (so the prompt always reflects the actual registry and
policy). Built-ins:

- `general` — solve the task with code; sub-LLMs for fuzzy subproblems.
- `context-explorer` — the RLM long-context pattern: material injected as
  the `context` variable, probed programmatically, never printed whole.
- `orchestrator` — decompose and delegate to registered sub-agents.

## Policy (all fail-closed)

`RlmPolicy`: `max_cells`, `max_script_bytes`, `max_output_bytes`,
`max_llm_calls`, `max_tool_calls`, `max_agent_calls`, `max_depth`,
`cell_timeout` (ms in JSON), `max_operations` (Rhai only). Counters are
session-cumulative. `RlmCancelFlag` provides sticky external cancellation,
observed mid-script (Rhai `on_progress`), mid-call (blocking bridge), and
before each cell.

## Runner loop details

- The driver must reply with exactly one fenced code block; the first fence
  is extracted regardless of its info string (models mislabel languages).
- A fence-less reply earns one nudge (it is often raw unfenced code); a
  second consecutive fence-less reply is accepted as a prose answer
  (`RlmStopReason::ModelAnswered`).
- Observations echo captured stdout, the cell value, and any script error.
- Stop reasons: `Answered` (a cell called `final_answer`), `ModelAnswered`,
  `CellBudgetExhausted`. The full per-cell trajectory is returned in
  `RlmOutcome::steps`.

## Tests and examples

- Unit: `src/rlm/test.rs` (config round-trips, templates, extraction, the
  embedded backend, runner loop against `ScriptedModel`).
- E2E: `tests/e2e_rlm.rs` — sub-agent delegation from scripts, real
  `python3`/`node` protocol round-trips (skipped when the binary is
  missing), fail-closed timeout kill, config-driven runner.
- Live: `tests/live_rlm.rs` — network-gated on `OPENAI_API_KEY`.
- Examples: `examples/rlm_rhai.rs` (context-explorer over expense records),
  `examples/rlm_python.rs` (tools + a real sub-agent, external Python).
  Run with `cargo run --features rlm --example rlm_rhai`.

## Relationship to `repl::session`

`repl::session` is the `.ragsh` interactive orchestration surface: Rhai
only, richer built-ins (graph authoring, batching), REPL-first. `rlm` is the
model-*driven* counterpart: interpreter-pluggable, config-first, and shaped
for embedding in other harnesses. They share design DNA deliberately — the
fail-closed blocking bridge, reserved capability boundary, and policy
posture are the same pattern.
