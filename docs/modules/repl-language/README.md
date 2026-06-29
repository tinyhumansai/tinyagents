# REPL Language Module Specification

The REPL language is an interactive orchestration layer for RustAgents. It is
inspired by Recursive Language Models (`rlm`) and CodeAct-style agents, where a
model can write small programs, inspect their output, call sub-models, and
iterate until it has a final answer.

This module is separate from the expressive language:

- the expressive language (`.rag`) is a declarative graph definition format
- the REPL language (`.ragsh`) is an imperative session language for inspecting,
  scripting, and recursively orchestrating harness and graph runs

Both layers compile or lower into the same harness and graph runtime. Neither
layer should bypass the model registry, tool registry, graph registry, event
system, recursion policy, or run limits.

## Detailed Module Docs

- [Design](design.md)

## Responsibilities

- Provide an interactive session runtime over harness and graph primitives.
- Execute small scripts with a persistent namespace.
- Expose registered models, agents, graphs, tools, stores, and context as
  capability-bound functions.
- Let sessions draft, validate, inspect, diff, compile, and optionally register
  graph blueprints through the expressive-language compiler.
- Support model-driven CodeAct loops where model output contains fenced REPL
  blocks.
- Capture stdout, return values, state changes, model calls, tool calls, graph
  calls, errors, and final answers as typed events.
- Support recursive sub-model, sub-agent, and sub-graph calls with depth
  tracking.
- Support batched model, agent, and graph calls with bounded concurrency.
- Preserve source spans and session history for diagnostics and replay.
- Provide deterministic test utilities for scripted sessions.

## Non-Responsibilities

- It is not a replacement for the declarative graph language.
- It is not a general-purpose unsafe host-code execution layer.
- It does not provide direct filesystem, network, environment variable, or
  process access.
- It does not own model provider logic.
- It does not own graph topology or checkpointing.
- It does not allow scripts to call unregistered tools or models.
- It does not install model-generated graph topology directly into the runtime;
  generated graphs must pass through the `.rag` compiler and policy checks.

## Recommended Direction

Use Rhai for the first in-process REPL runtime and document Python as a future
out-of-process compatibility sandbox. Rhai gives RustAgents a Rust-native,
capability-bound embedding surface, while Python remains useful for training and
RLM-compatible workflows where the sandbox boundary is explicit.

Recommended extension: `.ragsh`.
