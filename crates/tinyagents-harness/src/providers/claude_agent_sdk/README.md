# claude_agent_sdk

`ChatModel<()>` implementation that drives Anthropic's `claude` CLI as a
one-shot, stateless subprocess: `claude -p --model <model> --output-format
stream-json --no-color`, reading the request off stdin and the response off
stdout. Unlike [`../claude_code`](../claude_code/README.md), which resumes a
CLI-side session and lets the CLI run its own agentic loop, this provider
starts a fresh process for every [`ClaudeAgentSdkProvider::invoke`] call and
never lets the CLI decide anything beyond producing one reply — it is a
plain prompt-guided chat model, not an agent driver.

## File map

| File | Role |
| --- | --- |
| `mod.rs` | `ClaudeAgentSdkConfig`, `ClaudeAgentSdkProvider` (`ChatModel<()>` impl), CLI argument/stdin building, the subprocess spawn/read/timeout loop, and system-prompt/transcript rendering. |
| `protocol.rs` | `SdkMessage`/`SdkError`: the `serde` shapes for the `--output-format stream-json` NDJSON lines the CLI writes to stdout. |
| `mod_tests.rs` | Unit tests for construction, invocation building, transcript rendering, and response assembly. |
| `mod_serde_tests.rs` | `ClaudeAgentSdkConfig` TOML/JSON and JSON-schema parity with OpenHuman's former mirror type. |

## Request shaping

Because each invocation is a brand-new process, the full non-system
transcript has to be replayed on every call (`render_transcript`): a single
user turn is sent as-is, but a multi-turn transcript is labelled
(`[USER]`/`[ASSISTANT]`/`[TOOL]`) so the model can tell its own prior output
from the newest user turn. All system messages are concatenated into one
`[SYSTEM]...[/SYSTEM]` block prepended to stdin — the CLI only exposes a
single system prompt, and taking only the first system message (rather than
joining all of them) used to silently drop system content injected by
harness middleware (compression notices, artifact listings, turn-cap
wrap-up) that relies on being carried as a system message specifically
because that is the one thing trimming and compression will not remove.

## Response assembly

`invoke_cli` spawns the process with the request on stdin, drains stderr on
a concurrent task (to avoid a pipe-buffer stall), and decodes each stdout
line as an [`protocol::SdkMessage`]. `Text` chunks accumulate as a fallback;
the terminal `Result` message's text is preferred when present. Reading
stdout is bounded by a 120s timeout and waiting for process exit by a
separate 30s timeout; either firing kills the child and surfaces an error
rather than hanging the caller.

## Tool calling

`ModelProfile` carries no native tool-calling flag override here — when the
request declares tools, `mod.rs` runs the shared prompt-tool
instructions/coalescing (`tinyinference_llm::prompt_tools::with_tool_instructions`,
`tinyinference_llm::prompt_tools::coalesce_tool_results`) on the way in and
`tinyinference_llm::prompt_tools::recover_tool_calls` on the way out — the
`tinytools-agent` protocol used everywhere in the harness for models without
native tool support.

## Selection

Not registered automatically; construct a `ClaudeAgentSdkProvider` directly
(`::new` defaults to `config.default_model`, `::for_model` pins a specific
model) and pass it to the harness like any other `ChatModel`.
`ClaudeAgentSdkConfig::enabled` is a plain data field for the embedding
application's own gating — this crate does not read it.

`ClaudeAgentSdkConfig` derives `Serialize`, `Deserialize` and
`schemars::JsonSchema`, so a host can embed it directly as a config section
(OpenHuman's `[claude_agent_sdk]`). Every field is optional on input and takes
the `Default` value (`enabled = false`, `binary = "claude"`,
`default_model = "claude-sonnet-4-6"`, no budget); all four fields are
serialized, with a `None` budget written as JSON `null` and left out of TOML.

Doctor may construct `ClaudeAgentSdkProvider::for_inspection` only with an
explicit API key or bearer token captured by its host. A gateway bearer is
passed through `ANTHROPIC_AUTH_TOKEN`; it does not require `ANTHROPIC_API_KEY`.
This is distinct from OAuth state in a user's Claude Code profile, which the
inspection path cannot freeze and intentionally does not read. This path launches the CLI
with `--bare`, an empty system prompt, `--tools ""`,
`--no-session-persistence`, and one allowed turn. It sets
`CLAUDE_CODE_MAX_OUTPUT_TOKENS` from the request, capped at five, uses a
temporary HOME, and passes only the frozen explicit credential and optional
base URL. It rejects tool declarations and multi-message histories before
spawning. Ordinary `new` / `for_model` calls retain their existing CLI
configuration and behavior.
