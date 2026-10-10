# claude_code

`ChatModel<()>` implementation that drives Anthropic's `claude` CLI as a
long-lived, session-resuming subprocess instead of calling a hosted chat API
directly. Unlike [`../claude_agent_sdk`](../claude_agent_sdk/README.md),
which starts a fresh stateless process per call, this provider reuses a CC
(`claude` CLI) session across a conversation (`--session-id`/`--resume`) and
lets the CLI run its own built-in tools internally in its own agentic loop —
this harness never sees those tool calls (see [Tool calls are not
surfaced](#tool-calls-are-not-surfaced)).

## File map

| File | Role |
| --- | --- |
| `mod.rs` | `ClaudeCodeProvider` (`ChatModel<()>` impl), `PROVIDER_PREFIX`, `MAX_CONCURRENT_TURNS`, `render_request_stdin`, request/response conversion, per-thread concurrency control in `run_chat`. |
| `auth.rs` | `resolve()` → `(AuthSource, Option<key>)`: which credential the spawned CLI will use. |
| `auth_status.rs` | `probe()` — the CLI's own auth state (API key env / subscription login / signed out / unknown), via `claude auth status --json`. |
| `driver.rs` | `run_turn`: per-turn scratch dir, permission posture, macOS Seatbelt jail, MCP config, argv, stdin/stdout piping, `DEFAULT_TURN_TIMEOUT_SECS` (900, override `OPENHUMAN_CLAUDE_CODE_TURN_TIMEOUT_SECS`), stderr diagnostics cap. |
| `event_mapper.rs` | `ClaudeCodeEvent` → `ProviderDelta` / aggregated `ChatResponse`. `tool_use` blocks are tracked only to keep their `input_json_delta`s out of the visible text and are **not** surfaced as tool calls: the CLI is self-executing, so a tool block in the stream has already run; surfacing it would make the harness try to dispatch `Read`/`Bash`/… and loop on "unknown tool". |
| `input_builder.rs` | `build_stdin`: JSONL user turns for `--input-format stream-json` — full history folded into a text preamble on a new session, only the pending user turn(s) on `--resume`; inline images re-hydrated from `[IMAGE:...]`/`[OH_IMAGE:...]` markers (5 MiB cap). |
| `session_store.rs` | `SessionStore`: thread key → CC session UUID v4, persisted in `<workspace_dir>/claude-code-sessions.json`. |
| `settings.rs` | `ClaudeCodeSettings { full_access }` persisted in `<workspace_dir>/claude_code_settings.json`. |
| `stream_parser.rs` | Line-buffered JSONL parser for `--output-format stream-json`; permissive `serde_json::Value` payloads so a minor CLI schema bump does not break parsing. |
| `types.rs` | `MIN_CLI_VERSION` (`2.0.0`), `CliStatus`, `BRAND_LABEL`. |
| `version_check.rs` | `resolve_binary` (`OPENHUMAN_CLAUDE_CLI` env → `PATH` → well-known install dirs, because GUI launches inherit a stripped launchd `PATH`) and `probe()` against `MIN_CLI_VERSION`. |
| `bridge.rs` | Crate-private `ChatMessage`/`ChatResponse`/`UsageInfo`/`ProviderDelta` shapes passed between the modules above, independent of `tinyinference_llm`'s wire types. |

Per-file `*_tests.rs` sit alongside each module (`auth_tests.rs`,
`auth_status_tests.rs`, `driver_tests.rs`, `event_mapper_tests.rs`,
`input_builder_tests.rs`, `session_store_tests.rs`, `settings_tests.rs`,
`stream_parser_tests.rs`, `version_check_tests.rs`), plus `mod_tests.rs` for
thread-key resolution and `ModelProfile` construction, and
`pipeline_tests.rs` for an end-to-end replay of a captured CC 2.x transcript.

## CLI invocation

`ClaudeCodeProvider::run_chat` acquires one of `MAX_CONCURRENT_TURNS` (4)
`Semaphore` permits, plus a per-`thread_id` mutex so two overlapping calls
for the same conversation cannot race on the same session UUID, then
`driver::run_turn` spawns:

```text
claude -p --input-format stream-json --output-format stream-json --verbose
  --include-partial-messages --add-dir <project_dir>
  --permission-mode <acceptEdits|bypassPermissions>
  (--session-id <uuid>|--resume <uuid>|--no-session-persistence) --model <model>
```

plus `--append-system-prompt-file <scratch>/append-system-prompt.txt` when a
system prompt is present (kept off argv so a large harness prompt does not
hit Windows' argv length limit), `--mcp-config <scratch>/openhuman-mcp-config.json
--strict-mcp-config` when a host-supplied `McpEndpointProvider` resolved an
endpoint, and `--disallowedTools Bash,BashOutput,KillShell,WebFetch,WebSearch,Task`
unless full access is on (see [Permission posture](#permission-posture-sandbox-and-mcp)).
Requests with a caller-supplied conversation identity use `--session-id` on a
new CC session and `--resume` afterwards; the UUID comes from
`session_store.rs`. Requests without an explicit identity use
`--no-session-persistence`, so one-shot inference calls cannot create source
transcripts. `cwd` is `project_dir` — the caller's
project root, not this provider's own `workspace_dir` — so CC's file tools
act on the user's code.

The thread key that selects a session comes from `thread_key_from_request`
in `mod.rs`: it reads `metadata.thread_id` / `conversation_id` / `session_id`
off the `ModelRequest`, then `continuation_id`, and falls back to a fresh
`ephemeral_<uuid>` (an isolated, non-resumable session) when the request
carries no conversation identity at all — a request without an id never
risks resuming the wrong conversation.

## Auth resolution order

1. Process env `ANTHROPIC_API_KEY` (highest precedence) — `auth.rs::resolve`
   returns `AuthSource::EnvApiKey` and the key is set on the child at spawn.
2. Otherwise `AuthSource::CliCredentials`: the env var is *not* set and the
   CLI uses its own login (`~/.claude/.credentials.json`, or the Keychain on
   macOS).
3. `auth_status.rs` reports the richer state for a settings UI by spawning
   `claude auth status --json` (bounded by `AUTH_STATUS_TIMEOUT` = 10s)
   rather than reading the credentials file, because on macOS the CLI stores
   credentials in the Keychain (service `Claude Code-credentials`) — a
   logged-in macOS user would otherwise be misreported as signed out. Older
   CLIs without `auth status` map to `AuthSource::Unknown`, never to
   "signed out".
4. Picking up an Anthropic key from a host-provided auth-profile store and
   Claude Pro/Max OAuth are both future work (see `auth.rs`).

Tests that touch `ANTHROPIC_API_KEY` / `OPENHUMAN_CLAUDE_CODE_*` serialize
through `mod.rs::ENV_TEST_LOCK`.

## Permission posture, sandbox, and MCP

The user opts into Claude Code explicitly, so its toolset is not restricted
beyond a permission posture that is the user's choice
(`driver.rs::claude_code_full_access`):

- default: `--permission-mode acceptEdits` plus `--disallowedTools` for
  shell, network, and `Task` fan-out — file reads/edits in `project_dir`
  only;
- full access (`settings.rs` toggle, or `OPENHUMAN_CLAUDE_CODE_PERMISSION_MODE=
  bypass|bypassPermissions|full`): `--permission-mode bypassPermissions` and
  the entire CC toolset including Bash.

On macOS the spawn is additionally wrapped in a Seatbelt jail
(`sandbox-exec -p <profile>`, on by default when `/usr/bin/sandbox-exec`
exists; opt out with `OPENHUMAN_CLAUDE_CODE_SANDBOX=0`). The profile allows
everything the user can do *except* reads and writes under the first
`.openhuman*`-named ancestor directory of `workspace_dir` — this provider's
own on-disk state (session store, settings), not the CLI's own tools. Linux
and Windows have no OS wall yet; CC runs unconfined there.

A host can attach an MCP endpoint by implementing `driver::McpEndpointProvider`
and calling `ClaudeCodeProvider::with_mcp_provider`. When present,
`driver::run_turn` writes a per-turn `--mcp-config` pointing at the
returned loopback HTTP endpoint, with a bearer token carried in the config's
`Authorization` header. If the provider's `endpoint()` call fails, CC runs
without that MCP server and the turn still proceeds.

## Tool calls are not surfaced

`event_mapper.rs` tracks `tool_use` content blocks only long enough to keep
their argument-delta JSON out of the visible text; it never emits them as
harness tool calls. The CLI has already executed the tool by the time its
`tool_use` block appears in the stream — surfacing it would make the harness
try to dispatch a tool it does not own (`Read`, `Bash`, …), fail every time,
and loop until the harness's own failure circuit breaker aborts the run.
This provider therefore advertises `tool_calling: false` in its
`ModelProfile` and behaves as a chat model whose tool use is entirely
internal to the CLI process.

## Selection

Not registered automatically by this crate. A host constructs a provider
directly with `ClaudeCodeProvider::from_env(model, workspace_dir,
project_dir)` — which fails fast with an actionable error when the CLI is
missing, outdated (`types::MIN_CLI_VERSION`), or otherwise unusable
(`version_check::probe`) — or with `ClaudeCodeProvider::new` when the caller
has already resolved the binary path and credentials itself.
`PROVIDER_PREFIX` (`"claude-code:"`) is exported for a host's own routing
grammar to recognize a `claude-code:<model>` provider string; this crate
does not interpret it itself.
