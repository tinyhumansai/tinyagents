# harness::prompt

Prompt assembly: turns templates, runtime values, and cache-segment metadata
into the `ModelRequest` sent to a model.

In the recursive harness a prompt is data that is built, not a hard-coded
string — the same `PromptBuilder`/`PromptTemplate` machinery a top-level agent
uses to assemble its own request is available to construct the prompt for a
nested sub-agent or sub-model call. Segment-aware assembly is what keeps a
deep recursion's repeated calls cache-friendly: the stable (system/tools/
instructions) prefix stays byte-identical across calls so a provider's
KV-cache/prompt-cache keeps hitting.

## Public surface

- `PromptTemplate` — a `{name}`-substitution string template (`{{`/`}}` are
  escaped braces). `render`/`render_message`/`render_system`/`render_user`/
  `render_assistant` render it, failing with
  `TinyAgentsError::Validation` on an unknown or unclosed placeholder.
- `TemplateRole` — `System` / `User` / `Assistant`, selects the `Message`
  variant `PromptTemplate::render_message` produces.
- `MessagesTemplate` — an ordered sequence of `(TemplateRole, PromptTemplate)`
  pairs rendered together into a `Vec<Message>`.
- `PromptBuilder` — the main entry point. `push_system` /
  `push_system_messages` / `push_tools_segment`
  / `push_instructions` append **cacheable** segments; `push_history` /
  `push_volatile` append **non-cacheable** ones. `build(tail)` finalizes a
  `ModelRequest`, appending `tail` last (the current user turn) and stamping
  `ModelRequest::prompt_fingerprint` from `PromptBuilder::fingerprint` — a
  stable SHA-256 over the cacheable prefix's JSON shape.
- `PromptSection` / `PromptAssembly` / `PromptTruncation` / `PromptBudget` —
  a smaller, behavior-free assembly path (`assemble_sections`,
  `assemble_sections_with_budget`) for composing already-rendered text blocks
  under a byte and/or token cap, tracking which section (if any) was
  truncated. Independent of `PromptBuilder`/cache segments; used for context
  composition (e.g. retrieved-document catalogues) rather than the top-level
  model request.
- `PromptTier`, `DynamicPromptSection<C>`, `PromptComposer<C>`, and
  `TieredPrompt` — context-generic rendering for sections that return stable,
  session-context, or volatile slices. The composer renders each section once,
  groups output by tier while keeping declaration order within each tier, and
  reports byte offsets for reusable prefix boundaries. Pass the result to
  `PromptBuilder::push_tiered_system_prompt` to mark stable/context segments
  cacheable and the volatile tail non-cacheable.
- Free rendering helpers: `render_heading`, `render_optional_section`, and
  `render_tool_catalogue`.
- Model-guidance helpers: `needs_execution_discipline`,
  `execution_discipline_for`, and `execution_discipline_for_profile` select the
  exported `EXECUTION_DISCIPLINE` block by model family, falling back from an
  unknown or blank model alias to its provider while explicitly excluding
  Claude and Gemini families.
- Workspace prompt files (`workspace_files`): `sync_workspace_file` seeds a
  host's default file into a workspace and refreshes it when the default
  changes unless the user edited it, tracked by a SHA-256 sidecar named by
  `builtin_hash_sidecar_name` (`.{filename}.builtin-hash`, contents
  `builtin_hash`). `inject_inline_content` and `inject_workspace_file_capped`
  render content under a `### {label}` heading, capped at a character budget
  with a truncation marker naming the host's read-file tool. The host passes
  the default contents, the cap, and the read-tool name.
- System-segment helpers: `SYSTEM_SEGMENT_ID`, `system_segment_id`, and
  `is_system_segment_id` define the canonical `system`, `system.1`, ... IDs
  used for one cacheable segment per leading system-message tier.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: templates, `PromptBuilder` (and its private `BuiltSegment`), section assembly, and tiered dynamic prompt composition. |
| `workspace_files.rs` | Workspace prompt-file seeding (`sync_workspace_file`, SHA-256 builtin-hash sidecar) and capped `###`-heading injection. |
| `workspace_files_tests.rs` | Unit tests for the sidecar format, upgrade/edit/legacy-sidecar cases, and byte-exact injection and truncation. |
| `model_guidance.rs` | Model-family matching and the optional execution-discipline prompt block. |
| `mod.rs` | Behavioral code: the `{name}` template renderer, `PromptBuilder` methods (segment pushes, `build`, `fingerprint`), and the section-assembly free functions. |
| `mod_tests.rs` | Unit tests for placeholder substitution/escaping, error cases, per-role rendering, `MessagesTemplate` ordering, and `PromptBuilder` segment cacheability/fingerprinting. |

## Key invariants

- **Cache-segment ordering matters.** Callers should push segments in the
  order system → tools → instructions (all cacheable) → history → volatile
  (not cacheable), so the stable prefix stays at the head of the message list
  and providers can apply KV-cache reuse. `PromptBuilder` does not enforce
  this ordering itself — it is a convention the push methods are named to
  encourage.
- **Prompt tiers are ordered metadata.** `PromptComposer` emits the stable
  prefix first, then context, then volatile content. `push_tiered_system_prompt`
  preserves each tier as a distinct system segment with a `prompt-tier:*` ID;
  context remains cacheable and volatile content is marked non-cacheable.
- **Tiered system IDs are canonical and append-safe.**
  `push_system_messages` assigns one cacheable segment per message and
  continues numbering across repeated calls. Recognition rejects zero and
  leading-zero suffixes so middleware-owned layouts are not mistaken for the
  harness layout.
- **The fingerprint only covers cacheable segments and tool schemas.** It is a
  SHA-256 over serde's deterministic (sorted-key) JSON serialization, so it is
  stable across processes, Rust versions, and platforms — safe to persist and
  compare across restarts. It changes on any edit to a cacheable segment's
  content/role/id or the tool schema list, including changes that leave the
  segment's rendered text unchanged (e.g. a role swap).
- **`assemble_sections`/`assemble_sections_with_budget` are independent of
  cache segments.** They operate on plain, already-rendered `PromptSection`
  text and stop at the first section that does not fit the budget — later
  sections are omitted entirely, not partially included.

## Relation to neighbouring modules

- Consumed by the agent loop and by nodes/subagents that build their own
  `ModelRequest` before a model call.
- `PromptBuilder::build` produces a `tinyinference_llm::model::ModelRequest`
  with `cache_segments` populated; downstream, `crate::cache` and
  `crate::middleware`'s `PromptCacheGuardMiddleware` read those segments to
  detect prefix invalidation across calls.
