# result_expiry

Replaces old tool results in a run's stored transcript with a one-line note,
by age, whatever the context window has left.

## Why

A tool result is re-sent verbatim on every later call. Micro-compaction
(`MicrocompactMiddleware`) blanks old bodies only once the estimated tokens
nearly fill the window, and rewrites only the outgoing request. Age expiry acts
sooner and on the stored transcript.

## Surface

- `ToolResultExpiryPolicy` (`RunPolicy::result_expiry`): `enabled` (default
  `false`), `max_age_turns` (default 6), `exempt_tools` (default empty).
- `ExpiryOverrides` (`RunConfig::result_expiry`, `with_result_expiry_overrides`):
  per-session `Option` overrides of the three fields. Empty by default and not
  serialized when empty, so an older thread never changes until a host opts in.
- `expire_tool_results(messages, policy)`: the pure rewrite.

## Behaviour

- A result's age is the number of user messages after it. With
  `max_age_turns = 2`, a result with three user messages after it expires; one
  with two does not.
- The rewrite runs once, at the start of a run (the turn boundary), in place.
  The run's first model call, every later call, and the transcript the run
  returns all see the notes, so the cached prefix moves once per turn.
- The note is `[used <tool>: <first 120 chars on one line>]`, with ` …` before
  the bracket when cut. The format follows Chottu's idle-compaction note. The
  tool name comes from the assistant call the result answers (`tool` when none).
- Never rewritten: results marked `trusted_verbatim`, results of tools in
  `exempt_tools` (for example a collapsed `tool_search` result), and results
  that already are a note (idempotent). A tool message's `artifact` is kept.

## Composition

- **Micro-compaction** still blanks bodies near the window, with its own
  placeholder. It never sees an expired body, and a note is short, so it has
  less to reclaim. Run order is unchanged: expiry happens before the loop.
- **Compression/summarization** folds older history later; a note is what it
  summarizes if the result already expired.
- **TinyJuice** compacts a result when it is produced and keeps omitted content
  recoverable. An expired note carries no retrieval token and keeps only the
  first 120 characters, so a host that wants a result to stay retrievable
  lists its tool in `exempt_tools`.
