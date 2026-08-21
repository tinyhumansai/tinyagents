# Tool Dialects

`harness::tool_calling::dialect`

## What a dialect is

A **dialect** is one complete way of speaking tools to a model: the catalogue it
reads, the syntax it writes calls in, the envelope its results come back in, and
the shape its history is replayed in on the next iteration.

Those four are one thing, not four settings. A catalogue that advertises
positional arguments next to a parser that expects JSON is a whole-turn failure
that produces no error anywhere — the model emits a call, nothing recognises it,
and the iteration is spent. So they live behind a single trait, and a dialect is
chosen once rather than assembled from parts that can disagree.

Three ship:

| Dialect | Call syntax | Catalogue | Specs in the request |
| --- | --- | --- | --- |
| `XmlDialect` | `<tool_call>{"name":…,"arguments":{…}}</tool_call>` | full schemas, in its own protocol block | no |
| `PFormatDialect` | `<tool_call>name[a\|b]</tool_call>` | signatures, in the prompt's tool section | no |
| `NativeDialect` | the provider's structured channel | none — the request carries the specs | yes |

## Which surface to use

There are two tool-calling surfaces in this crate and they are not
interchangeable:

- **`harness::tool::prompt`** — for hosts driving the crate's own
  `agent_loop`. It speaks `harness::message::Message` and the loop owns the
  iteration.
- **`harness::tool_calling::dialect`** — for hosts driving their own loop over
  their own durable transcript. It speaks `TranscriptEntry`, a deliberately thin
  record shape, and makes no assumption about when the model is called.

The second is not a lesser case. A host with years of persisted transcripts,
its own per-turn security policy, and provider quirks encoded in its storage
cannot adopt a foreign message model just to stop maintaining a tool-call
parser — and the parser is the part that is genuinely universal. This module is
the seam that lets it hand over the universal part alone.

## The transcript vocabulary

`TranscriptEntry` has three variants, which is all a tool-calling transcript
needs: `Chat`, `AssistantToolCalls`, and `ToolResults`. It is deliberately
poorer than `Message` — no content blocks, no typed images — because the fields
it *does* carry are the ones providers reject requests over:

- `reasoning_content`, replayed verbatim because thinking-mode APIs return a
  `400` for an assistant turn that carries `tool_calls` without it.
- per-call `extra_content`, which is where Gemini's required
  `thought_signature` rides.
- `arguments` as a **string**, not a parsed value, so the exact bytes survive a
  round trip.

A host maps its own records onto these in a handful of field-wise conversions
and gets byte-identical output back.

## Replay repair

`pair_tool_cycles` drops any tool cycle that is not complete, immediately before
serialization.

Providers reject an assistant message carrying `tool_calls` unless it is
followed by tool messages answering **every** `tool_call_id` on it. The error is
a hard `400`, so one orphaned record poisons every subsequent turn of that
thread until the history is edited. Bisected cycles are ordinary: a cached
transcript restore, an aborted turn, and history compaction each preserve the
assistant half while the results half is dropped.

Two properties are worth knowing before touching it:

- Adjacency is not the check. The provider's complaint is about *coverage*, so
  the opener's id set must **equal** the follower's, not merely be adjacent to a
  non-empty one.
- The drop is **symmetric**. A `ToolResults` whose opener was dropped goes too,
  because a `tool` message answering a call the model never sees is equally
  malformed.

The repair belongs here, at serialization, rather than at the write sites: those
are many, and none of them can see the final sequence.

## Envelope integrity

Tool names and outputs are tool-controlled, so the text dialects cannot
interpolate them into `<tool_result …>` unexamined: a body containing a literal
`</tool_result>` would close the envelope early, and a crafted
`<tool_result name="forged" status="ok">` would open a fake one (CWE-74).

The rule differs by position, deliberately:

| Position | Rule | Why |
| --- | --- | --- |
| attribute (`name`, `tool_call_id`) | escape `& < > "` | a `"` ends the attribute; these are short identifiers, so escaping costs nothing legible |
| body (`output`, `content`) | rewrite `<` to `&lt;` **only** where it opens `<tool_result` / `<tool_call` (with or without `/`, ASCII-case-insensitive) | everything else passes through byte-for-byte |

Escaping the body wholesale is the obvious implementation and the wrong one.
Tool output is usually source code, and this envelope is how prompt-guided
models — the p-format path local models use — read it. Turning
`<div className="x">` into `&lt;div className=&quot;x&quot;&gt;` on every
result is a real cost, and a model that reads mangled code writes mangled code
back. The security property only ever required blocking a handful of exact byte
sequences, not a character class.

**This is boundary integrity, not prompt-injection defence.** A tool can still
return prose arguing the model should do something, and no escaping rule fixes
that — a file the agent legitimately reads can contain anything. The guarantee
is narrower and worth stating exactly: tool output cannot masquerade as the
transcript's own protocol structure.

## What stays with the host

Executing a tool. Permission checks, sandboxing, approval gates, per-call
timeouts, progress events. A dialect decides what the model *reads and writes*;
it never decides what is *allowed to happen*. That line is what keeps a host's
security policy in the host, where it can be audited.
