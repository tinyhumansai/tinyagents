//! Server-sent-events stream parsing and incremental accumulation
//! (`SseState`, `OpenAiStreamAcc`, `sse_next`).
//!
//! Split out of `openai/mod.rs`; see that module's doc comment for the
//! full provider overview.

use super::*;

/// In-progress reconstruction of a single tool call across streamed fragments.
#[derive(Clone, Debug, Default)]
pub(super) struct ToolCallBuild {
    /// Provider-assigned call id (filled from the first fragment carrying it).
    id: String,
    /// Function name (filled from the first fragment carrying it).
    name: String,
    /// Concatenated stringified-JSON argument fragments.
    args: String,
}

/// Provider-side accumulator that rebuilds the authoritative [`ModelResponse`]
/// from streamed chunks. Distinct from the generic
/// [`StreamAccumulator`][crate::harness::model::StreamAccumulator]: it tracks
/// tool-call names and ids (which the neutral deltas omit) so the terminal
/// [`ModelStreamItem::Completed`] carries a faithful response.
#[derive(Clone, Debug, Default)]
pub(super) struct OpenAiStreamAcc {
    id: Option<String>,
    text: String,
    /// Accumulated **side-channel** reasoning/thinking fragments
    /// (`reasoning_content` / `reasoning`), preserved on the final message as a
    /// leading [`ContentBlock::Thinking`] block (unsigned — the
    /// OpenAI-compatible path has no provider signature). Inline `<think>` tag
    /// reasoning is not accumulated here; it is recomputed from `text` in
    /// [`into_response`](Self::into_response) so the terminal response matches
    /// the non-streaming path exactly.
    reasoning: String,
    tool_calls: Vec<ToolCallBuild>,
    usage: Option<Usage>,
    finish_reason: Option<String>,
    /// Inline `<think>` extraction config (`None` disables it). Applied to the
    /// terminal response via [`extract_reasoning`].
    reasoning_tags: Option<ReasoningTagExtraction>,
    /// Live per-delta inline-tag splitter, so streamed content deltas route
    /// chain-of-thought onto the reasoning channel instead of leaking it.
    extractor: Option<ReasoningTagStream>,
}

impl OpenAiStreamAcc {
    /// Builds an accumulator with the given inline reasoning-tag extraction
    /// config (`None` disables inline extraction).
    pub(super) fn new(reasoning_tags: Option<ReasoningTagExtraction>) -> Self {
        let extractor = reasoning_tags.as_ref().map(ReasoningTagStream::new);
        Self {
            reasoning_tags,
            extractor,
            ..Self::default()
        }
    }

    /// Folds one parsed chunk into the accumulator and pushes the corresponding
    /// neutral [`ModelStreamItem`]s onto `pending`.
    fn ingest(&mut self, chunk: ChatCompletionChunk, pending: &mut VecDeque<ModelStreamItem>) {
        if let Some(id) = chunk.id
            && self.id.is_none()
        {
            self.id = Some(id);
        }
        if let Some(usage_wire) = chunk.usage {
            let usage = convert_usage(usage_wire);
            self.usage = Some(usage);
            pending.push_back(ModelStreamItem::UsageDelta(usage));
        }
        for mut choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
            let reasoning = delta_reasoning_text(&mut choice.delta);
            if !reasoning.is_empty() {
                self.reasoning.push_str(&reasoning);
                pending.push_back(ModelStreamItem::MessageDelta(MessageDelta {
                    text: String::new(),
                    reasoning,
                    tool_call: None,
                }));
            }
            if let Some(content) = choice.delta.content.filter(|c| !c.is_empty()) {
                // Retain the raw content so the terminal response can recompute
                // an authoritative inline-tag split (see `into_response`).
                self.text.push_str(&content);
                match self.extractor.as_mut() {
                    // Inline extraction on: scan this delta, holding back any
                    // trailing partial tag, and route the split onto the two
                    // channels as separate deltas.
                    Some(extractor) => {
                        let mut visible = String::new();
                        let mut reasoning = String::new();
                        extractor.push(&content, &mut visible, &mut reasoning);
                        if !reasoning.is_empty() {
                            pending.push_back(ModelStreamItem::MessageDelta(MessageDelta {
                                text: String::new(),
                                reasoning,
                                tool_call: None,
                            }));
                        }
                        if !visible.is_empty() {
                            pending.push_back(ModelStreamItem::MessageDelta(MessageDelta {
                                text: visible,
                                reasoning: String::new(),
                                tool_call: None,
                            }));
                        }
                    }
                    None => {
                        pending.push_back(ModelStreamItem::MessageDelta(MessageDelta {
                            text: content,
                            reasoning: String::new(),
                            tool_call: None,
                        }));
                    }
                }
            }
            for fragment in choice.delta.tool_calls {
                let idx = self.resolve_slot(&fragment);
                let slot = &mut self.tool_calls[idx];
                if let Some(id) = fragment.id.filter(|id| !id.is_empty()) {
                    slot.id = id;
                }
                if let Some(function) = fragment.function {
                    if let Some(name) = function.name.filter(|n| !n.is_empty()) {
                        slot.name = name;
                    }
                    if let Some(args) = function.arguments.filter(|a| !a.is_empty()) {
                        slot.args.push_str(&args);
                        let call_id = tool_call_id(idx, &slot.id);
                        pending.push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                            call_id,
                            content: args,
                            // Surface the tool name (captured into `slot.name` from
                            // the call-opening fragment) so consumers can label the
                            // call as it streams; the accumulator keeps the first.
                            tool_name: Some(slot.name.clone()).filter(|n| !n.is_empty()),
                        }));
                    }
                }
            }
        }
    }

    /// Resolves the accumulator slot a streamed tool-call fragment belongs to.
    ///
    /// OpenAI itself always sends a stable `index`; some OpenAI-compatible
    /// backends omit it. When `index` is present it selects the slot directly
    /// (growing the vector as needed). When it is absent, fragments are
    /// correlated by `id`: a fragment carrying a new id opens a new slot, one
    /// carrying a known id reuses that slot, and an id-less continuation fragment
    /// (arguments only) appends to the most recent slot — so parallel calls no
    /// longer all collapse onto slot 0.
    ///
    /// One exception guards Ollama's `/v1` parallel-tool-call bug
    /// (ollama/ollama#15457), where every parallel call arrives with `index: 0`:
    /// when an explicit index carries a non-empty id that *conflicts* with the id
    /// already recorded at that slot, the fragment is a distinct new call, not a
    /// continuation, so it opens a fresh slot (or reuses an existing slot already
    /// opened for that id) instead of silently merging two calls onto one.
    fn resolve_slot(&mut self, fragment: &ToolCallChunkWire) -> usize {
        if let Some(index) = fragment.index {
            let idx = index as usize;
            while self.tool_calls.len() <= idx {
                self.tool_calls.push(ToolCallBuild::default());
            }
            if let Some(id) = fragment.id.as_deref().filter(|id| !id.is_empty()) {
                let occupant = &self.tool_calls[idx];
                if !occupant.id.is_empty() && occupant.id != id {
                    // Conflict: a second distinct call reusing index 0. Reuse a
                    // slot already opened for this id if one exists (so its later
                    // continuation fragments still land correctly), else open a
                    // fresh slot rather than overwriting the occupant.
                    if let Some(pos) = self.tool_calls.iter().position(|slot| slot.id == id) {
                        return pos;
                    }
                    self.tool_calls.push(ToolCallBuild::default());
                    return self.tool_calls.len() - 1;
                }
            }
            return idx;
        }
        if let Some(id) = fragment.id.as_deref().filter(|id| !id.is_empty()) {
            if let Some(pos) = self.tool_calls.iter().position(|slot| slot.id == id) {
                return pos;
            }
            self.tool_calls.push(ToolCallBuild::default());
            return self.tool_calls.len() - 1;
        }
        if self.tool_calls.is_empty() {
            self.tool_calls.push(ToolCallBuild::default());
        }
        self.tool_calls.len() - 1
    }

    /// Consumes the accumulator into the final, merged [`ModelResponse`].
    ///
    /// Infallible: a tool call whose reassembled arguments cannot be parsed is
    /// surfaced as an [`ToolCall::invalid`] call (raw arguments preserved) rather
    /// than failing the whole stream, so the agent loop can feed the error back
    /// to the model and the call still resolves instead of stalling the loop.
    fn into_response(self) -> ModelResponse {
        let mut content = Vec::new();
        // Recompute the inline-tag split over the raw accumulated content so the
        // terminal response is byte-identical to the non-streaming path, then
        // combine it with side-channel reasoning. Both feed one leading
        // `Thinking` block (side-channel leads) so it survives persistence and
        // provider replay.
        let (visible_text, reasoning) = match &self.reasoning_tags {
            Some(config) => {
                let (visible, inline) = extract_reasoning(config, &self.text);
                let mut reasoning = self.reasoning.clone();
                if !inline.is_empty() {
                    if !reasoning.is_empty() {
                        reasoning.push_str(config.separator());
                    }
                    reasoning.push_str(&inline);
                }
                (visible, reasoning)
            }
            None => (self.text.clone(), self.reasoning.clone()),
        };
        if !reasoning.is_empty() {
            content.push(ContentBlock::Thinking {
                text: reasoning,
                signature: None,
            });
        }
        if !visible_text.is_empty() {
            content.push(ContentBlock::Text(visible_text));
        }
        // Enumerate over the full slot vector *before* filtering so the synthetic
        // fallback id (`tool-{idx}`) matches the one streamed in `ToolCallDelta`
        // items — filtering first would renumber the slots and desynchronize the
        // delta ids from the final call ids.
        let tool_calls = self
            .tool_calls
            .into_iter()
            .enumerate()
            .filter(|(_, b)| !b.name.is_empty() || !b.args.is_empty())
            .map(|(idx, b)| tool_call_from_wire("openai stream", idx, &b.id, &b.name, &b.args))
            .collect::<Vec<_>>();
        let message = AssistantMessage {
            id: self.id,
            content,
            tool_calls,
            usage: self.usage,
        };
        ModelResponse {
            message,
            usage: self.usage,
            finish_reason: self.finish_reason,
            raw: None,
            resolved_model: None,
        }
    }
}

/// Mutable driver state threaded through [`futures::stream::unfold`] while
/// parsing the SSE byte stream into [`ModelStreamItem`]s.
pub(super) struct SseState {
    /// Raw response byte chunks (errors already mapped onto the crate error).
    ///
    /// The item type is [`bytes::Bytes`] — the buffer `reqwest` hands us — so
    /// each network chunk is forwarded by reference count instead of being
    /// copied into a fresh `Vec<u8>`. This type is crate-internal
    /// (`pub(super)`), so `bytes` stays out of the public API.
    pub(super) bytes: Pin<Box<dyn Stream<Item = Result<bytes::Bytes>> + Send>>,
    /// Raw bytes received but not yet split into complete lines. Kept as bytes
    /// (not a `String`) so a multi-byte UTF-8 character split across two network
    /// chunks is reassembled before decoding, instead of being corrupted into
    /// replacement characters by a premature lossy decode.
    pub(super) buf: Vec<u8>,
    /// Parsed items waiting to be yielded, in order.
    pub(super) pending: VecDeque<ModelStreamItem>,
    /// Provider-side response reconstruction.
    pub(super) acc: OpenAiStreamAcc,
    /// Provider family id used in normalized stream failures.
    pub(super) provider: String,
    /// Provider model id used in normalized stream failures.
    pub(super) model: String,
    /// Whether the leading [`ModelStreamItem::Started`] has been emitted.
    pub(super) started: bool,
    /// Whether the byte stream ended or `[DONE]` was seen.
    pub(super) finished: bool,
    /// Whether the terminal [`ModelStreamItem::Completed`]/[`ModelStreamItem::Failed`]
    /// has been emitted.
    pub(super) terminal_emitted: bool,
}

impl SseState {
    /// Splits buffered bytes into complete newline-terminated lines and folds
    /// each SSE `data:` payload into the accumulator. The trailing partial line
    /// (if any) is kept in `buf` for the next chunk, so a `data:` line split
    /// across chunk boundaries — including one that splits a multi-byte UTF-8
    /// character — is only decoded once it is complete.
    fn drain_lines(&mut self) {
        // Scan with a moving start offset and drain the whole consumed prefix
        // once at the end, rather than `drain(..=pos)` per line: the old form
        // allocated a `Vec<u8>` per line and shifted the remaining buffer down
        // on every line (O(n^2) over a chunk carrying many lines).
        let mut start = 0;
        while let Some(rel) = self.buf[start..].iter().position(|&b| b == b'\n') {
            let end = start + rel;
            // A complete line (bounded by the ASCII `\n`) is whole UTF-8, so a
            // lossy decode here can no longer straddle a chunk boundary. The
            // `into_owned` detaches the line from `buf` so `process_line` can
            // borrow `self` mutably.
            let line = String::from_utf8_lossy(&self.buf[start..end]).into_owned();
            start = end + 1;
            self.process_line(&line);
        }
        if start > 0 {
            self.buf.drain(..start);
        }
    }

    /// Folds any bytes still buffered after the byte stream ends into a final
    /// line. Providers that terminate the last SSE event without a trailing
    /// newline would otherwise leave the final `data:` payload unprocessed.
    fn drain_remaining(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let line = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        self.process_line(&line);
    }

    /// Parses one SSE line and folds any resulting chunk into the accumulator.
    fn process_line(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        let Some(rest) = line.strip_prefix("data:") else {
            return;
        };
        let payload = rest.trim();
        if payload == "[DONE]" {
            self.finished = true;
            return;
        }
        // Ignore keepalives / unparseable lines rather than failing the run.
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        // Some providers stream a mid-stream `{"error": ...}` payload instead of
        // a chunk. This also deserializes cleanly as an all-defaults
        // `ChatCompletionChunk`, so it must be detected first and surfaced as a
        // terminal failure rather than folded in as an empty chunk and swallowed.
        if let Some(error) = value.get("error") {
            self.pending
                .push_back(ModelStreamItem::ProviderFailed(self.stream_error(error)));
            self.finished = true;
            self.terminal_emitted = true;
            return;
        }
        if let Ok(chunk) = serde_json::from_value::<ChatCompletionChunk>(value) {
            let mut pending = std::mem::take(&mut self.pending);
            self.acc.ingest(chunk, &mut pending);
            self.pending = pending;
        }
    }

    /// Builds a normalized [`ProviderError`] from a streamed `error` payload.
    fn stream_error(&self, error: &Value) -> ProviderError {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .filter(|message| !message.trim().is_empty())
            .unwrap_or("provider reported a stream error")
            .to_string();
        let code = error
            .get("code")
            .or_else(|| error.get("type"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let retryable =
            crate::harness::retry::classify_provider_failure(None, code.as_deref(), &message)
                .is_retryable();
        ProviderError {
            provider: self.provider.clone(),
            model: Some(self.model.clone()),
            code,
            message,
            retryable,
            raw: Some(error.clone()),
            ..ProviderError::default()
        }
    }
}

/// Advances the SSE [`SseState`] by one item for [`futures::stream::unfold`].
pub(super) async fn sse_next(mut state: SseState) -> Option<(ModelStreamItem, SseState)> {
    loop {
        if let Some(item) = state.pending.pop_front() {
            return Some((item, state));
        }
        if !state.started {
            state.started = true;
            return Some((ModelStreamItem::Started, state));
        }
        if state.finished {
            if state.terminal_emitted {
                return None;
            }
            state.terminal_emitted = true;
            // Reconstruction is infallible: malformed tool arguments become an
            // `ToolCall::invalid` call inside the response (not a stream
            // failure), so the agent loop recovers instead of aborting the run.
            let response = std::mem::take(&mut state.acc).into_response();
            return Some((ModelStreamItem::Completed(response), state));
        }
        match state.bytes.next().await {
            Some(Ok(chunk)) => {
                state.buf.extend_from_slice(&chunk);
                state.drain_lines();
            }
            Some(Err(error)) => {
                state.finished = true;
                state.terminal_emitted = true;
                let provider_error = ProviderError {
                    provider: state.provider.clone(),
                    model: Some(state.model.clone()),
                    message: error.to_string(),
                    retryable: true,
                    ..ProviderError::default()
                };
                return Some((ModelStreamItem::ProviderFailed(provider_error), state));
            }
            None => {
                // Drain any final `data:` line the provider sent without a
                // trailing newline before terminating.
                state.drain_remaining();
                state.finished = true;
            }
        }
    }
}
