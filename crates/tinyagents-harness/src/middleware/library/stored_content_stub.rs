//! [`StoredContentStubMiddleware`]: send a reference in place of content a tool
//! has already stored.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::context::RunContext;
use crate::error::Result as TaResult;
use crate::middleware::Middleware;
use tinyinference_llm::message::{ContentBlock, Message};
use tinyinference_llm::model::ModelRequest;
use tinyinference_llm::tool::ToolCall;

/// What a stub is rendered from.
#[derive(Clone, Copy, Debug)]
pub struct StoredContent<'a> {
    /// The tool that stored or returned the content.
    pub tool: &'a str,
    /// The call's arguments, as the model wrote them.
    pub arguments: &'a Value,
    /// The text the stub would replace.
    pub content: &'a str,
    /// The text of the call's tool result.
    pub result: &'a str,
}

/// Renders the reference sent in place of [`StoredContent::content`], or `None`
/// to send the content as it is, which is the answer for a call that stored
/// nothing and for content that already is a stub.
///
/// It must be a pure function of its input: the same transcript has to render
/// the same bytes on every call and after a resume.
pub type StubRenderer = Arc<dyn Fn(&StoredContent<'_>) -> Option<String> + Send + Sync>;

/// Middleware that sends a short reference in place of content a tool holds
/// durably, so the model reads it back on demand instead of carrying it, and
/// acting on, a copy in its history.
///
/// A tool that writes a document takes the whole body as an argument, so the
/// body sits in the assistant message's tool call for the rest of the
/// conversation. Clearing tool *results* ([`MicrocompactMiddleware`]) does not
/// reach it. In `before_model` this rewrites the outgoing request:
///
/// - **an argument** named with [`Self::with_argument`] is replaced once the
///   call has a result. The renderer sees that result, so a failed call, which
///   stored nothing, can keep its content for the model to fix and retry;
/// - **a result** named with [`Self::with_earlier_result`] is replaced once it
///   belongs to an earlier turn: the model answered after it, and a user message
///   came after that. The turn that read the content keeps it.
///
/// Only the request is rewritten. The agent loop rebuilds every request from its
/// working transcript, so the stored transcript stays whole, and the rewrite is
/// a pure function of the messages, so a resumed session renders the same bytes.
///
/// # Prompt-cache stability
///
/// A tool call is first sent on the request after the model wrote it, by which
/// point its result exists, so an argument goes out as a stub from its first
/// send and the prefix stays append-only. A result has to reach the model once,
/// so its stub changes one position already sent, at the first request of the
/// next turn. Everything before it stays cached and the stub never changes again.
///
/// Install it ahead of the middleware that reduces or summarizes a request, so
/// those see the stubs and the renderer sees whole results.
pub struct StoredContentStubMiddleware {
    arguments: HashMap<String, Vec<String>>,
    results: HashSet<String>,
    render: StubRenderer,
}

impl StoredContentStubMiddleware {
    /// A middleware with no tools named, stubbing through `render`.
    #[must_use]
    pub fn new(render: StubRenderer) -> Self {
        Self {
            arguments: HashMap::new(),
            results: HashSet::new(),
            render,
        }
    }

    /// Stub the string argument `argument` of `tool`'s calls. A tool can name
    /// several.
    #[must_use]
    pub fn with_argument(mut self, tool: impl Into<String>, argument: impl Into<String>) -> Self {
        self.arguments
            .entry(tool.into())
            .or_default()
            .push(argument.into());
        self
    }

    /// Stub `tool`'s results once they belong to an earlier turn.
    #[must_use]
    pub fn with_earlier_result(mut self, tool: impl Into<String>) -> Self {
        self.results.insert(tool.into());
        self
    }

    /// Stubs the named arguments of every call that has a result. Returns how
    /// many it replaced.
    fn stub_arguments(&self, messages: &mut [Message]) -> usize {
        let mut stubbed = 0;
        for index in 0..messages.len() {
            let Message::Assistant(assistant) = &messages[index] else {
                continue;
            };
            let results: Vec<Option<String>> = assistant
                .tool_calls
                .iter()
                .map(|call| result_of(messages, index, &call.id))
                .collect();
            let Message::Assistant(assistant) = &mut messages[index] else {
                continue;
            };
            for (call, result) in assistant.tool_calls.iter_mut().zip(results) {
                let Some(arguments) = self.arguments.get(&call.name) else {
                    continue;
                };
                let Some(result) = result.filter(|_| call.invalid.is_none()) else {
                    continue;
                };
                for argument in arguments {
                    let stub = call
                        .arguments
                        .get(argument)
                        .and_then(Value::as_str)
                        .and_then(|content| {
                            (self.render)(&StoredContent {
                                tool: &call.name,
                                arguments: &call.arguments,
                                content,
                                result: &result,
                            })
                        });
                    if let Some(stub) = stub {
                        call.arguments[argument] = Value::String(stub);
                        stubbed += 1;
                    }
                }
            }
        }
        stubbed
    }

    /// Stubs the named results that belong to an earlier turn. Returns how many
    /// it replaced.
    fn stub_earlier_results(&self, messages: &mut [Message]) -> usize {
        let Some(answered) = last_answered_turn_end(messages) else {
            return 0;
        };
        let mut stubbed = 0;
        for index in 0..answered {
            let text = messages[index].text();
            let Message::Tool(tool) = &messages[index] else {
                continue;
            };
            let Some(call) = call_of(messages, index, &tool.tool_call_id)
                .filter(|call| self.results.contains(&call.name))
                .cloned()
            else {
                continue;
            };
            if tool.trusted_verbatim {
                continue;
            }
            let stub = (self.render)(&StoredContent {
                tool: &call.name,
                arguments: &call.arguments,
                content: &text,
                result: &text,
            });
            let Message::Tool(tool) = &mut messages[index] else {
                continue;
            };
            if let Some(stub) = stub {
                tool.content = vec![ContentBlock::Text(stub)];
                stubbed += 1;
            }
        }
        stubbed
    }
}

/// The result of the call `id` made by the assistant message at `assistant`.
/// Results follow their calls, and a provider may reuse an id later on, so an id
/// only means something within its assistant message.
fn result_of(messages: &[Message], assistant: usize, id: &str) -> Option<String> {
    messages[assistant + 1..]
        .iter()
        .take_while(|message| !matches!(message, Message::Assistant(_)))
        .find(|message| matches!(message, Message::Tool(tool) if tool.tool_call_id == id))
        .map(Message::text)
}

/// The call that produced the tool message at `tool`: the nearest one before it
/// with that id.
fn call_of<'a>(messages: &'a [Message], tool: usize, id: &str) -> Option<&'a ToolCall> {
    messages[..tool]
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::Assistant(assistant) => assistant.tool_calls.iter().find(|call| call.id == id),
            _ => None,
        })
}

/// The index of the last assistant message that a user message follows:
/// everything before it is a turn the model has finished with.
fn last_answered_turn_end(messages: &[Message]) -> Option<usize> {
    let user = messages
        .iter()
        .rposition(|message| matches!(message, Message::User(_)))?;
    messages[..user]
        .iter()
        .rposition(|message| matches!(message, Message::Assistant(_)))
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> Middleware<State, Ctx> for StoredContentStubMiddleware {
    fn name(&self) -> &str {
        "stored_content_stub"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<Ctx>,
        _state: &State,
        request: &mut ModelRequest,
    ) -> TaResult<()> {
        let arguments = self.stub_arguments(&mut request.messages);
        let earlier_results = self.stub_earlier_results(&mut request.messages);
        tracing::debug!(
            arguments,
            earlier_results,
            "[stored_content_stub] request rewritten"
        );
        Ok(())
    }
}
