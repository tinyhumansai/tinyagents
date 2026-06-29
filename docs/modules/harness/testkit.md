# Harness Testkit Feature

The testkit provides deterministic utilities for exercising inherently
nondeterministic model and tool workflows.

## Source Inspiration

LangChain keeps fake models, standard provider tests, trajectory-style agent
tests, event-stream tests, cache tests, and callback tests:

- fake chat models:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/fake_chat_models.py>
- standard integration tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/standard-tests>
- core unit tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/tests/unit_tests>
- v1 agent tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/tests>

TinyAgents should ship a reusable testkit so application code and provider
adapters can assert behavior without live providers.

## Responsibilities

- Provide fake chat models.
- Provide deterministic fake embedding models.
- Provide scripted streaming models.
- Provide fake tools.
- Provide failing and flaky tools/models.
- Provide in-memory stores and event collectors.
- Provide deterministic clocks and id generators.
- Provide trajectory assertions for agent loops.
- Provide graph trajectory assertions for state-graph runs.
- Provide conformance tests for provider adapters.
- Provide replay fixtures from recorded events.
- Provide helpers for structured-output validation tests.
- Provide in-memory vector stores and record managers for retrieval/indexing
  tests.

## Core Utilities

```rust
pub struct ScriptedModel {
    pub script: Vec<ModelScriptStep>,
}

pub enum ModelScriptStep {
    Respond(AssistantMessage),
    Stream(Vec<ModelStreamItem>),
    Fail(RustAgentsError),
}

pub struct EventCollector {
    pub events: Arc<Mutex<Vec<HarnessEvent>>>,
}

pub struct DeterministicFakeEmbedding {
    pub dimensions: usize,
    pub seed_namespace: String,
}
```

## Trajectory Assertions

Tests should be able to assert:

- number of model calls
- number of tool calls
- exact tool names and arguments
- final assistant message
- structured output value
- emitted event kinds
- retry attempts
- fallback selections
- cache hit/miss behavior
- summary creation
- usage and cost records
- no unexpected provider calls
- retrieved document ids and scores
- indexed document hashes
- graph run status
- graph node transition sequence
- checkpoint labels and state snapshots
- HITL interrupt kind/options/resume node

## Provider Conformance

Provider adapters should run a shared suite that covers:

- text invoke
- text stream
- system messages
- multiple user/assistant turns
- Unicode
- multimodal input when profile declares support
- tool binding
- tool-call id preservation
- no-argument tool calls
- streamed tool-call chunks when profile declares support
- structured output in provider mode
- structured output in tool mode
- usage metadata
- cancellation
- timeout
- rate-limit error classification

State-graph conformance should cover:

- linear route
- conditional route
- fork and merge reducer
- cycle guarded by max steps
- cancellation
- node error propagation
- HITL pause/resume/reject
- in-memory and durable checkpointer round trip
- run list pagination and newest-first order
- per-agent blueprint validation
- live turn graph tool advertisement, allowlist enforcement, malformed
  argument handling, and final output depending on actual tool execution
- provider options passthrough

Tests requiring live providers should be opt-in and documented with environment
variables. Unit-level conformance should run with fakes in normal CI.
