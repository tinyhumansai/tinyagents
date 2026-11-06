//! Built-in middleware library.
//!
//! This module collects the ready-to-use middleware that ship with the harness.
//! They use the model-wrap and lifecycle extension surfaces from
//! [`crate::middleware`]:
//!
//! - **Resilience (wrap)** — [`RetryMiddleware`], [`TimeoutMiddleware`],
//!   [`ModelFallbackMiddleware`], and [`RateLimitMiddleware`] implement the
//!   around-call [`ModelMiddleware`] trait and surround the real model call.
//! - **Policy / guard / observation (lifecycle)** —
//!   [`ToolAllowlistMiddleware`], [`DynamicToolSelectionMiddleware`],
//!   [`HumanApprovalMiddleware`], [`StructuredOutputValidatorMiddleware`],
//!   [`DynamicPromptMiddleware`], [`RedactionMiddleware`], and
//!   [`TracingMiddleware`] implement the lifecycle [`Middleware`] trait.
//!
//! - **Context ladder and loop guards** — [`ImageAwareMessageTrimMiddleware`],
//!   [`ArtifactIndexTocMiddleware`], [`FinalCallWrapUpMiddleware`],
//!   [`RepeatProgressMiddleware`] / [`RepeatEvictionObserver`],
//!   [`ArgRecoveryMiddleware`] and [`CredentialScrubMiddleware`]. All are
//!   generic over the run-context payload; host policy (tool names, the
//!   cleared-result placeholder, exemptions, per-tool scrubbing) is injected
//!   through constructor arguments.
//!
//! Type definitions live in `types`; this file holds the constructors and
//! trait impls. Tests live in sibling `*_tests.rs` files.
//!
//! # Testability
//!
//! None of these middleware sleep on the wall clock in a way that tests cannot
//! control: [`RetryMiddleware`] sleeps on backoff only when its policy opts in
//! via [`RetryPolicy::with_backoff_sleep`] (off by default),
//! [`TimeoutMiddleware`] is exercised under `tokio::time` paused-time tests, and
//! [`RateLimitMiddleware`] takes an injectable clock and a configurable poll
//! interval so its wait loop can be driven deterministically.

mod types;

pub use types::*;

use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::context::{MiddlewareControl, RunConfig, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::AgentEvent;
use crate::ids::CallId;
use crate::middleware::{
    Middleware, MiddlewareModelOutcome, ModelHandler, ModelMiddleware, ToolInvocationIdentity,
};
use crate::retry::{RateLimiter, RetryPolicy, is_retryable};
use crate::structured::{StructuredExtractor, StructuredStrategy};
use tinyinference_llm::message::{ContentBlock, Message};
use tinyinference_llm::model::{ModelDelta, ModelRequest, ModelResponse, ResponseFormat};
use tinyinference_llm::tool::{ToolCall, ToolDelta, ToolSchema};
use tinytools::ToolResult;

mod arg_recovery;
mod artifact_toc;
mod budget;
mod call_effect;
mod cli_rpc_only;
mod compaction_pressure;
mod context;
mod context_ladder;
mod credential_scrub;
mod ephemeral;
mod failure_scope;
mod image_trim;
mod loop_guards;
mod memory_protocol;
mod memory_protocol_middleware;
mod observe;
mod output_cap;
mod output_ladder;
mod policy_gate;
mod repeat_loop_stop;
mod repeat_progress;
mod repeat_progress_state;
mod research_budget;
mod resilience;
mod response_shape;
mod shell_turn_budget;
mod stored_content_stub;
mod tool_exposure_shadow;
mod tool_outcome_capture;
mod tool_policy;
mod transcript_snapshot;
mod turn_clock;
mod unmet_deliverable;
mod verify_before_finish;
mod wind_down;
mod workload;
mod wrap_up;

pub use arg_recovery::ArgRecoveryMiddleware;
pub use artifact_toc::{
    ARTIFACT_INDEX_NAMESPACE, ArtifactIndexTocMiddleware, FOOTER_ALLOWANCE, NO_WINDOW_ALLOWANCE,
    split_input_allowance,
};
pub use call_effect::{
    CallEffect, DispatchTarget, ToolEffectFacts, ToolFactsLookup, call_effect, dispatch_target,
    facts_for_tool, name_effect, tool_sets_lookup,
};
pub use cli_rpc_only::CliRpcOnlyMiddleware;
pub use context_ladder::{
    COMPACTION_TAIL_WINDOW_FRACTION, ContextLadder, UNKNOWN_WINDOW_FALLBACK_TOKENS,
    compression_policy, microcompact_for_window, with_turn_aware_tail,
};
pub use credential_scrub::{
    CredentialScrubMiddleware, REDACTION_PLACEHOLDER, ToolScrubber, redaction_notice,
    scrub_with_notice,
};
pub use ephemeral::rehome_ephemeral_system_instructions;
pub use ephemeral::{HARNESS_NOTE_HEADER, push_ephemeral_instruction};
pub use failure_scope::{
    FailureAssessment, FailureScope, FailureScopeMiddleware, FailureScopePolicy, FailureStopKind,
};
pub use image_trim::{
    IMAGE_MARKER_TOKEN_COST, ImageAwareMessageTrimMiddleware, estimate_message_tokens,
    estimate_text_tokens, legacy_max_input_tokens,
};
pub use loop_guards::{
    LoopGuardThresholds, RECOVERABLE_NO_PROGRESS_FAILURE_THRESHOLD,
    RECOVERABLE_REPEAT_FAILURE_THRESHOLD, is_repeat_call_exempt,
};
pub use memory_protocol::{
    MEMORY_PROTOCOL_MARKER, MemoryOp, MemoryProtocolObservation, MemoryProtocolSpec,
    MemoryProtocolTracker, ModeTool,
};
pub use memory_protocol_middleware::MemoryProtocolMiddleware;
pub use output_cap::{
    ListedOutputCapPolicy, OutputCapPolicy, OutputStagePlan, disclose, plan_output_stages,
    should_compact,
};
pub use output_ladder::{CompactedOutput, OutputLadderInput, OutputProcessor, run_output_ladder};
pub use policy_gate::{
    ApprovalExecutionOutcome, ApprovalGateMiddleware, ApprovalResolution, ApprovalResolver,
    DenialRenderer, GateVerdict, PolicyDecision, SessionToolCeiling, SessionToolListing,
    StructuredDenialParts, StructuredDenialRenderOptions, ToolCallPolicy, ToolPolicyGate,
    ToolPolicyGateMiddleware, ToolPolicyPostGate, record_approved, structured_denial_renderer,
};
pub use repeat_loop_stop::{RepeatLoopWitness, is_correctable_class, repeat_loop_halt_summary};
pub use repeat_progress::{
    HaltSummarySlot, ReadOnlyCheck, RepeatEvictionObserver, RepeatExemption,
    RepeatProgressMiddleware,
};
pub use repeat_progress_state::{
    REPEAT_GUARD_BLOCKED, REPEAT_GUARD_HALTED, REPEAT_GUARD_METADATA_KEY, repeat_guard_marker,
};
pub use research_budget::{
    DIRECT_WEB_READ_LIMIT, RESEARCH_CLOSE_INSTRUCTION, ResearchBudgetMiddleware, WEB_BLOCKED_NOTE,
    WEB_BUDGET_EXHAUSTED_INSTRUCTION,
};
pub use response_shape::{
    FinalResponse, ResponseShape, ResponseShapeScope, ResponseUsage, ResponseValidator,
    install_response_shape,
};
pub use shell_turn_budget::{
    DEFAULT_MAX_TIMEOUT_SECS, SHELL_TIMEOUT_RESERVE, ShellTimeoutClamp,
    ShellTimeoutClampMiddleware, ShellTimeoutNoteMiddleware, ShellTurnBudget, clamp_shell_timeout,
    install_time_notes,
};
pub use stored_content_stub::{StoredContent, StoredContentStubMiddleware, StubRenderer};
pub use tool_exposure_shadow::{ExposureDivergence, ToolExposureShadowMiddleware};
pub use tool_outcome_capture::ToolOutcomeCaptureMiddleware;
pub use transcript_snapshot::{
    TranscriptSnapshot, TranscriptSnapshotMiddleware, TranscriptSnapshotSink,
    new_transcript_snapshot_sink, render_unanswered_steps,
};
pub(crate) use turn_clock::is_json_document;
pub use turn_clock::{TurnClock, TurnClockMiddleware};
pub use unmet_deliverable::{
    UnmetDeliverableMiddleware, candidate_paths, install as install_unmet_deliverable,
    notice as unmet_deliverable_notice, wrap_harness_instruction,
};
pub use verify_before_finish::{
    DEFAULT_MIN_REMAINING_WALL_CLOCK, FinishActivity, FinishCheckTrigger,
    MIN_REMAINING_MODEL_CALLS, VerifyBeforeFinishMiddleware,
};
pub use wind_down::{WindDownCallback, WindDownMiddleware, install as install_wind_down};
pub use workload::{
    CostEstimator, FallbackObserverMiddleware, RequiredCapabilitiesMiddleware,
    ResolvedRouteMiddleware, UsageCarryMiddleware,
};
pub use wrap_up::{
    CapturedOutcomes, DEFAULT_CLEARED_PLACEHOLDER, FinalCallWrapUpMiddleware, OutcomesUnavailable,
};

#[cfg(test)]
#[path = "arg_recovery_tests.rs"]
mod arg_recovery_test;
#[cfg(test)]
#[path = "artifact_toc_tests.rs"]
mod artifact_toc_test;
#[cfg(test)]
#[path = "context_fold_tests.rs"]
mod context_fold_test;
#[cfg(test)]
#[path = "context_ladder_tests.rs"]
mod context_ladder_test;
#[cfg(test)]
#[path = "context_loop_tests.rs"]
mod context_loop_test;
#[cfg(test)]
#[path = "context_pin_tests.rs"]
mod context_pin_test;
#[cfg(test)]
#[path = "context_task_state_tests.rs"]
mod context_task_state_test;
#[cfg(test)]
#[path = "credential_scrub_tests.rs"]
mod credential_scrub_test;
#[cfg(test)]
#[path = "image_trim_tests.rs"]
mod image_trim_test;
#[cfg(test)]
#[path = "policy_gate_tests.rs"]
mod policy_gate_test;
#[cfg(test)]
#[path = "repeat_escalation_loop_tests.rs"]
mod repeat_escalation_loop_test;
#[cfg(test)]
#[path = "repeat_escalation_tests.rs"]
mod repeat_escalation_test;
#[cfg(test)]
#[path = "repeat_progress_tests.rs"]
mod repeat_progress_test;
#[cfg(test)]
#[path = "stored_content_stub_tests.rs"]
mod stored_content_stub_test;
#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
#[cfg(test)]
#[path = "wrap_up_tests.rs"]
mod wrap_up_test;

#[cfg(test)]
#[path = "workload_tests.rs"]
mod workload_test;

mod ephemeral_instructions;
pub use ephemeral_instructions::{EphemeralInstructionMiddleware, EphemeralInstructionQueue};
