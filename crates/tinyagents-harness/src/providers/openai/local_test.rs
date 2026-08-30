//! Unit tests for local-runtime identification, probing, and classification.

use super::*;
use serde_json::json;

const LMSTUDIO_CHAT_TEMPLATE_REJECTION: &str = "lmstudio returned: Engine protocol predict \
    request returned 400: {\"error\":{\"code\":400,\"message\":\"Unable to generate parser \
    for this template. Automatic parser generation failed: While executing CallExpression at \
    line 79, column 24 in source: {{- raise_exception('No user query found in messages.') }}. \
    Error: Jinja Exception: No user query found in messages.\"}}";

#[test]
fn chat_template_rejections_are_classified_inside_runtime_wrappers() {
    let aggregate =
        format!("The model may not be available. Attempts: {LMSTUDIO_CHAT_TEMPLATE_REJECTION}");
    assert!(is_chat_template_rejection_message(&aggregate));
}

#[test]
fn chat_template_rejection_detection_is_case_insensitive() {
    assert!(is_chat_template_rejection_message(
        "Error: JINJA EXCEPTION: No User Query Found In Messages."
    ));
}

#[test]
fn unrelated_provider_rejections_are_not_chat_template_failures() {
    for body in [
        "openai API error (400): invalid temperature: only 1 is allowed for this model",
        "The model `gpt-5.5` does not exist or you do not have access to it.",
        "lmstudio returned: model 'qwen3.5-9b' does not support tools",
        "openrouter API error (429): rate limited",
        "Failed to render the prompt template file on disk",
    ] {
        assert!(!is_chat_template_rejection_message(body), "{body:?}");
    }
}

#[test]
fn native_root_strips_the_openai_compat_suffix() {
    let kind = LocalRuntimeKind::Ollama;
    assert_eq!(
        kind.native_root("http://localhost:11434/v1"),
        "http://localhost:11434"
    );
    assert_eq!(
        kind.native_root("http://localhost:11434/v1/"),
        "http://localhost:11434"
    );
    // Idempotent for a base that is already the root.
    assert_eq!(
        kind.native_root("http://localhost:11434"),
        "http://localhost:11434"
    );
}

#[test]
fn ollama_show_reports_the_real_window_not_the_model_id_guess() {
    // `llama3.2:3b` matches the generic hint table's `("llama3", Substring,
    // 128_000)` entry. The server says 8192. The probe must report the server.
    let body = json!({
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 8192,
            "llama.embedding_length": 3072
        },
        "capabilities": ["completion", "tools"]
    });
    let probe = probe_from_ollama_show(&body);
    assert_eq!(probe.max_input_tokens, Some(8192));
    assert_eq!(probe.tool_calling, Some(true));
    assert_eq!(probe.vision, Some(false));
    assert_eq!(probe.reasoning, Some(false));
}

#[test]
fn ollama_show_reads_vision_and_thinking_capabilities() {
    let body = json!({
        "model_info": { "qwen3.context_length": 40960 },
        "capabilities": ["completion", "tools", "vision", "thinking"]
    });
    let probe = probe_from_ollama_show(&body);
    assert_eq!(probe.max_input_tokens, Some(40960));
    assert_eq!(probe.tool_calling, Some(true));
    assert_eq!(probe.vision, Some(true));
    assert_eq!(probe.reasoning, Some(true));
}

#[test]
fn an_absent_capabilities_array_leaves_every_capability_unknown() {
    // "the server did not tell us" must not be read as "the model cannot".
    let body = json!({ "model_info": { "llama.context_length": 4096 } });
    let probe = probe_from_ollama_show(&body);
    assert_eq!(probe.max_input_tokens, Some(4096));
    assert_eq!(probe.tool_calling, None);
    assert_eq!(probe.vision, None);
    assert_eq!(probe.reasoning, None);
}

#[test]
fn a_body_without_model_info_probes_to_nothing_rather_than_a_guess() {
    assert!(probe_from_ollama_show(&json!({})).is_empty());
    assert!(probe_from_ollama_show(&json!({ "error": "model not found" })).is_empty());
}

#[test]
fn context_length_scan_is_architecture_agnostic_and_conservative() {
    let info = json!({ "gemma3.context_length": 8192, "clip.context_length": 77 })
        .as_object()
        .cloned()
        .unwrap();
    // Multimodal models carry a second, tiny window for the projector. Taking
    // the max would overstate; take the min.
    assert_eq!(context_length_from_model_info(&info), Some(77));

    let zeroed = json!({ "llama.context_length": 0 })
        .as_object()
        .cloned()
        .unwrap();
    assert_eq!(context_length_from_model_info(&zeroed), None);
}

#[test]
fn lm_studio_listing_reports_loaded_and_trained_windows() {
    let body = json!({
        "data": [
            { "id": "other-model", "max_context_length": 999999 },
            {
                "id": "qwen3-4b",
                "type": "llm",
                "max_context_length": 40960,
                "loaded_context_length": 4096
            }
        ]
    });
    let probe = probe_from_lm_studio_models(&body, "qwen3-4b");
    assert_eq!(probe.max_input_tokens, Some(40960));
    assert_eq!(probe.loaded_num_ctx, Some(4096));
    assert_eq!(probe.vision, Some(false));
    // The loaded window is the ceiling a live request actually has.
    assert_eq!(probe.effective_context_window(), Some(4096));
}

#[test]
fn lm_studio_vision_models_are_detected_by_type() {
    let body = json!({ "data": [{ "id": "llava", "type": "vlm", "max_context_length": 4096 }] });
    assert_eq!(
        probe_from_lm_studio_models(&body, "llava").vision,
        Some(true)
    );
}

#[test]
fn an_unlisted_lm_studio_model_probes_to_nothing() {
    let body = json!({ "data": [{ "id": "a", "max_context_length": 4096 }] });
    assert!(probe_from_lm_studio_models(&body, "b").is_empty());
}

#[test]
fn effective_window_prefers_the_loaded_ctx_over_the_trained_window() {
    let probe = LocalProbe {
        max_input_tokens: Some(131_072),
        loaded_num_ctx: Some(2048),
        ..LocalProbe::default()
    };
    // Overstating is the LOCAL-1 defect: compaction is gated on this number, so
    // a window bigger than the server honours means it never fires.
    assert_eq!(probe.effective_context_window(), Some(2048));
}

#[test]
fn load_body_carries_num_ctx_and_keep_alive_natively() {
    let options = json!({ "num_ctx": 8192, "num_batch": 512 });
    let body = ollama_load_body("llama3.2", Some(&options), Some("30m"));
    assert_eq!(body["model"], json!("llama3.2"));
    assert_eq!(body["messages"], json!([]));
    // The whole point: `num_ctx` reaches a field Ollama actually reads.
    assert_eq!(body["options"]["num_ctx"], json!(8192));
    assert_eq!(body["options"]["num_batch"], json!(512));
    assert_eq!(body["keep_alive"], json!("30m"));
}

#[test]
fn load_body_omits_absent_and_malformed_options() {
    let body = ollama_load_body("m", None, None);
    assert!(body.get("options").is_none());
    assert!(body.get("keep_alive").is_none());

    let scalar = json!("not an object");
    let body = ollama_load_body("m", Some(&scalar), None);
    assert!(body.get("options").is_none());
}

#[test]
fn options_are_lifted_out_of_the_documented_provider_options_shape() {
    let provider_options = json!({ "options": { "num_ctx": 8192 }, "keep_alive": "5m" });
    assert_eq!(
        local_options_object(&provider_options),
        Some(&json!({ "num_ctx": 8192 }))
    );
    assert_eq!(local_options_object(&json!({})), None);
    // A non-object `options` is caller error, not something to forward.
    assert_eq!(local_options_object(&json!({ "options": 7 })), None);
}

#[test]
fn missing_model_404_gains_an_actionable_remediation() {
    let message = missing_model_remediation(
        LocalRuntimeKind::Ollama,
        404,
        r#"{"error":"model 'llama3.2' not found"}"#,
        "llama3.2",
        "http://localhost:11434/v1",
    )
    .expect("a missing-model 404 is rewritten");
    assert!(message.contains("ollama pull llama3.2"), "{message}");

    let lm = missing_model_remediation(
        LocalRuntimeKind::LmStudio,
        404,
        "model does not exist",
        "qwen3-4b",
        "http://localhost:1234/v1",
    )
    .expect("LM Studio gets its own wording");
    assert!(lm.contains("list_models()"), "{lm}");
}

#[test]
fn unrelated_failures_keep_their_original_message() {
    assert!(missing_model_remediation(LocalRuntimeKind::Ollama, 500, "boom", "m", "u").is_none());
    assert!(
        missing_model_remediation(LocalRuntimeKind::Ollama, 404, "route not found", "m", "u")
            .is_none()
    );
    assert!(
        missing_model_remediation(LocalRuntimeKind::Ollama, 401, "model not found", "m", "u")
            .is_none()
    );
}

#[test]
fn context_overflow_is_recognised_across_provider_phrasings() {
    assert!(is_context_overflow(
        400,
        "This model's maximum context length is 4096 tokens, however you requested 5000"
    ));
    assert!(is_context_overflow(413, "prompt is too long"));
    assert!(is_context_overflow(
        400,
        "Please reduce the length of the messages"
    ));
    // Not an overflow.
    assert!(!is_context_overflow(400, "invalid api key"));
    assert!(!is_context_overflow(401, "maximum context length exceeded"));
}

#[test]
fn tools_rejections_are_told_apart_from_tool_choice_rejections() {
    assert!(mentions_tools_unsupported(
        "registry.ollama.ai/library/gemma3 does not support tools"
    ));
    assert!(mentions_tools_unsupported("unknown parameter: 'tools'"));
    // `tool_choice` has its own latch; flipping the tools latch for it would
    // disable native tools on a server that supports them fine.
    assert!(!mentions_tools_unsupported(
        "invalid parameter: tool_choice must be a string"
    ));
    assert!(!mentions_tools_unsupported("rate limited"));
}
