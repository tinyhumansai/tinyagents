//! Unit tests for provider construction, CLI argument/stdin building,
//! transcript rendering, system-prompt coalescing, and NDJSON response
//! assembly (including the timeout and error-propagation paths).

use super::*;
#[cfg(unix)]
use tinyinference_llm::tool::ToolCall;

#[test]
fn provider_constructs_with_default_config() {
    let config = ClaudeAgentSdkConfig::default();
    let provider = ClaudeAgentSdkProvider::new(config);
    assert_eq!(provider.config.binary, "claude");
    assert_eq!(provider.config.default_model, "claude-sonnet-4-6");
}

#[test]
fn config_default_disabled() {
    let config = ClaudeAgentSdkConfig::default();
    assert!(!config.enabled);
    assert!(config.max_budget_usd.is_none());
}

#[test]
fn large_request_is_delivered_over_stdin_instead_of_argv() {
    let system_prompt = "system instruction\n".repeat(2_500);
    assert!(system_prompt.len() > 32_767);

    let invocation = build_invocation(Some(&system_prompt), "hello", "claude-sonnet-4-6", None);

    assert_eq!(
        invocation.args,
        [
            "-p",
            "--model",
            "claude-sonnet-4-6",
            "--output-format",
            "stream-json",
            "--no-color"
        ]
    );
    assert!(
        !invocation
            .args
            .iter()
            .any(|arg| arg.contains(&system_prompt))
    );
    assert_eq!(
        invocation.stdin,
        format!("[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\nhello")
    );
}

#[test]
fn invocation_preserves_plain_message_and_budget_flags() {
    let invocation = build_invocation(None, "hello", "claude-opus-4-6", Some(1.25));

    assert_eq!(invocation.stdin, "hello");
    assert_eq!(
        &invocation.args[6..],
        ["--max-turns", "10", "--budget", "1.2500"]
    );
}

#[test]
fn spawn_error_message_includes_the_os_source() {
    let source = std::io::Error::from_raw_os_error(206);
    let error = spawn_error(r"C:\Users\test\.local\bin\claude.exe", source);

    assert!(error.to_string().contains("os error 206"));
    assert_eq!(
        error.chain().count(),
        2,
        "io::Error source must be preserved"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn provider_pipes_large_request_to_cli_stdin() {
    let dir = tempfile::tempdir().expect("tempdir");
    let script = dir.path().join("claude");
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/providers/claude_agent_sdk/fixtures/fake-claude");
    std::os::unix::fs::symlink(&fixture, &script).expect("link fake claude fixture");

    let config = ClaudeAgentSdkConfig {
        binary: script.display().to_string(),
        ..Default::default()
    };
    let provider = ClaudeAgentSdkProvider::new(config);
    let system_prompt = "system instruction\n".repeat(2_500);

    let output = provider
        .invoke(
            &(),
            ModelRequest::new(vec![
                Message::system(&system_prompt),
                Message::user("hello"),
            ])
            .with_model("claude-sonnet-4-6"),
        )
        .await
        .expect("fake claude response")
        .text();

    assert_eq!(output, "captured");
    assert_eq!(
        std::fs::read_to_string(format!("{}.stdin", script.display())).expect("captured stdin"),
        format!("[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\nhello")
    );
}

#[tokio::test]
async fn provider_spawn_error_includes_the_os_source() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = ClaudeAgentSdkConfig {
        binary: dir.path().join("missing-claude").display().to_string(),
        ..Default::default()
    };
    let provider = ClaudeAgentSdkProvider::new(config);

    let error = provider
        .invoke_cli(None, "hello", "claude-sonnet-4-6", None)
        .await
        .expect_err("missing binary must fail");

    assert!(error.to_string().contains("failed to spawn claude binary"));
    assert!(error.to_string().contains("os error"));
    assert_eq!(
        error.chain().count(),
        2,
        "io::Error source must be preserved"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn chat_model_uses_prompt_guided_protocol_and_model_override() {
    use std::os::unix::fs::PermissionsExt;
    use tinyinference_llm::tool::ToolSchema;

    let dir = tempfile::tempdir().expect("tempdir");
    let script = dir.path().join("claude");
    let staging = dir.path().join("claude.staging");
    std::fs::write(
            &staging,
            r#"#!/bin/sh
cat > "$0.stdin"
printf '%s\n' "$@" > "$0.args"
printf '%s\n' '{"type":"result","result":"Calling.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"query\":\"needle\"}}</tool_call>","is_error":false}'
"#,
        )
        .expect("write fake claude");
    std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o700))
        .expect("make fake claude executable");
    std::fs::rename(&staging, &script).expect("publish fake claude");

    let config = ClaudeAgentSdkConfig {
        binary: script.display().to_string(),
        ..Default::default()
    };
    let provider = ClaudeAgentSdkProvider::for_model(config, "profile-model");
    let mut prior_call = ModelResponse::assistant("calling").message;
    prior_call.tool_calls.push(ToolCall {
        id: "call-1".to_string(),
        name: "lookup".to_string(),
        arguments: serde_json::json!({"query": "needle"}),
        invalid: None,
    });
    let request = ModelRequest {
        messages: vec![
            Message::system("Base system"),
            Message::user("original question"),
            Message::Assistant(prior_call),
            Message::tool("call-1", "first result"),
            Message::tool("call-2", "second result"),
        ],
        tools: vec![ToolSchema::new(
            "lookup",
            "looks up data",
            serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } }
            }),
        )],
        model: Some("request-model".to_string()),
        ..Default::default()
    };

    let response = provider
        .invoke(&(), request)
        .await
        .expect("fake claude response");

    assert_eq!(response.text(), "Calling.");
    assert_eq!(response.message.tool_calls.len(), 1);
    assert_eq!(response.message.tool_calls[0].name, "lookup");
    assert_eq!(
        response.message.tool_calls[0].arguments,
        serde_json::json!({"query": "needle"})
    );
    let stdin =
        std::fs::read_to_string(format!("{}.stdin", script.display())).expect("captured stdin");
    assert!(stdin.contains("Base system"));
    assert!(
        stdin.contains("[USER]\noriginal question\n[/USER]"),
        "unexpected CLI stdin: {stdin:?}"
    );
    assert!(
        stdin.contains("[ASSISTANT]\ncalling\n<tool_call>"),
        "the assistant transcript must retain its text before the prior tool call: {stdin:?}"
    );
    assert!(stdin.contains("## Tool Use Protocol"));
    let prior_call = stdin
        .lines()
        .find_map(|line| {
            line.strip_prefix("<tool_call>")?
                .strip_suffix("</tool_call>")
        })
        .map(|body| serde_json::from_str::<serde_json::Value>(body).expect("tool call JSON"));
    assert_eq!(
        prior_call,
        Some(serde_json::json!({"name": "lookup", "arguments": {"query": "needle"}})),
        "prior structured tool call must survive in CLI stdin: {stdin:?}"
    );
    // Results are replayed under the protocol crate's envelope, keyed by the
    // call id they answer.
    assert!(
        stdin.contains("[Tool results]\n<tool_result id=\"call-1\">\nfirst result\n</tool_result>"),
        "unexpected CLI stdin: {stdin:?}"
    );
    assert!(stdin.contains("<tool_result id=\"call-2\">\nsecond result\n</tool_result>"));
    let args =
        std::fs::read_to_string(format!("{}.args", script.display())).expect("captured args");
    assert!(args.contains("request-model"));
    assert_eq!(
        provider
            .profile()
            .and_then(|profile| profile.model.as_deref()),
        Some("profile-model")
    );
    assert_eq!(
        provider
            .profile()
            .and_then(|profile| profile.provider.as_deref()),
        Some("claude-agent-sdk")
    );
}

#[cfg(unix)]
#[tokio::test]
async fn inspection_invocation_uses_supported_cli_flags_and_caps_output_with_frozen_auth() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("tempdir");
    let script = dir.path().join("claude-inspection");
    let args_path = dir.path().join("args.txt");
    let input_path = dir.path().join("input.txt");
    let auth_path = dir.path().join("auth.txt");
    let cap_path = dir.path().join("cap.txt");
    let endpoint_path = dir.path().join("endpoint.txt");
    let home_path = dir.path().join("home.txt");
    let canonical_home_path = dir.path().join("canonical-home.txt");
    let cwd_path = dir.path().join("cwd.txt");
    let isolated_paths = dir.path().join("isolated-paths.txt");
    let state_path = dir.path().join("state.txt");
    let staging = dir.path().join("claude-inspection.staging");
    let source = format!(
        "#!/bin/sh\n\
         case \" $* \" in *\" --no-color \"*) echo 'unsupported --no-color' >&2; exit 64;; esac\n\
         case \" $* \" in *\" --output-format stream-json \"*)\n\
           case \" $* \" in *\" --verbose \"*) ;; *) echo 'stream-json requires --verbose' >&2; exit 65;; esac;;\n\
         esac\n\
         printf '%s\\n' \"$@\" > '{}'\n\
         cat > '{}'\n\
         printf '%s' \"${{ANTHROPIC_API_KEY-}}\" > '{}'\n\
         printf '%s' \"${{CLAUDE_CODE_MAX_OUTPUT_TOKENS-}}\" > '{}'\n\
         printf '%s' \"${{ANTHROPIC_BASE_URL-}}\" > '{}'\n\
         printf '%s' \"$HOME\" > '{}'\n\
         (cd \"$HOME\" && pwd -P) > '{}'\n\
         pwd > '{}'; printf '%s|' \"$TMPDIR\" \"$XDG_CONFIG_HOME\" \"$XDG_DATA_HOME\" \"$XDG_STATE_HOME\" \"$XDG_CACHE_HOME\" \"$USERPROFILE\" \"$APPDATA\" \"$LOCALAPPDATA\" \"$TEMP\" \"$TMP\" > '{}'\n\
         if [ -e \"$HOME/.claude\" ]; then printf yes > '{}'; else printf no > '{}'; fi\n\
         printf '%s\\n' '{{\"type\":\"result\",\"result\":\"OK\",\"is_error\":false}}'\n",
        args_path.display(),
        input_path.display(),
        auth_path.display(),
        cap_path.display(),
        endpoint_path.display(),
        home_path.display(),
        canonical_home_path.display(),
        cwd_path.display(),
        isolated_paths.display(),
        state_path.display(),
        state_path.display(),
    );
    std::fs::write(&staging, source).expect("write fake claude");
    std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o700))
        .expect("make fake claude executable");
    std::fs::rename(&staging, &script).expect("publish fake claude");

    let config = ClaudeAgentSdkConfig {
        binary: script.display().to_string(),
        ..Default::default()
    };
    let auth = ClaudeAgentSdkInspectionAuth::new(
        Some("dummy-inspection-key".into()),
        None,
        Some("https://anthropic.example.test".into()),
    )
    .expect("explicit key is required for inspection");
    let provider = ClaudeAgentSdkProvider::for_inspection(config, "model-for-inspection", auth);
    let response = provider
        .invoke(
            &(),
            ModelRequest::new(vec![Message::user("Reply OK")]).with_max_tokens(5),
        )
        .await
        .expect("fake CLI response");

    assert_eq!(response.text(), "OK");
    let args = std::fs::read_to_string(args_path).expect("captured arguments");
    for flag in [
        "--bare",
        "--no-session-persistence",
        "--tools",
        "--max-turns",
        "--model",
        "--system-prompt",
        "--verbose",
        "dontAsk",
        "model-for-inspection",
    ] {
        assert!(
            args.contains(flag),
            "missing inspection argument {flag}: {args:?}"
        );
    }
    assert!(
        args.contains("--tools\n\n"),
        "built-in tools must be disabled: {args:?}"
    );
    assert!(
        args.contains("--max-turns\n1\n"),
        "one turn is allowed: {args:?}"
    );
    assert!(
        args.contains("--system-prompt\n\n"),
        "the CLI default prompt must be replaced: {args:?}"
    );
    assert!(
        !args.contains("--no-color"),
        "unsupported CLI arguments must be omitted: {args:?}"
    );
    assert!(
        !args.contains("dummy-inspection-key"),
        "credentials must never be passed as argv"
    );
    assert_eq!(
        std::fs::read_to_string(input_path).expect("captured prompt"),
        "Reply OK",
        "inspection receives no transcript or prior history"
    );
    assert_eq!(
        std::fs::read_to_string(auth_path).expect("captured auth"),
        "dummy-inspection-key"
    );
    assert_eq!(
        std::fs::read_to_string(cap_path).expect("captured token cap"),
        "5"
    );
    assert_eq!(
        std::fs::read_to_string(endpoint_path).expect("captured endpoint"),
        "https://anthropic.example.test"
    );
    let cli_home = std::fs::read_to_string(home_path).expect("captured isolated HOME");
    assert!(cli_home.starts_with(std::env::temp_dir().to_str().unwrap()));
    let cli_cwd = std::fs::read_to_string(cwd_path).expect("captured isolated working directory");
    let canonical_home = std::fs::read_to_string(canonical_home_path)
        .expect("captured isolated HOME resolved while it existed");
    assert_eq!(
        cli_cwd.trim(),
        canonical_home.trim(),
        "the CLI must not discover repository instructions from the caller's working directory"
    );
    let isolated = std::fs::read_to_string(isolated_paths).expect("captured isolated user paths");
    let isolated_paths = isolated.split('|').collect::<Vec<_>>();
    assert_eq!(isolated_paths.len(), 11, "all path variables were captured");
    assert!(
        isolated_paths[..5].iter().all(|path| *path == cli_home),
        "temporary and XDG profile paths must stay private: {isolated_paths:?}"
    );
    #[cfg(windows)]
    assert!(
        isolated_paths[5..10].iter().all(|path| *path == cli_home),
        "Windows user and temporary profile paths must stay private: {isolated_paths:?}"
    );
    #[cfg(unix)]
    assert!(
        isolated_paths[5..10].iter().all(|path| path.is_empty()),
        "Windows-only profile variables must not be inherited on Unix: {isolated_paths:?}"
    );
    assert_eq!(
        std::fs::read_to_string(state_path).expect("captured session state"),
        "no",
        "inspection must not expose or load an existing CLI login/session"
    );
}

#[test]
fn inspection_auth_rejects_opaque_cli_managed_login() {
    let result = ClaudeAgentSdkInspectionAuth::new(None, None, None);
    assert!(
        result.is_err(),
        "inspection cannot read the CLI's saved login"
    );
}

#[test]
fn inspection_child_environment_is_isolated_from_platform_profiles() {
    let home = tempfile::tempdir().expect("private inspection home");
    let private_home = home.path().to_string_lossy().into_owned();
    let auth = ClaudeAgentSdkInspectionAuth::new(
        Some("dummy-inspection-key".into()),
        None,
        Some("https://anthropic.example.test".into()),
    )
    .expect("explicit inspection key");
    let mut command = tokio::process::Command::new("claude");
    configure_inspection_child(&mut command, &auth, home.path(), 5);

    let environment = command
        .as_std()
        .get_envs()
        .filter_map(|(name, value)| {
            value.map(|value| {
                (
                    name.to_string_lossy().into_owned(),
                    value.to_string_lossy().into_owned(),
                )
            })
        })
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(environment.get("HOME"), Some(&private_home));
    assert_eq!(environment.get("TMPDIR"), Some(&private_home));
    assert_eq!(
        environment.get("ANTHROPIC_API_KEY"),
        Some(&"dummy-inspection-key".into())
    );
    assert_eq!(
        environment.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
        Some(&"5".into())
    );
    assert!(
        environment.contains_key("PATH"),
        "the configured CLI remains launchable"
    );

    #[cfg(not(windows))]
    for key in [
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
    ] {
        assert_eq!(
            environment.get(key),
            Some(&private_home),
            "{key} must be private"
        );
    }
    #[cfg(windows)]
    for key in ["USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"] {
        assert_eq!(
            environment.get(key),
            Some(&private_home),
            "{key} must be private"
        );
    }
}

#[test]
fn inspection_child_environment_passes_an_explicit_gateway_bearer_without_an_api_key() {
    let home = tempfile::tempdir().expect("private inspection home");
    let private_home = home.path().to_string_lossy().into_owned();
    let auth = ClaudeAgentSdkInspectionAuth::new(
        None,
        Some("dummy-gateway-bearer".into()),
        Some("https://gateway.example.test".into()),
    )
    .expect("explicit gateway bearer");
    let mut command = tokio::process::Command::new("claude");
    configure_inspection_child(&mut command, &auth, home.path(), 5);

    let environment = command
        .as_std()
        .get_envs()
        .filter_map(|(name, value)| {
            value.map(|value| {
                (
                    name.to_string_lossy().into_owned(),
                    value.to_string_lossy().into_owned(),
                )
            })
        })
        .collect::<std::collections::HashMap<_, _>>();

    assert_eq!(
        environment.get("ANTHROPIC_AUTH_TOKEN"),
        Some(&"dummy-gateway-bearer".into())
    );
    assert!(!environment.contains_key("ANTHROPIC_API_KEY"));
    assert_eq!(
        environment.get("ANTHROPIC_BASE_URL"),
        Some(&"https://gateway.example.test".into())
    );
    assert_eq!(environment.get("HOME"), Some(&private_home));
    assert_eq!(environment.get("TMPDIR"), Some(&private_home));
    assert_eq!(
        environment.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
        Some(&"5".into())
    );
    assert!(environment.contains_key("PATH"));
}

#[test]
fn inspection_binary_path_is_resolved_before_switching_to_private_home() {
    let caller_dir = std::env::current_dir().expect("current directory");
    assert_eq!(
        inspection_binary_path("tools/claude").expect("relative configured binary"),
        caller_dir.join("tools/claude")
    );
    assert_eq!(
        inspection_binary_path("claude").expect("PATH binary"),
        std::path::PathBuf::from("claude"),
        "a bare executable name must continue to resolve through PATH"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn inspection_invocation_clamps_requested_output_cap_to_five_tokens() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("tempdir");
    let script = dir.path().join("claude-inspection");
    let cap_path = dir.path().join("cap.txt");
    let request_path = dir.path().join("request.txt");
    let staging = dir.path().join("claude-inspection.staging");
    let source = format!(
        "#!/bin/sh\ncat > '{}'\nprintf '%s' \"${{CLAUDE_CODE_MAX_OUTPUT_TOKENS-}}\" > '{}'\nprintf '%s\\n' '{{\"type\":\"result\",\"result\":\"OK\",\"is_error\":false}}'\n",
        request_path.display(),
        cap_path.display(),
    );
    std::fs::write(&staging, source).expect("write fake claude");
    std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o700))
        .expect("make fake claude executable");
    std::fs::rename(&staging, &script).expect("publish fake claude");

    let config = ClaudeAgentSdkConfig {
        binary: script.display().to_string(),
        ..Default::default()
    };
    let auth = ClaudeAgentSdkInspectionAuth::new(Some("dummy-key".into()), None, None)
        .expect("explicit key is required");
    let provider = ClaudeAgentSdkProvider::for_inspection(config, "model", auth);
    let request_text = "Reply OK ".repeat(512 * 1024);
    provider
        .invoke(
            &(),
            ModelRequest::new(vec![Message::user(&request_text)]).with_max_tokens(64),
        )
        .await
        .expect("fake CLI response");

    assert_eq!(
        std::fs::read_to_string(cap_path).expect("captured cap"),
        "5",
        "caller-supplied limits cannot raise the inspection cap"
    );
    assert_eq!(
        std::fs::read_to_string(request_path).expect("fake CLI consumed stdin"),
        request_text,
        "the full request must reach the CLI before its response is emitted"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn inspection_invocation_rejects_tool_requests_before_spawning() {
    use std::os::unix::fs::PermissionsExt;
    use tinyinference_llm::tool::ToolSchema;

    let dir = tempfile::tempdir().expect("tempdir");
    let script = dir.path().join("claude-inspection");
    let marker_path = dir.path().join("spawned.txt");
    let staging = dir.path().join("claude-inspection.staging");
    std::fs::write(
        &staging,
        format!("#!/bin/sh\nprintf spawned > '{}'\n", marker_path.display()),
    )
    .expect("write fake claude");
    std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o700))
        .expect("make fake claude executable");
    std::fs::rename(&staging, &script).expect("publish fake claude");

    let config = ClaudeAgentSdkConfig {
        binary: script.display().to_string(),
        ..Default::default()
    };
    let auth = ClaudeAgentSdkInspectionAuth::new(Some("dummy-key".into()), None, None)
        .expect("explicit key is required");
    let provider = ClaudeAgentSdkProvider::for_inspection(config, "model", auth);
    let mut request = ModelRequest::new(vec![Message::user("Reply OK")]);
    request.tools.push(ToolSchema::new(
        "should-never-run",
        "test tool",
        serde_json::json!({"type": "object"}),
    ));

    let error = provider
        .invoke(&(), request)
        .await
        .expect_err("inspection must refuse a request that could execute a tool");
    assert!(matches!(error, tinyinference_llm::Error::Unsupported(_)));
    assert!(
        !marker_path.exists(),
        "rejected calls must not spawn the CLI"
    );
}

#[test]
fn cache_identity_includes_selected_model() {
    let config = ClaudeAgentSdkConfig::default();
    let first = ClaudeAgentSdkProvider::for_model(config.clone(), "model-a");
    let second = ClaudeAgentSdkProvider::for_model(config, "model-b");

    assert_ne!(first.cache_identity(), second.cache_identity());
}

/// The CLI takes one `--system-prompt`, so every system message has to be
/// folded into it. Taking only the first dropped the artifact contents list and
/// the turn-cap wrap-up, which are appended as system messages exactly so the
/// context ladder cannot remove them (codex on #6068).
#[test]
fn every_system_message_reaches_the_cli_prompt() {
    let messages = vec![
        Message::system("you are a helpful agent"),
        Message::user("fetch the issues"),
        Message::system("## Stored results from this turn\n\n- `file_read` → `a.json`"),
    ];

    let system = coalesce_system_prompt(&messages).expect("a system prompt");

    assert!(
        system.contains("you are a helpful agent"),
        "the persona must survive: {system}"
    );
    assert!(
        system.contains("Stored results from this turn"),
        "a later system message must not be dropped: {system}"
    );
    assert!(
        system.find("you are a helpful agent") < system.find("Stored results"),
        "order must be preserved so the persona still leads: {system}"
    );
}

#[test]
fn a_conversation_with_no_system_message_yields_none() {
    let messages = vec![Message::user("hello")];
    assert!(coalesce_system_prompt(&messages).is_none());
    // Blank system messages are not a prompt either.
    let blank = vec![Message::system("   ")];
    assert!(coalesce_system_prompt(&blank).is_none());
}
