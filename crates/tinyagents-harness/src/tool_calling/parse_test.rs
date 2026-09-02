use super::*;

#[test]
fn parse_argument_helpers_cover_string_non_string_and_missing_values() {
    assert_eq!(
        parse_arguments_value(Some(&serde_json::json!("{\"value\":1}"))),
        serde_json::json!({ "value": 1 })
    );
    assert_eq!(
        parse_arguments_value(Some(&serde_json::json!("not-json"))),
        serde_json::json!({})
    );
    assert_eq!(
        parse_arguments_value(Some(&serde_json::json!({ "value": 2 }))),
        serde_json::json!({ "value": 2 })
    );
    assert_eq!(parse_arguments_value(None), serde_json::json!({}));
}

#[test]
fn parse_tool_call_value_supports_function_shape_flat_shape_and_invalid_names() {
    let function_shape = serde_json::json!({
        "function": {
            "name": "shell",
            "arguments": "{\"command\":\"ls\"}"
        }
    });
    let parsed = parse_tool_call_value(&function_shape).expect("function call should parse");
    assert_eq!(parsed.name, "shell");
    assert_eq!(parsed.arguments, serde_json::json!({ "command": "ls" }));

    let flat_shape = serde_json::json!({
        "name": "echo",
        "arguments": { "value": "hi" }
    });
    let parsed = parse_tool_call_value(&flat_shape).expect("flat call should parse");
    assert_eq!(parsed.name, "echo");
    assert_eq!(parsed.arguments, serde_json::json!({ "value": "hi" }));

    assert!(parse_tool_call_value(&serde_json::json!({ "name": "   " })).is_none());
    assert!(parse_tool_call_value(&serde_json::json!({ "function": {} })).is_none());
}

#[test]
fn parse_tool_call_value_accepts_argument_key_aliases() {
    // Correct name but the model used `args`/`parameters` instead of the
    // canonical `arguments` — recover the call rather than drop it and burn
    // an agent iteration (bug-report-2026-05-26 A3).
    let with_args = serde_json::json!({ "name": "echo", "args": { "value": "hi" } });
    let parsed = parse_tool_call_value(&with_args).expect("args alias should parse");
    assert_eq!(parsed.name, "echo");
    assert_eq!(parsed.arguments, serde_json::json!({ "value": "hi" }));

    let with_parameters = serde_json::json!({
        "function": { "name": "shell", "parameters": "{\"command\":\"ls\"}" }
    });
    let parsed = parse_tool_call_value(&with_parameters).expect("parameters alias should parse");
    assert_eq!(parsed.name, "shell");
    assert_eq!(parsed.arguments, serde_json::json!({ "command": "ls" }));

    // Name stays strict: an arg alias without a recognized name key is not
    // a tool call (guards the whole-response JSON parse path).
    assert!(parse_tool_call_value(&serde_json::json!({ "tool": "echo", "args": {} })).is_none());
}

#[test]
fn whole_response_singleton_ignores_generic_arg_aliases() {
    // A plain JSON answer that happens to carry a `name` plus a generic,
    // object-valued `input`. Tagged contexts widen `input` into arguments…
    let answer = serde_json::json!({ "name": "Alice", "input": { "value": "hi" } });
    let tagged = parse_tool_calls_from_json_value(&answer);
    assert_eq!(tagged.len(), 1);
    assert_eq!(tagged[0].arguments, serde_json::json!({ "value": "hi" }));

    // …but the whole-response (bare singleton) path must treat this as plain
    // text, not a tool call: it carries no canonical `arguments` marker, only
    // a `name` that happens to match a tool (CodeRabbit, #2683).
    let whole = parse_tool_calls_from_json_value_aliased(&answer, false);
    assert!(
        whole.is_empty(),
        "bare whole-response object without canonical `arguments` must not dispatch a tool call"
    );

    // A bare object WITH the canonical `arguments` key is still recognized on
    // the whole-response path — `arguments` is the explicit tool-call marker.
    let bare_call = serde_json::json!({ "name": "echo", "arguments": { "value": "hi" } });
    let calls = parse_tool_calls_from_json_value_aliased(&bare_call, false);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "echo");
    assert_eq!(calls[0].arguments, serde_json::json!({ "value": "hi" }));

    // The `tool_calls`-keyed envelope is an explicit marker and stays
    // permissive even when aliases are forbidden for bare objects.
    let envelope = serde_json::json!({
        "tool_calls": [ { "name": "echo", "input": { "value": "hi" } } ]
    });
    let calls = parse_tool_calls_from_json_value_aliased(&envelope, false);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "echo");
    assert_eq!(calls[0].arguments, serde_json::json!({ "value": "hi" }));
}

#[test]
fn parse_tool_calls_from_json_value_handles_tool_calls_array_arrays_and_singletons() {
    let wrapped = serde_json::json!({
        "tool_calls": [
            { "name": "echo", "arguments": { "value": "one" } },
            { "function": { "name": "shell", "arguments": "{\"command\":\"pwd\"}" } }
        ],
        "content": "assistant text"
    });
    let calls = parse_tool_calls_from_json_value(&wrapped);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "echo");
    assert_eq!(calls[1].name, "shell");

    let array = serde_json::json!([
        { "name": "echo", "arguments": { "value": "two" } },
        { "name": "   " }
    ]);
    let calls = parse_tool_calls_from_json_value(&array);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, serde_json::json!({ "value": "two" }));

    let single = serde_json::json!({ "name": "echo", "arguments": { "value": "three" } });
    let calls = parse_tool_calls_from_json_value(&single);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "echo");
}

#[test]
fn tag_and_json_extractors_cover_common_edge_cases() {
    assert_eq!(
        find_first_tag("hi <invoke>there", &["<tool_call>", "<invoke>"]),
        Some((3, "<invoke>"))
    );
    assert_eq!(
        matching_tool_call_close_tag("<toolcall>"),
        Some("</toolcall>")
    );
    assert_eq!(matching_tool_call_close_tag("<nope>"), None);

    let extracted = extract_first_json_value_with_end(" text {\"ok\":true} trailing ")
        .expect("json should be found");
    assert_eq!(extracted.0, serde_json::json!({ "ok": true }));
    assert!(extracted.1 > 0);

    assert_eq!(
        strip_leading_close_tags(" </tool_call>  </invoke> hi "),
        "hi "
    );
    assert_eq!(strip_leading_close_tags("plain"), "plain");

    let values = extract_json_values("before {\"a\":1} [1,2] after");
    assert_eq!(
        values,
        vec![serde_json::json!({ "a": 1 }), serde_json::json!([1, 2])]
    );

    assert_eq!(
        find_json_end("  {\"a\":\"}\"}tail"),
        Some("  {\"a\":\"}\"}".len())
    );
    assert_eq!(find_json_end("[1,2,3]"), None);
}

#[test]
fn glm_helpers_parse_aliases_urls_and_commands() {
    assert_eq!(map_glm_tool_alias("browser_open"), "shell");
    assert_eq!(map_glm_tool_alias("http"), "http_request");
    assert_eq!(map_glm_tool_alias("custom_tool"), "custom_tool");

    assert_eq!(
        build_curl_command("https://example.com?q=1"),
        Some("curl -s 'https://example.com?q=1'".into())
    );
    assert_eq!(
        build_curl_command("https://exa'mple.com"),
        Some("curl -s 'https://exa'\\''mple.com'".into())
    );
    assert!(build_curl_command("ftp://example.com").is_none());
    assert!(build_curl_command("https://example.com/has space").is_none());

    let calls = parse_glm_style_tool_calls(
        "browser_open/url>https://example.com\nhttp_request/url>https://api.example.com\nplain text\nhttps://rust-lang.org",
    );
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].0, "shell");
    assert_eq!(calls[1].0, "http_request");
    assert_eq!(calls[2].0, "shell");
}

#[test]
fn parse_tool_calls_supports_native_json_xml_markdown_and_glm_formats() {
    let native = serde_json::json!({
        "content": "native text",
        "tool_calls": [
            { "name": "echo", "arguments": { "value": "one" } }
        ]
    })
    .to_string();
    let (text, calls) = parse_tool_calls(&native);
    assert_eq!(text, "native text");
    assert_eq!(calls.len(), 1);

    let xml = "before\n<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"value\":\"two\"}}\n</tool_call>\nafter";
    let (text, calls) = parse_tool_calls(xml);
    assert_eq!(text, "before\nafter");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, serde_json::json!({ "value": "two" }));

    let unclosed = "<invoke>{\"name\":\"echo\",\"arguments\":{\"value\":\"three\"}}</invoke>";
    let (text, calls) = parse_tool_calls(unclosed);
    assert!(text.is_empty());
    assert_eq!(calls.len(), 1);

    let markdown =
        "lead\n```tool_call\n{\"name\":\"echo\",\"arguments\":{\"value\":\"four\"}}\n```\ntrail";
    let (text, calls) = parse_tool_calls(markdown);
    assert_eq!(text, "lead\ntrail");
    assert_eq!(calls.len(), 1);

    let glm = "shell/command>ls -la";
    let (text, calls) = parse_tool_calls(glm);
    assert!(text.is_empty());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "shell");
}

// ── lenient recovery of pipe-garbled <tool_call> tags ────────────────────────

#[test]
fn garbled_pipe_tags_with_json_body_and_call_prefix_parse() {
    // Exact shape seen from a small Composio sub-agent: native `<|…|>` sentinel
    // pipes leaked into the tags (`<|tool_call>` / `<tool_call|>`) and the body
    // is prefixed with `call:`. Without the normalizer this drops silently and
    // the tool never runs; with it, the real call is recovered.
    let garbled = r#"<|tool_call>call:{"name": "GMAIL_LIST_THREADS", "arguments": {"query": "\"University of Colorado\"", "verbose": true}}<tool_call|>"#;
    let (_text, calls) = parse_tool_calls(garbled);
    assert_eq!(calls.len(), 1, "expected the garbled call to be recovered");
    assert_eq!(calls[0].name, "GMAIL_LIST_THREADS");
    assert_eq!(calls[0].arguments["query"], "\"University of Colorado\"");
    assert_eq!(calls[0].arguments["verbose"], true);
}

#[test]
fn garbled_pipe_tags_recover_multiple_parallel_calls() {
    let garbled = concat!(
        r#"<|tool_call>call:{"name": "GMAIL_LIST_THREADS", "arguments": {"query": "a"}}<tool_call|>"#,
        r#"<|tool_call>call:{"name": "GMAIL_LIST_THREADS", "arguments": {"query": "b"}}<tool_call|>"#,
    );
    let (_t, calls) = parse_tool_calls(garbled);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].arguments["query"], "a");
    assert_eq!(calls[1].arguments["query"], "b");
}

#[test]
fn normalize_leaves_clean_output_untouched() {
    // No piped marker → cheap Borrowed no-op, and a canonical call still parses.
    let clean = r#"<tool_call>{"name":"echo","arguments":{}}</tool_call>"#;
    assert!(matches!(
        normalize_garbled_tool_call_tags(clean),
        std::borrow::Cow::Borrowed(_)
    ));
    let (_t, calls) = parse_tool_calls(clean);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "echo");
}

#[test]
fn normalize_repairs_open_and_close_pipe_variants() {
    // Leading pipe → open; trailing pipe (no slash) → close.
    assert_eq!(
        normalize_garbled_tool_call_tags("<|tool_call>BODY<tool_call|>").as_ref(),
        "<tool_call>BODY</tool_call>"
    );
    // Slash-bearing close variants normalize too.
    assert_eq!(
        normalize_garbled_tool_call_tags("<tool_call>b</tool_call|>").as_ref(),
        "<tool_call>b</tool_call>"
    );
}

#[test]
fn normalize_pairs_symmetric_both_pipe_tags() {
    // `<|tool_call|>` on BOTH sides — a hardcoded open/close map can't tell them
    // apart; positional pairing does (1st = open, 2nd = close).
    let garbled = r#"<|tool_call|>{"name":"echo","arguments":{"msg":"hi"}}<|tool_call|>"#;
    let (_t, calls) = parse_tool_calls(garbled);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "echo");
    assert_eq!(calls[0].arguments["msg"], "hi");
}

#[test]
fn normalize_leaves_pformat_pipe_args_untouched() {
    // Pipes appear in P-Format BODIES (`name[a|b]`), not the tags — the
    // garbled-tag guard must not touch a well-formed positional call.
    let clean = "<tool_call>get_weather[London|metric]</tool_call>";
    assert!(matches!(
        normalize_garbled_tool_call_tags(clean),
        std::borrow::Cow::Borrowed(_)
    ));
}

// ── #5119: recover the Kimi `NAME{…}` argument-sentinel body ─────────────────

#[test]
fn garbled_kimi_name_brace_body_with_quote_sentinels_parses() {
    // The EXACT shape observed from `integrations_agent`/`burst-v1` (Kimi-K2) on
    // the post-contract retry: garbled tags PLUS a `NAME{…}` body with unquoted
    // keys and the `<|"|>` argument-quote sentinel around string values. Before
    // the recovery this parsed to zero tool calls (the tag fix alone left an
    // unparseable body), so GMAIL_FETCH_EMAILS never ran and the turn looped.
    let garbled = r#"<|tool_call>call:GMAIL_FETCH_EMAILS{label_ids:[<|"|>INBOX<|"|>],max_results:1,verbose:true}<tool_call|>"#;
    let (_text, calls) = parse_tool_calls(garbled);
    assert_eq!(calls.len(), 1, "the garbled Kimi call must be recovered");
    assert_eq!(calls[0].name, "GMAIL_FETCH_EMAILS");
    assert_eq!(
        calls[0].arguments["label_ids"],
        serde_json::json!(["INBOX"])
    );
    assert_eq!(calls[0].arguments["max_results"], 1);
    assert_eq!(calls[0].arguments["verbose"], true);
}

#[test]
fn garbled_kimi_name_brace_body_integer_only_parses() {
    // The integer-only variant (no string values → no `<|"|>` sentinel, but the
    // body is still the unparseable `NAME{unquoted-keys}` shape). Observed as
    // `{max_results:5}` on the staging repro.
    let garbled = r#"<|tool_call>call:GMAIL_FETCH_EMAILS{max_results:5}<tool_call|>"#;
    let (_text, calls) = parse_tool_calls(garbled);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "GMAIL_FETCH_EMAILS");
    assert_eq!(calls[0].arguments["max_results"], 5);
}

#[test]
fn recover_sentinel_body_leaves_canonical_and_pformat_untouched() {
    // Canonical JSON body → empty leading name → not our shape → None.
    assert!(recover_sentinel_tool_call_body(r#"{"name":"echo","arguments":{}}"#).is_none());
    // P-Format body (`NAME[…]`, no brace) → None.
    assert!(recover_sentinel_tool_call_body("get_weather[London|metric]").is_none());
    // Trailing garbage after the object → not a clean `NAME{…}` → None.
    assert!(recover_sentinel_tool_call_body("FOO{a:1} trailing").is_none());
}

#[test]
fn quote_bare_json_object_keys_respects_string_values() {
    // A `,ident:` sequence INSIDE a string value must not be quoted; only
    // structural keys after `{`/`,` are rewritten.
    let out = quote_bare_json_object_keys(r#"{query:"from:john,to:x",n:1}"#);
    assert_eq!(out, r#"{"query":"from:john,to:x","n":1}"#);
    // Parses as strict JSON with the value preserved verbatim.
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["query"], "from:john,to:x");
    assert_eq!(v["n"], 1);
}

#[test]
fn quote_bare_json_object_keys_leaves_array_literals_unquoted() {
    // Bare literals inside arrays must not be quoted. A comma inside an array
    // should not trigger `expect_key = true` because arrays do not have keys.
    let out = quote_bare_json_object_keys(r#"{flags:[true,false],n:null}"#);
    assert_eq!(out, r#"{"flags":[true,false],"n":null}"#);
    // Parses as strict JSON with the values preserved as booleans and null.
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["flags"], serde_json::json!([true, false]));
    assert_eq!(v["n"], serde_json::Value::Null);
}

#[test]
fn parse_tool_calls_with_pformat_preserves_multi_call_tag_bodies() {
    // A single <tool_call> tag body can hold multiple JSON calls (e.g., two
    // adjacent objects or a {"tool_calls":[...]} envelope). The ordinal pairing
    // must not drop them when re-parsing the tag body.
    use crate::tool_calling::PFormatRegistry;

    let registry = PFormatRegistry::new();
    let response = r#"<tool_call>{"name":"get_weather","arguments":{"city":"London"}}{"name":"get_time","arguments":{"tz":"UTC"}}</tool_call>"#;
    let (_, calls) = parse_tool_calls_with_pformat(response, &registry);

    assert_eq!(
        calls.len(),
        2,
        "both JSON calls in the tag body must be recovered"
    );
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments["city"], "London");
    assert_eq!(calls[1].name, "get_time");
    assert_eq!(calls[1].arguments["tz"], "UTC");
}

// ── Regression probe: mixed p-format + non-JSON tags ─────────────────────────

use crate::tool_calling::{PFormatRegistry, PFormatToolParams};

/// A p-format tag alongside a GLM-style sibling.
///
/// Once any tag yields a p-format call, the walk stops falling back to the
/// canonical parse, so every remaining tag is on its own. A GLM body
/// (`shell/command>ls -la`) is not JSON, so before the GLM fallback existed
/// this call was silently dropped — the agent lost a tool invocation it had
/// asked for and nothing reported it.
#[test]
fn a_pformat_tag_does_not_suppress_a_sibling_glm_tag() {
    let mut reg = PFormatRegistry::new();
    reg.insert(
        "echo".to_string(),
        PFormatToolParams::from_schema(&serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } }
        })),
    );

    let response = concat!(
        "<tool_call>echo[0|hello]</tool_call>\n",
        "<tool_call>shell/command>ls -la</tool_call>"
    );
    let (_narrative, calls) = parse_tool_calls_with_pformat(response, &reg);
    let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();

    assert!(
        names.contains(&"echo"),
        "the p-format call must survive: {names:?}"
    );
    assert_eq!(
        calls.len(),
        2,
        "the sibling non-JSON tag was dropped — got {names:?}"
    );
}

/// The same shape, but the sibling body is JSON inside a markdown fence.
#[test]
fn a_pformat_tag_does_not_suppress_a_sibling_fenced_json_tag() {
    let mut reg = PFormatRegistry::new();
    reg.insert(
        "echo".to_string(),
        PFormatToolParams::from_schema(&serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } }
        })),
    );

    let response = concat!(
        "<tool_call>echo[0|hello]</tool_call>\n",
        "<tool_call>\n```json\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}\n```\n</tool_call>"
    );
    let (_narrative, calls) = parse_tool_calls_with_pformat(response, &reg);
    let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();

    assert!(names.contains(&"echo"), "p-format call survives: {names:?}");
    assert!(
        names.contains(&"shell"),
        "the fenced-JSON sibling was dropped — got {names:?}"
    );
}

/// A JSON body that ALSO looks like GLM's `name/key>value` grammar must not
/// yield the call twice.
///
/// The GLM fallback runs only when the JSON path found nothing, and this is
/// what pins that ordering. If it ever ran unconditionally, a body containing
/// a `/` and a `>` inside a string value would be counted once as JSON and
/// again as GLM — the agent would execute the same tool twice.
#[test]
fn a_json_body_is_not_double_counted_by_the_glm_fallback() {
    let mut reg = PFormatRegistry::new();
    reg.insert(
        "echo".to_string(),
        PFormatToolParams::from_schema(&serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } }
        })),
    );

    let response = concat!(
        "<tool_call>echo[0|hello]</tool_call>\n",
        "<tool_call>{\"name\": \"shell\", \"arguments\": {\"command\": \"cat a/b>c\"}}</tool_call>"
    );
    let (_narrative, calls) = parse_tool_calls_with_pformat(response, &reg);
    let shell_calls = calls.iter().filter(|c| c.name == "shell").count();
    assert_eq!(
        shell_calls,
        1,
        "the JSON body was counted twice: {:?}",
        calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>()
    );
}

/// A tagged body may use the argument-key aliases.
///
/// A `<tool_call>` tag is an explicit tool-call marker, so `args` /
/// `parameters` / `input` are honoured inside it — unlike a bare top-level
/// object, where they are refused so a plain JSON answer cannot read as a
/// call. This pins that the tag path keeps the permissive behaviour: routing
/// it through `parse_tool_calls` instead would silently drop this call.
#[test]
fn a_tagged_body_still_honours_argument_key_aliases() {
    let mut reg = PFormatRegistry::new();
    reg.insert(
        "echo".to_string(),
        PFormatToolParams::from_schema(&serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } }
        })),
    );

    let response = concat!(
        "<tool_call>echo[0|hello]</tool_call>\n",
        "<tool_call>{\"name\": \"shell\", \"args\": {\"command\": \"ls\"}}</tool_call>"
    );
    let (_narrative, calls) = parse_tool_calls_with_pformat(response, &reg);
    let shell = calls
        .iter()
        .find(|c| c.name == "shell")
        .expect("the aliased tagged call must survive");
    assert_eq!(shell.arguments["command"], "ls");
}
