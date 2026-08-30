//! Unit tests for prompt-driven tool selection: verb classification,
//! abbreviation expansion, stopword filtering, and ranked overlap scoring.

use super::*;

fn tool<'a>(name: &'a str, desc: &'a str) -> SelectableTool<'a> {
    SelectableTool::new(name, desc)
}

fn github_sample() -> Vec<SelectableTool<'static>> {
    vec![
        tool(
            "GITHUB_CREATE_A_PULL_REQUEST",
            "Creates a pull request in a GitHub repository, requiring existing base and head branches.",
        ),
        tool(
            "GITHUB_CREATE_A_REVIEW_FOR_A_PULL_REQUEST",
            "Creates a pull request review, allowing approval, change requests, or comments.",
        ),
        tool(
            "GITHUB_CREATE_A_DEPLOYMENT_BRANCH_POLICY",
            "Creates a deployment branch or tag policy for an existing environment in a repository.",
        ),
        tool(
            "GITHUB_DELETE_A_REVIEW_COMMENT_FOR_A_PULL_REQUEST",
            "Deletes a review comment on a pull request.",
        ),
        tool(
            "GITHUB_FIND_PULL_REQUESTS",
            "Primary tool to find and search pull requests.",
        ),
        tool(
            "GITHUB_GET_A_PULL_REQUEST",
            "Retrieves a specific pull request by number.",
        ),
        tool(
            "GITHUB_LIST_ASSIGNEES",
            "Lists users who can be assigned to issues in a repository.",
        ),
    ]
}

#[test]
fn create_pr_ranks_create_a_pull_request_first() {
    let tools = github_sample();
    let idx = rank_tools_by_prompt("create a PR from my feature branch to main", &tools, 5);
    assert!(!idx.is_empty());
    // Top match must be a CREATE verb tool (not DELETE/GET).
    let top_name = tools[idx[0]].name;
    assert!(
        top_name.contains("CREATE") && top_name.contains("PULL_REQUEST"),
        "expected top match to be a CREATE + PULL_REQUEST tool, got {top_name}"
    );
    // The DELETE tool must not appear — verb gate should drop it.
    for &i in &idx {
        assert!(
            !tools[i].name.starts_with("GITHUB_DELETE"),
            "DELETE tool leaked past verb gate: {}",
            tools[i].name
        );
    }
}

#[test]
fn list_prs_ranks_find_pull_requests_first() {
    let tools = github_sample();
    let idx = rank_tools_by_prompt("list open PRs assigned to me", &tools, 5);
    assert!(!idx.is_empty());
    let top_name = tools[idx[0]].name;
    assert!(
        top_name == "GITHUB_FIND_PULL_REQUESTS" || top_name == "GITHUB_LIST_ASSIGNEES",
        "expected FIND_PULL_REQUESTS or LIST_ASSIGNEES on top, got {top_name}"
    );
}

/// Exact-ordering snapshot, captured from the pre-extraction implementation in
/// the OpenHuman host crate before this module was moved up. This is a
/// relevance ranker whose output decides which tools a model is shown, so a
/// scoring change is a silent behaviour change; pin the whole ordering, not
/// just the winner.
#[test]
fn ranking_order_matches_the_pre_extraction_snapshot() {
    let tools = github_sample();
    let cases: &[(&str, &[&str])] = &[
        (
            "create a PR from my feature branch to main",
            &[
                "GITHUB_CREATE_A_PULL_REQUEST",
                "GITHUB_CREATE_A_REVIEW_FOR_A_PULL_REQUEST",
                "GITHUB_CREATE_A_DEPLOYMENT_BRANCH_POLICY",
            ],
        ),
        (
            "list open PRs assigned to me",
            &["GITHUB_FIND_PULL_REQUESTS", "GITHUB_LIST_ASSIGNEES"],
        ),
        (
            "delete a review comment",
            &["GITHUB_DELETE_A_REVIEW_COMMENT_FOR_A_PULL_REQUEST"],
        ),
        // No MERGE-prefixed tool in the sample, and the gate drops every
        // non-merge verb, so this query is empty by construction.
        ("merge pull request 42", &[]),
    ];
    for (prompt, expected) in cases {
        let got: Vec<&str> = rank_tools_by_prompt(prompt, &tools, 10)
            .into_iter()
            .map(|i| tools[i].name)
            .collect();
        assert_eq!(&got, expected, "ranking drifted for prompt {prompt:?}");
    }
}

#[test]
fn empty_prompt_returns_empty() {
    let tools = github_sample();
    let idx = rank_tools_by_prompt("", &tools, 5);
    assert!(idx.is_empty());
}

#[test]
fn empty_catalogue_returns_empty() {
    assert!(rank_tools_by_prompt("create a PR", &[], 5).is_empty());
}

#[test]
fn abbreviation_expansion_works() {
    let qt = query_tokens("create a PR from feature branch");
    assert!(qt.contains("pr"));
    assert!(qt.contains("pull"));
    assert!(qt.contains("request"));
}

#[test]
fn stopwords_removed() {
    let qt = query_tokens("send the email to my manager");
    assert!(!qt.contains("the"));
    assert!(!qt.contains("to"));
    assert!(!qt.contains("my"));
    assert!(qt.contains("send"));
    assert!(qt.contains("email"));
    assert!(qt.contains("manager"));
}

#[test]
fn verb_detection_handles_aliases() {
    // Exact assertion, not `contains(Send) || contains(Create)`: the
    // regression this pins is specifically that `Send` must be retained
    // ALONGSIDE `Create` here, not merely that one of the two survives — an
    // implementation that suppresses `Send` whenever ANY verb is found would
    // still pass a permissive `||` assertion while reintroducing the exact
    // ranking regression (`SLACK_SEND_MESSAGE` falling out of the top-k).
    let v = detect_verbs("post a message to general channel");
    assert_eq!(v, HashSet::from([ToolVerb::Create, ToolVerb::Send]));

    let v = detect_verbs("delete all promotional emails");
    assert!(v.contains(&ToolVerb::Delete));

    let v = detect_verbs("merge pull request 42");
    assert!(v.contains(&ToolVerb::Merge));
}

#[test]
fn resource_noun_does_not_add_send_alongside_an_explicit_conflicting_verb() {
    // "read email" and "delete a message" must not ALSO detect Send from the
    // resource noun — only the explicit action verb should be present.
    let v = detect_verbs("read email");
    assert_eq!(v, HashSet::from([ToolVerb::Read]));

    let v = detect_verbs("delete a message");
    assert_eq!(v, HashSet::from([ToolVerb::Delete]));

    // No explicit verb at all: the resource noun alone may still imply Send.
    let v = detect_verbs("message support channel");
    assert!(v.contains(&ToolVerb::Send));
}

#[test]
fn compound_task_preserves_send_alongside_a_distant_unrelated_verb() {
    // A genuinely compound task: "find Alice" (List) AND "email her the
    // report" (Send) are two independent intents joined by "and", not one
    // verb governing the noun. Suppressing Send here (as a blanket
    // "any conflicting verb anywhere in the prompt" gate would) drops
    // GMAIL_SEND_EMAIL from the gated tool set entirely for a query that
    // genuinely needs it.
    let v = detect_verbs("find Alice and email her the report");
    assert!(
        v.contains(&ToolVerb::Send),
        "a distant, unrelated List verb must not suppress a compound task's \
         Send intent: {v:?}"
    );
    assert!(v.contains(&ToolVerb::List));

    // Still suppressed when the conflicting verb is close enough to read as
    // the noun's direct object, even with a determiner between them.
    let v = detect_verbs("delete that old email");
    assert_eq!(v, HashSet::from([ToolVerb::Delete]));
}

#[test]
fn tool_verb_handles_plurals() {
    assert_eq!(tool_verb("SLACK_DELETES_A_MESSAGE"), Some(ToolVerb::Delete));
    assert_eq!(
        tool_verb("GITHUB_CREATE_A_PULL_REQUEST"),
        Some(ToolVerb::Create)
    );
    assert_eq!(tool_verb("GMAIL_SEND_EMAIL"), Some(ToolVerb::Send));
    assert_eq!(tool_verb("NOTION_QUERY_DATABASE"), Some(ToolVerb::List));
    // Neutral — no verb prefix recognised
    assert_eq!(tool_verb("GITHUB_GIST_COMMENT"), None);
}

#[test]
fn tool_verb_classifies_canonical_lowercase_names() {
    // `ToolSchema::name` is canonical snake_case; comparing raw case against
    // the (uppercase) prefix tables previously left every lowercase name
    // unclassified, silently keeping the verb gate a no-op for real tool
    // catalogues.
    assert_eq!(
        tool_verb("github_delete_a_pull_request"),
        Some(ToolVerb::Delete)
    );
    assert_eq!(
        tool_verb("github_create_a_pull_request"),
        Some(ToolVerb::Create)
    );
    assert_eq!(tool_verb("gmail_send_email"), Some(ToolVerb::Send));
}

#[test]
fn tool_verb_classifies_an_unprefixed_action_slug() {
    // A name with no vendor prefix carries its verb in the first segment;
    // unconditionally stripping the first segment as an assumed vendor
    // prefix discarded the only verb present.
    assert_eq!(tool_verb("create_a_pull_request"), Some(ToolVerb::Create));
    assert_eq!(tool_verb("CREATE_A_PULL_REQUEST"), Some(ToolVerb::Create));
    assert_eq!(tool_verb("delete_message"), Some(ToolVerb::Delete));
}

#[test]
fn delete_query_excludes_create_tools() {
    let tools = vec![
        tool("GMAIL_SEND_EMAIL", "Sends an email."),
        tool("GMAIL_DELETE_MESSAGE", "Deletes a message by id."),
        tool("GMAIL_DELETE_THREAD", "Deletes a thread."),
        tool("GMAIL_BATCH_DELETE_MESSAGES", "Bulk delete messages."),
    ];
    let idx = rank_tools_by_prompt("delete all promotional emails", &tools, 10);
    for &i in &idx {
        assert!(
            tools[i].name.contains("DELETE"),
            "non-DELETE tool leaked: {}",
            tools[i].name
        );
    }
    assert!(idx.len() >= 3);
}
