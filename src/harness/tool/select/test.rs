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
    let v = detect_verbs("post a message to general channel");
    assert!(v.contains(&ToolVerb::Send) || v.contains(&ToolVerb::Create));

    let v = detect_verbs("delete all promotional emails");
    assert!(v.contains(&ToolVerb::Delete));

    let v = detect_verbs("merge pull request 42");
    assert!(v.contains(&ToolVerb::Merge));
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

#[test]
fn selectable_tool_converts_from_a_name_description_pair() {
    let t: SelectableTool<'_> = ("GMAIL_SEND_EMAIL", "Sends an email.").into();
    assert_eq!(t.name, "GMAIL_SEND_EMAIL");
    assert_eq!(t.description, "Sends an email.");
}
