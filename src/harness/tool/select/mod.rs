//! Prompt-driven tool selection: rank a large tool catalogue against a task
//! prompt and keep only the handful of tools that plausibly matter.
//!
//! A sub-agent bound to a large third-party catalogue is the motivating case.
//! GitHub's action catalogue alone runs to ~500 entries; advertising every one
//! of them balloons the prompt, dilutes the model's attention, and costs real
//! money on every turn. The parent-refined task prompt is usually specific
//! enough that a handful of actions are relevant, so a cheap local ranker can
//! do the narrowing before the model ever sees the list.
//!
//! The ranker is a five-stage pipeline — no model load, pure CPU, stdlib only:
//!
//! 1. **Verb detection** — map the prompt to CRUD-ish intents
//!    (`create`/`send`/`read`/`list`/`update`/`delete`/`merge`).
//! 2. **Verb gate** — drop tools whose first-word verb conflicts with the
//!    detected intent. Tools with a neutral prefix (e.g. `GITHUB_FIND_*`) are
//!    kept as ambiguous.
//! 3. **Query token expansion** — strip stopwords and expand common
//!    abbreviations (`pr` → `pull request`, `dm` → `direct message`) so the
//!    ranker can match casual phrasing against formal tool names.
//! 4. **Weighted token overlap** — 3× weight on hits in the tool name, 1× on
//!    hits in the description. Cheap, effective, explainable.
//! 5. **Verb-alignment boost** — a small additive bonus when the tool's
//!    first-word verb matches the detected intent, a penalty when it clearly
//!    conflicts.
//!
//! The scoring is deliberately explainable rather than clever: it is a
//! relevance ranker whose output decides what a model is allowed to see, so a
//! reviewer must be able to read why a tool made the cut.
//!
//! Entry point: [`rank_tools_by_prompt`]. Callers should treat a thin result
//! as no result — see [`MIN_CONFIDENT_HITS`].
//!
//! ```
//! use tinyagents::harness::tool::{SelectableTool, rank_tools_by_prompt};
//!
//! let tools = [
//!     SelectableTool::new("GITHUB_CREATE_A_PULL_REQUEST", "Creates a pull request."),
//!     SelectableTool::new("GITHUB_DELETE_A_REPOSITORY", "Deletes a repository."),
//! ];
//! let hits = rank_tools_by_prompt("create a PR from my branch", &tools, 5);
//! assert_eq!(hits, vec![0]);
//! ```

mod types;

use std::collections::HashSet;

pub use types::{SelectableTool, ToolVerb};

/// Minimum number of hits the ranker must produce to be trusted. Below this,
/// the caller should fall back to the unfiltered catalogue — a too-narrow
/// selection is worse than no selection at all, because it starves the agent
/// of the one tool it needed.
pub const MIN_CONFIDENT_HITS: usize = 3;

/// Rank `tools` against `prompt` and return indices into `tools` for the top
/// `max_results` matches, ordered best-first.
///
/// Returns an empty `Vec` when `prompt` is blank, `tools` is empty, or no
/// token hits are found. Callers should check the result against
/// [`MIN_CONFIDENT_HITS`] and fall back to the unfiltered catalogue when it
/// comes up short.
pub fn rank_tools_by_prompt(
    prompt: &str,
    tools: &[SelectableTool<'_>],
    max_results: usize,
) -> Vec<usize> {
    if prompt.trim().is_empty() || tools.is_empty() {
        return Vec::new();
    }

    let verbs = detect_verbs(prompt);
    let qt = query_tokens(prompt);

    // Stage 1-2: verb gate. Keep tools whose verb matches the query, or whose
    // prefix is neutral (no recognised verb).
    let gated: Vec<usize> = tools
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            if verbs.is_empty() {
                return true;
            }
            match tool_verb(t.name) {
                Some(v) => verbs.contains(&v),
                None => true,
            }
        })
        .map(|(i, _)| i)
        .collect();

    // Stage 3-5: weighted token overlap + verb-alignment bonus, then sort.
    let mut scored: Vec<(i32, usize)> = gated
        .iter()
        .map(|&i| {
            let t = &tools[i];
            let score = weighted_overlap(&qt, t.name, t.description) + verb_bonus(t.name, &verbs);
            (score, i)
        })
        .collect();

    scored.sort_by_key(|item| std::cmp::Reverse(item.0));

    // Only keep positively-scored results. Zero-overlap tools would add noise.
    scored
        .into_iter()
        .filter(|(s, _)| *s > 0)
        .take(max_results)
        .map(|(_, i)| i)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────
// Verb detection
// ─────────────────────────────────────────────────────────────────────────

fn verb_aliases(v: ToolVerb) -> &'static [&'static str] {
    match v {
        ToolVerb::Create => &[
            "create", "make", "new", "add", "start", "write", "post", "draft",
        ],
        ToolVerb::Send => &[
            "send", "email", "message", "dm", "reply", "forward", "notify",
        ],
        ToolVerb::Read => &["read", "get", "fetch", "show", "view", "see", "retrieve"],
        ToolVerb::List => &["list", "search", "find", "lookup", "browse"],
        ToolVerb::Update => &[
            "update", "edit", "modify", "change", "rename", "move", "set",
        ],
        ToolVerb::Delete => &["delete", "remove", "drop", "archive", "unsubscribe"],
        ToolVerb::Merge => &["merge", "accept", "approve"],
    }
}

const ALL_VERBS: [ToolVerb; 7] = [
    ToolVerb::Create,
    ToolVerb::Send,
    ToolVerb::Read,
    ToolVerb::List,
    ToolVerb::Update,
    ToolVerb::Delete,
    ToolVerb::Merge,
];

/// Tool-name prefixes (uppercase, after the vendor prefix is stripped) that map
/// to each verb. Checked against the first two words of the stripped tool name;
/// a trailing `S` is tolerated (`DELETES` → `DELETE`).
fn tool_verb_prefixes(v: ToolVerb) -> &'static [&'static str] {
    match v {
        ToolVerb::Create => &["CREATE", "ADD", "NEW", "POST", "DRAFT", "START", "INSERT"],
        ToolVerb::Send => &["SEND", "REPLY", "FORWARD", "NOTIFY"],
        ToolVerb::Read => &[
            "GET", "FETCH", "SHOW", "READ", "RETRIEVE", "DESCRIBE", "CHECK",
        ],
        ToolVerb::List => &["LIST", "SEARCH", "FIND", "BROWSE", "COUNT", "QUERY"],
        ToolVerb::Update => &[
            "UPDATE", "EDIT", "MODIFY", "RENAME", "MOVE", "SET", "PATCH", "UPSERT",
        ],
        ToolVerb::Delete => &["DELETE", "REMOVE", "DROP", "ARCHIVE", "UNSUBSCRIBE"],
        ToolVerb::Merge => &["MERGE", "APPROVE", "ACCEPT", "DISMISS"],
    }
}

fn detect_verbs(prompt: &str) -> HashSet<ToolVerb> {
    let lowered = prompt.to_ascii_lowercase();
    let mut found = HashSet::new();
    for &v in &ALL_VERBS {
        for alias in verb_aliases(v) {
            if contains_whole_word(&lowered, alias) {
                found.insert(v);
                break;
            }
        }
    }
    found
}

/// Classify a tool name (e.g. `"GITHUB_CREATE_A_PULL_REQUEST"`) by verb.
/// Returns `None` when no verb prefix is recognised — such tools are kept as
/// neutral by the gate.
fn tool_verb(name: &str) -> Option<ToolVerb> {
    // Strip the vendor prefix (everything up to and including the first `_`).
    let stripped = match name.split_once('_') {
        Some((_, rest)) => rest,
        None => name,
    };
    // Check the first two words.
    for word in stripped.split('_').take(2) {
        let trimmed = word.strip_suffix('S').unwrap_or(word);
        for &v in &ALL_VERBS {
            for &prefix in tool_verb_prefixes(v) {
                if word == prefix || trimmed == prefix {
                    return Some(v);
                }
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────
// Token handling
// ─────────────────────────────────────────────────────────────────────────

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "to", "from", "for", "of", "with", "my", "me", "i", "and", "or", "on", "in",
    "at", "is", "are", "by", "this", "that", "it", "about", "all", "any", "some", "new", "old",
];

/// Abbreviation map applied to query tokens. If the query has `pr`, we add
/// `pull` and `request`; the tool name side already spells the words out, so
/// expanding the query alone bridges the two.
const ABBREVS: &[(&str, &[&str])] = &[
    ("pr", &["pull", "request"]),
    ("prs", &["pull", "requests"]),
    ("dm", &["direct", "message"]),
    ("dms", &["direct", "messages"]),
    ("repo", &["repository"]),
    ("repos", &["repositories"]),
    ("org", &["organization"]),
    ("orgs", &["organizations"]),
    ("msg", &["message"]),
    ("ch", &["channel"]),
];

/// Tokenize a string into lowercase alphanumeric words.
fn tokenize(s: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut current = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            current.push(c.to_ascii_lowercase());
        } else if !current.is_empty() {
            out.insert(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.insert(current);
    }
    out
}

fn query_tokens(query: &str) -> HashSet<String> {
    let raw: HashSet<String> = tokenize(query)
        .into_iter()
        .filter(|t| t.len() > 1 && !STOPWORDS.contains(&t.as_str()))
        .collect();
    let mut expanded = raw.clone();
    for t in &raw {
        for (abbr, replacements) in ABBREVS {
            if t == abbr {
                for r in *replacements {
                    expanded.insert((*r).to_string());
                }
            }
        }
    }
    expanded
}

fn weighted_overlap(qt: &HashSet<String>, name: &str, desc: &str) -> i32 {
    let name_tokens = tokenize(name);
    let desc_tokens = tokenize(desc);
    let name_hits = qt.intersection(&name_tokens).count() as i32;
    let desc_hits = qt.intersection(&desc_tokens).count() as i32;
    3 * name_hits + desc_hits
}

fn verb_bonus(name: &str, query_verbs: &HashSet<ToolVerb>) -> i32 {
    if query_verbs.is_empty() {
        return 0;
    }
    match tool_verb(name) {
        Some(v) if query_verbs.contains(&v) => 3,
        Some(_) => -2,
        None => 0,
    }
}

fn contains_whole_word(haystack: &str, needle: &str) -> bool {
    // Cheap whole-word check without regex. Works on ASCII; task prompts from
    // orchestrators are essentially ASCII anyway.
    let mut start = 0;
    while let Some(idx) = haystack[start..].find(needle) {
        let abs = start + idx;
        let before_ok = abs == 0 || !haystack.as_bytes()[abs - 1].is_ascii_alphanumeric();
        let end = abs + needle.len();
        let after_ok = end == haystack.len() || !haystack.as_bytes()[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}

#[cfg(test)]
mod test;
