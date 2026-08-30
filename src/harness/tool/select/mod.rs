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
        // Deliberately action words only ("send", "reply", ...) — the
        // ambiguous resource nouns ("email", "message", "dm") that used to
        // live here moved to `SEND_NOUN_ALIASES`, checked separately in
        // `detect_verbs` only when no explicit verb is otherwise present.
        // Keeping them here made "read email" or "delete a message" match
        // Send *alongside* the explicit Read/Delete intent, since a noun
        // is not the same signal as an action word.
        ToolVerb::Send => &["send", "reply", "forward", "notify"],
        ToolVerb::Read => &["read", "get", "fetch", "show", "view", "see", "retrieve"],
        ToolVerb::List => &["list", "search", "find", "lookup", "browse"],
        ToolVerb::Update => &[
            "update", "edit", "modify", "change", "rename", "move", "set",
        ],
        ToolVerb::Delete => &["delete", "remove", "drop", "archive", "unsubscribe"],
        ToolVerb::Merge => &["merge", "accept", "approve"],
    }
}

/// Resource nouns associated with `ToolVerb::Send` (as distinct from the
/// actual action words in `verb_aliases`). A resource noun alone is a much
/// weaker signal than an action word: "message support" has no explicit verb
/// and inferring Send from "message" is reasonable, but "delete a message" or
/// "read email" already carry an explicit conflicting verb (Delete, Read),
/// and a noun must not add Send alongside it — see `detect_verbs`.
const SEND_NOUN_ALIASES: &[&str] = &["email", "message", "dm"];

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

/// Word tokens of `prompt`, in order, lowercase, alphanumeric-only (contiguous
/// runs split on anything else). Unlike `tokenize`'s `HashSet`, order and
/// duplicates matter here — this feeds the proximity check in `detect_verbs`.
fn ordered_words(prompt: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in prompt.chars() {
        if c.is_ascii_alphanumeric() {
            current.push(c.to_ascii_lowercase());
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Verbs whose action words genuinely conflict with a Send-resource-noun
/// reading: "read email" / "delete a message" name Read/Delete, not Send.
const SEND_CONFLICTING_VERBS: &[ToolVerb] = &[
    ToolVerb::Read,
    ToolVerb::List,
    ToolVerb::Update,
    ToolVerb::Delete,
    ToolVerb::Merge,
];

/// Words that join two independent clauses/intents in a short imperative task
/// description ("find Alice AND email her the report"). A conflicting verb
/// separated from a Send-resource-noun by one of these reads as a second,
/// independent instruction rather than that verb's direct object — distance
/// alone cannot tell the two apart: "delete THAT OLD email" and "find Alice
/// AND email her" both have exactly two words between the verb and the noun,
/// and only the second is a compound task.
const CLAUSE_BOUNDARY_WORDS: &[&str] = &["and", "then", "also", "plus", "next"];

/// Whether a conflicting verb at `verb_idx` and a Send-resource-noun at
/// `noun_idx` are in the same clause — i.e. no [`CLAUSE_BOUNDARY_WORDS`] word
/// appears strictly between them. Order-agnostic (`verb_idx` may be before or
/// after `noun_idx`).
fn same_clause(words: &[String], verb_idx: usize, noun_idx: usize) -> bool {
    let (lo, hi) = if verb_idx < noun_idx {
        (verb_idx, noun_idx)
    } else {
        (noun_idx, verb_idx)
    };
    !words[lo + 1..hi]
        .iter()
        .any(|w| CLAUSE_BOUNDARY_WORDS.contains(&w.as_str()))
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
    if found.contains(&ToolVerb::Send) {
        return found;
    }

    // Resource nouns for Send are checked whenever `Send` was not already
    // matched by one of its action aliases above. This is close to (but not
    // identical to) the pre-extraction behaviour, where these nouns lived in
    // `Send`'s own alias list unconditionally: `Send` was added iff ANY of
    // its aliases matched, action word or noun, with no gate at all.
    //
    // A blanket "any conflicting verb found anywhere in the prompt" gate is
    // NOT equivalent and is a real ranking regression: "Post a message to
    // #general" matches "post" (a `Create` alias, not conflicting, so this
    // case is unaffected) but "find Alice and email her the report" matches
    // "find" (a `List` alias, which DOES conflict) and would suppress Send
    // for a genuinely compound task that needs both List and Send tools —
    // dropping GMAIL_SEND_EMAIL from the gated set entirely rather than
    // merely deprioritizing it, since the verb gate filters out (not just
    // down-ranks) a tool whose verb isn't in the query's detected verb set.
    //
    // Same-clause is the fix: a conflicting verb only suppresses the noun
    // when no clause boundary ("and", "then", ...) separates them — i.e.
    // when the noun reads as the verb's direct object ("read email",
    // "delete that old email") rather than a second, independent instruction
    // ("find Alice AND email her the report"). Distance alone cannot
    // distinguish these: both examples have exactly two words between the
    // verb and the noun. `Create` is deliberately never in
    // `SEND_CONFLICTING_VERBS` at any distance: "post/write/draft a message"
    // is a send intent expressed with a creation verb, which is what dropped
    // SLACK_SEND_MESSAGE out of the top 15 when this was (wrongly) gated on
    // `found.is_empty()` instead. Both are pinned by the host's
    // pre-extraction ranking snapshot and real-data suite.
    let words = ordered_words(&lowered);
    let conflicting_aliases: Vec<&'static str> = SEND_CONFLICTING_VERBS
        .iter()
        .flat_map(|&v| verb_aliases(v).iter().copied())
        .collect();

    'nouns: for alias in SEND_NOUN_ALIASES {
        for (noun_idx, word) in words.iter().enumerate() {
            if word != alias {
                continue;
            }
            let conflict_in_clause = words.iter().enumerate().any(|(verb_idx, w)| {
                conflicting_aliases.contains(&w.as_str()) && same_clause(&words, verb_idx, noun_idx)
            });
            if !conflict_in_clause {
                found.insert(ToolVerb::Send);
                break 'nouns;
            }
        }
    }
    found
}

/// Classifies `word` (already uppercased) as a recognised verb prefix,
/// tolerating a trailing plural `S` (`"CREATES"` -> `"CREATE"`).
fn word_as_verb_prefix(word_upper: &str) -> Option<ToolVerb> {
    let trimmed = word_upper.strip_suffix('S').unwrap_or(word_upper);
    ALL_VERBS.into_iter().find(|&v| {
        tool_verb_prefixes(v)
            .iter()
            .any(|&prefix| word_upper == prefix || trimmed == prefix)
    })
}

/// Classify a tool name (e.g. `"GITHUB_CREATE_A_PULL_REQUEST"` or the
/// canonical lowercase `"github_create_a_pull_request"`) by verb. Returns
/// `None` when no verb prefix is recognised — such tools are kept as neutral
/// by the gate.
///
/// `ToolSchema::name` is canonical `snake_case`, so segments are uppercased
/// before comparison against the (uppercase) verb prefix tables — comparing
/// raw case previously left every canonical lowercase name unclassified,
/// silently disabling the verb gate for them.
///
/// The first segment is only stripped as an assumed vendor prefix when it is
/// NOT itself a recognised verb. An unprefixed action slug (e.g.
/// `"create_a_pull_request"`, no vendor segment) has its verb in that first
/// position; unconditionally stripping it discarded the only verb present.
fn tool_verb(name: &str) -> Option<ToolVerb> {
    let first_segment = name.split('_').next().unwrap_or(name);
    let first_is_verb = word_as_verb_prefix(&first_segment.to_ascii_uppercase()).is_some();

    // Strip an assumed vendor prefix (everything up to and including the
    // first `_`) unless the first segment is itself the verb.
    let stripped = if first_is_verb {
        name
    } else {
        match name.split_once('_') {
            Some((_, rest)) => rest,
            None => name,
        }
    };

    // Check the first two words.
    stripped
        .split('_')
        .take(2)
        .find_map(|word| word_as_verb_prefix(&word.to_ascii_uppercase()))
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
