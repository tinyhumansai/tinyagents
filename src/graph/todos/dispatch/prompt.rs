//! Turning a task card into the prompt an autonomous run works from.
//!
//! Two pieces, both pure:
//!
//! - [`build_task_prompt`] — the goal prompt: the card's objective, plan, and
//!   acceptance criteria, plus provenance for a card ingested from an external
//!   source.
//! - [`build_progress_instruction`] — the addendum that tells the run how to
//!   keep its own card current while it works, and how to *stop* by blocking
//!   rather than guessing.
//!
//! The instruction text names two tools by convention — a memory-recall tool
//! and the board's own card-update tool. A host that registers them under
//! different names should pass its own names via [`TaskPromptTools`].

use crate::graph::todos::types::TaskBoardCard;

/// Tool names the generated prompts point the model at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPromptTools {
    /// Tool that pulls related context out of memory. Omit to drop the
    /// "recall related context" sentence entirely.
    pub memory_recall: Option<String>,
    /// Tool that edits a card on a named board by id.
    pub update_task: String,
}

impl Default for TaskPromptTools {
    fn default() -> Self {
        Self {
            memory_recall: Some("memory_recall".to_string()),
            update_task: "update_task".to_string(),
        }
    }
}

/// Render `card` into the goal prompt handed to an autonomous run.
///
/// Leads with the card's `objective` (falling back to its title), then the
/// `plan` steps and `acceptance_criteria` that define done. When the card
/// carries `source_metadata` naming a provider, the prompt also points the run
/// at the originating item — so it can pull the item's prior discussion out of
/// memory before it starts, and record the outcome back on the source when it
/// finishes.
pub fn build_task_prompt(card: &TaskBoardCard, tools: &TaskPromptTools) -> String {
    let mut lines: Vec<String> = Vec::new();

    let objective = card
        .objective
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| card.title.trim());
    lines.push(format!(
        "You are autonomously executing one task to completion. Objective:\n{objective}"
    ));

    if !card.plan.is_empty() {
        lines.push("\nPlan:".to_string());
        for (index, step) in card.plan.iter().enumerate() {
            lines.push(format!("{}. {}", index + 1, step.trim()));
        }
    }

    if !card.acceptance_criteria.is_empty() {
        lines.push("\nAcceptance criteria (the task is done only when all hold):".to_string());
        for criterion in &card.acceptance_criteria {
            lines.push(format!("- {}", criterion.trim()));
        }
    }

    if let Some(meta) = &card.source_metadata {
        let provider = meta.get("provider").and_then(|v| v.as_str());
        let external_id = meta.get("external_id").and_then(|v| v.as_str());
        let url = meta.get("url").and_then(|v| v.as_str());
        let origin = source_origin(provider, meta.get("repo").and_then(|v| v.as_str()), external_id);

        // Gated on a known provider so the origin string is always meaningful:
        // an id-only card would otherwise render a bare "#123".
        if provider.is_some() {
            if let Some(recall) = &tools.memory_recall {
                lines.push(format!(
                    "\nThis task originates from {origin}. Its activity has been ingested into \
                     memory — use your {recall} tool to pull related context (prior discussion, \
                     linked items) before and while you work."
                ));
            } else {
                lines.push(format!("\nThis task originates from {origin}."));
            }
        }
        if let Some(url) = url {
            lines.push(format!("Source link: {url}"));
        }
        // When the upstream item is addressable, close the loop on it: the run
        // reports back through whatever integration tools it already holds,
        // under their existing write scope.
        if provider.is_some() && external_id.is_some() {
            lines.push(format!(
                "\nWhen the task is complete, record the outcome on the upstream source \
                 ({origin}): use your integration tools to add a comment summarising the \
                 resolution and, if the work fully addresses it, close/resolve the item. If you \
                 lack the permission or connection to do so, say so in your final summary instead \
                 of guessing."
            ));
        }
    }

    lines.push(
        "\nWork the task to completion. Do not pick up unrelated work. When finished, your final \
         message should summarise what you did and the evidence (commits, PRs, results)."
            .to_string(),
    );

    lines.join("\n")
}

/// A one-line `provider repo#id` origin, with the blank parts left out.
fn source_origin(provider: Option<&str>, repo: Option<&str>, external_id: Option<&str>) -> String {
    let mut origin = String::new();
    if let Some(provider) = provider {
        origin.push_str(provider);
    }
    if let Some(repo) = repo {
        origin.push(' ');
        origin.push_str(repo);
    }
    if let Some(external_id) = external_id {
        origin.push('#');
        origin.push_str(external_id);
    }
    origin.trim().to_string()
}

/// The addendum appended to a run's prompt so it keeps its own card current.
///
/// The card is already `InProgress` — the dispatcher claimed it before spawning
/// the run — and is addressed by exact id and board, because a card-update tool
/// that defaults to some other board would silently miss it. Two behaviours are
/// asked for: append progress as it happens, and **block instead of guessing**
/// when the run needs a decision it cannot make. A run that blocks leaves the
/// card paused for a human rather than force-completed.
pub fn build_progress_instruction(card_id: &str, thread_id: &str, tools: &TaskPromptTools) -> String {
    let update = &tools.update_task;
    format!(
        "\n\nThis task is tracked as card `{card_id}` on the `{thread_id}` board. As you work, \
         call the `{update}` tool (id `{card_id}`, threadId `{thread_id}`) to keep the card \
         current — append `notes`/`evidence` as you make progress.\n\nIf you need a decision or \
         information from the user, or you genuinely cannot proceed (missing access, ambiguous \
         requirement, an action that needs the user's confirmation), call `{update}` with \
         `status: blocked` and a `blocker` that states exactly what you need from the user. The \
         task will stay paused in that blocked state until the user responds — do NOT guess, \
         fabricate, or take a risky irreversible action just to avoid blocking. If instead you \
         finish the work, end with a summary of what you did and the evidence; completion is \
         recorded automatically."
    )
}
