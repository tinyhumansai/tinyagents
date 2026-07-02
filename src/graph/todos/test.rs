//! Unit tests for the task-board domain types.

use super::types::*;

fn card(id: &str, title: &str, status: TaskCardStatus) -> TaskBoardCard {
    TaskBoardCard {
        id: id.to_string(),
        title: title.to_string(),
        status,
        ..TaskBoardCard::new(title)
    }
}

#[test]
fn status_strings_match_serialized() {
    assert_eq!(TaskCardStatus::Todo.as_str(), "todo");
    assert_eq!(
        TaskCardStatus::AwaitingApproval.as_str(),
        "awaiting_approval"
    );
    assert_eq!(TaskCardStatus::Ready.as_str(), "ready");
    assert_eq!(TaskCardStatus::InProgress.as_str(), "in_progress");
    assert_eq!(TaskCardStatus::Blocked.as_str(), "blocked");
    assert_eq!(TaskCardStatus::Done.as_str(), "done");
    assert_eq!(TaskCardStatus::Rejected.as_str(), "rejected");
    assert_eq!(TaskApprovalMode::Required.as_str(), "required");
    assert_eq!(TaskApprovalMode::NotRequired.as_str(), "not_required");
}

#[test]
fn parse_status_accepts_aliases() {
    assert_eq!(parse_status("todo").unwrap(), TaskCardStatus::Todo);
    assert_eq!(parse_status("PENDING").unwrap(), TaskCardStatus::Todo);
    assert_eq!(
        parse_status("in-progress").unwrap(),
        TaskCardStatus::InProgress
    );
    assert_eq!(parse_status("approved").unwrap(), TaskCardStatus::Ready);
    assert_eq!(parse_status("done").unwrap(), TaskCardStatus::Done);
    assert_eq!(parse_status("denied").unwrap(), TaskCardStatus::Rejected);
    assert!(parse_status("nope").is_err());
}

#[test]
fn card_and_board_round_trip_through_json() {
    let mut c = card("task-1", "Draft plan", TaskCardStatus::AwaitingApproval);
    c.approval_mode = Some(TaskApprovalMode::Required);
    c.plan = vec!["step one".into()];
    let board = TaskBoard {
        thread_id: "t".into(),
        cards: vec![c.clone()],
        updated_at: "0".into(),
    };
    let json = serde_json::to_value(&board).unwrap();
    assert_eq!(json["threadId"], "t");
    assert_eq!(json["cards"][0]["approvalMode"], "required");
    let back: TaskBoard = serde_json::from_value(json).unwrap();
    assert_eq!(back, board);
}

#[test]
fn render_markdown_uses_status_markers_and_sub_lines() {
    let mut done = card("task-1", "Ship it", TaskCardStatus::Done);
    done.objective = Some("release the crate".into());
    let mut blocked = card("task-2", "Wait on CI", TaskCardStatus::Blocked);
    blocked.blocker = Some("CI is red".into());
    let in_progress = card("task-3", "Write docs", TaskCardStatus::InProgress);
    let awaiting = card("task-4", "Approve plan", TaskCardStatus::AwaitingApproval);
    let rejected = card("task-5", "Nope", TaskCardStatus::Rejected);
    let todo = card("task-6", "Later", TaskCardStatus::Todo);

    let md = render_markdown(&[done, blocked, in_progress, awaiting, rejected, todo]);
    assert!(md.contains("- [x] Ship it  `(task-1)`"));
    assert!(md.contains("  - objective: release the crate"));
    assert!(md.contains("- [!] Wait on CI"));
    assert!(md.contains("  - _blocked:_ CI is red"));
    assert!(md.contains("- [~] Write docs"));
    assert!(md.contains("- [?] Approve plan"));
    assert!(md.contains("- [-] Nope"));
    assert!(md.contains("- [ ] Later"));
}

#[test]
fn render_markdown_empty_is_placeholder() {
    assert_eq!(render_markdown(&[]), "_No todos yet._");
}

#[test]
fn normalise_trims_generates_ids_and_recomputes_order() {
    let mut board = TaskBoard {
        thread_id: "  t  ".into(),
        cards: vec![
            TaskBoardCard {
                order: 99,
                objective: Some("  ship briefs  ".into()),
                plan: vec!["  extend schema  ".into(), "   ".into()],
                allowed_tools: vec![" todo ".into(), "".into()],
                ..card("", "  Draft plan  ", TaskCardStatus::Todo)
            },
            // Empty title → dropped.
            card("empty", "   ", TaskCardStatus::Todo),
            // Blocked without a blocker → backfilled from notes.
            TaskBoardCard {
                notes: Some("waiting on user".into()),
                ..card("blocked", "Need approval", TaskCardStatus::Blocked)
            },
        ],
        updated_at: String::new(),
    };
    normalise_board(&mut board);

    assert_eq!(board.thread_id, "t");
    assert_eq!(board.cards.len(), 2, "empty-title card dropped");
    assert_eq!(board.cards[0].title, "Draft plan");
    assert_eq!(board.cards[0].objective.as_deref(), Some("ship briefs"));
    assert_eq!(board.cards[0].plan, vec!["extend schema"]);
    assert_eq!(board.cards[0].allowed_tools, vec!["todo"]);
    assert!(board.cards[0].id.starts_with("task-"), "blank id generated");
    assert_eq!(board.cards[0].order, 0);
    assert_eq!(board.cards[1].order, 1);
    assert_eq!(board.cards[1].blocker.as_deref(), Some("waiting on user"));
}

mod store_tests {
    use std::sync::Arc;

    use super::super::store;
    use super::super::types::{CardPatch, TaskBoardCard, TaskCardStatus};
    use crate::harness::store::{InMemoryStore, Store};

    fn store() -> Arc<dyn Store> {
        Arc::new(InMemoryStore::default())
    }

    #[tokio::test]
    async fn add_list_remove_round_trip() {
        let s = store();
        assert!(store::list(&s, "t").await.unwrap().cards.is_empty());

        let snap = store::add(&s, "t", "Write the RFC", CardPatch::default())
            .await
            .unwrap();
        assert_eq!(snap.thread_id, "t");
        assert_eq!(snap.cards.len(), 1);
        assert_eq!(snap.cards[0].title, "Write the RFC");
        assert!(snap.markdown.contains("Write the RFC"));
        let id = snap.cards[0].id.clone();

        let listed = store::list(&s, "t").await.unwrap();
        assert_eq!(listed.cards.len(), 1);

        let after = store::remove(&s, "t", &id).await.unwrap();
        assert!(after.cards.is_empty());
        assert!(
            store::remove(&s, "t", &id).await.is_err(),
            "unknown id errors"
        );
    }

    #[tokio::test]
    async fn add_rejects_empty_content_and_blank_thread() {
        let s = store();
        assert!(
            store::add(&s, "t", "   ", CardPatch::default())
                .await
                .is_err()
        );
        assert!(
            store::add(&s, "  ", "x", CardPatch::default())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn single_in_progress_invariant_is_enforced() {
        let s = store();
        let a = store::add(&s, "t", "A", CardPatch::default())
            .await
            .unwrap();
        let a_id = a.cards[0].id.clone();
        let b = store::add(&s, "t", "B", CardPatch::default())
            .await
            .unwrap();
        let b_id = b.cards[1].id.clone();

        store::update_status(&s, "t", &a_id, TaskCardStatus::InProgress)
            .await
            .unwrap();
        // A second in-progress card is rejected, not silently fixed.
        let err = store::update_status(&s, "t", &b_id, TaskCardStatus::InProgress)
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("in_progress"));
        // The board still has exactly one in-progress card.
        let listed = store::list(&s, "t").await.unwrap();
        let in_progress = listed
            .cards
            .iter()
            .filter(|c| c.status == TaskCardStatus::InProgress)
            .count();
        assert_eq!(in_progress, 1);
    }

    #[tokio::test]
    async fn replace_enforces_invariant() {
        let s = store();
        let two_in_progress = vec![
            TaskBoardCard {
                status: TaskCardStatus::InProgress,
                ..TaskBoardCard::new("A")
            },
            TaskBoardCard {
                status: TaskCardStatus::InProgress,
                ..TaskBoardCard::new("B")
            },
        ];
        assert!(store::replace(&s, "t", two_in_progress).await.is_err());
    }

    #[tokio::test]
    async fn decide_plan_only_from_awaiting_approval() {
        let s = store();
        let snap = store::add(
            &s,
            "t",
            "Gated work",
            CardPatch {
                status: Some(TaskCardStatus::AwaitingApproval),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let id = snap.cards[0].id.clone();

        let approved = store::decide_plan(&s, "t", &id, true).await.unwrap();
        assert_eq!(approved.cards[0].status, TaskCardStatus::Ready);
        // A second decision on a now-Ready card errors (can't resurrect).
        assert!(store::decide_plan(&s, "t", &id, false).await.is_err());
    }

    #[tokio::test]
    async fn revise_plan_rejects_all_awaiting_and_is_lenient_when_empty() {
        let s = store();
        store::add(
            &s,
            "t",
            "Gated",
            CardPatch {
                status: Some(TaskCardStatus::AwaitingApproval),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let after = store::revise_plan(&s, "t").await.unwrap();
        assert_eq!(after.cards[0].status, TaskCardStatus::Rejected);
        // Nothing awaiting now → benign no-op.
        let again = store::revise_plan(&s, "t").await.unwrap();
        assert_eq!(again.cards[0].status, TaskCardStatus::Rejected);
    }

    #[tokio::test]
    async fn claim_card_cas_accepts_then_rejects() {
        let s = store();
        let snap = store::add(
            &s,
            "t",
            "Runnable",
            CardPatch {
                status: Some(TaskCardStatus::Ready),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let id = snap.cards[0].id.clone();

        let claimed = store::claim_card(
            &s,
            "t",
            &id,
            &[TaskCardStatus::Ready],
            TaskCardStatus::InProgress,
        )
        .await
        .unwrap();
        assert_eq!(claimed.status, TaskCardStatus::InProgress);
        // A second claim expecting Ready now fails (already in progress).
        assert!(
            store::claim_card(
                &s,
                "t",
                &id,
                &[TaskCardStatus::Ready],
                TaskCardStatus::InProgress
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn edit_leaves_unset_fields_untouched() {
        let s = store();
        let snap = store::add(
            &s,
            "t",
            "Task",
            CardPatch {
                objective: Some("keep me".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let id = snap.cards[0].id.clone();
        // Edit only the notes; objective is preserved.
        let edited = store::edit(
            &s,
            "t",
            &id,
            CardPatch {
                notes: Some("a note".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(edited.cards[0].objective.as_deref(), Some("keep me"));
        assert_eq!(edited.cards[0].notes.as_deref(), Some("a note"));
    }

    #[tokio::test]
    async fn set_session_thread_links_then_clears() {
        let s = store();
        let snap = store::add(&s, "t", "Task", CardPatch::default())
            .await
            .unwrap();
        let id = snap.cards[0].id.clone();
        let linked = store::set_session_thread(&s, "t", &id, Some("thread-xyz".into()))
            .await
            .unwrap();
        assert_eq!(
            linked.cards[0].session_thread_id.as_deref(),
            Some("thread-xyz")
        );
        let cleared = store::set_session_thread(&s, "t", &id, Some("  ".into()))
            .await
            .unwrap();
        assert_eq!(cleared.cards[0].session_thread_id, None);
    }
}
