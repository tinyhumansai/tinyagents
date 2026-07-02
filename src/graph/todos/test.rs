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
