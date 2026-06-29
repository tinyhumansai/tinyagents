use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
struct TestState {
    count: usize,
}

#[tokio::test]
async fn runs_direct_graph() {
    let graph = StateGraph::new()
        .add_node(Node::new("increment", |mut state: TestState| async move {
            state.count += 1;
            Ok(NodeOutput::continue_with(state))
        }))
        .add_node(Node::new("finish", |state| async move {
            Ok(NodeOutput::end(state))
        }))
        .set_start("increment")
        .add_edge("increment", "finish");

    let run = graph.run(TestState { count: 0 }).await.unwrap();

    assert_eq!(run.state.count, 1);
    assert_eq!(run.visited, vec!["increment", "finish"]);
}

#[tokio::test]
async fn runs_conditional_graph() {
    let graph = StateGraph::new()
        .add_node(Node::new("router", |state: TestState| async move {
            let route = if state.count == 0 { "empty" } else { "ready" };
            Ok(NodeOutput::route(state, route))
        }))
        .add_node(Node::new("empty", |mut state: TestState| async move {
            state.count = 1;
            Ok(NodeOutput::end(state))
        }))
        .add_node(Node::new("ready", |state| async move {
            Ok(NodeOutput::end(state))
        }))
        .set_start("router")
        .add_conditional_edges("router", [("empty", "empty"), ("ready", "ready")]);

    let run = graph.run(TestState { count: 0 }).await.unwrap();

    assert_eq!(run.state.count, 1);
    assert_eq!(run.visited, vec!["router", "empty"]);
}
