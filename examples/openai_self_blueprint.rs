//! The agent authors its own graph: OpenAI emits `.rag` source, which we then
//! run through the SAFE compile pipeline.
//!
//! Flow:
//! 1. Ask OpenAI to output **only** `.rag` source for a small agent graph,
//!    giving it the grammar plus a worked example in the system prompt.
//! 2. Extract the `.rag` text from the reply (stripping ``` fences).
//! 3. Run it through the safe pipeline:
//!    `parse_str` -> `compile` -> print the [`Blueprint`] ->
//!    `bind_capabilities` against a [`CapabilityResolver`] allowlist (the policy
//!    gate: only allowlisted models/tools pass) -> `build_graph` with a trivial
//!    [`NodeFactory`] -> run to END.
//!
//! The model never executes code — it only produces declarative source that a
//! Rust-side factory materialises, and the capability allowlist is the safety
//! boundary. If the model's output fails to parse or compile, the diagnostic
//! and the offending source are printed instead of panicking.
//!
//! Run with:
//!
//! ```text
//! cargo run --features openai --example openai_self_blueprint
//! ```

use std::sync::Arc;

use tinyagents::Result;
use tinyagents::graph::{CompiledGraph, END, NodeFuture};
use tinyagents::harness::message::Message;
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::AgentHarness;
use tinyagents::language::compiler::{
    BoxedNode, CapabilityResolver, NodeFactory, bind_capabilities, build_graph, compile,
};
use tinyagents::language::parser::parse_str;
use tinyagents::language::types::{NodeSpec, Routing};
use tinyagents::{Command, NodeContext, NodeResult};

/// Grammar + worked example handed to the model so it emits valid `.rag`.
const SYSTEM_PROMPT: &str = r#"
You author agent graphs in the tinyagents expressive language (`.rag`).
Output ONLY `.rag` source code. No prose, no explanation, no markdown fences.

Grammar (one graph):

  graph <name> {
    start <node>

    node <node> {
      kind agent            // or: tool_executor
      model "<model>"       // agent nodes only
      system "<instruction>"
      tools ["<tool>", ...] // optional
      routes {              // conditional routing (agent nodes)
        tool_call -> <node>
        final -> END
      }
    }

    node <node> {
      kind tool_executor
      next <node>           // unconditional successor
    }
  }

Worked example:

  graph helpdesk {
    start agent

    node agent {
      kind agent
      model "default"
      system "Answer support questions, using tools when useful."
      tools ["search_kb"]
      routes {
        tool_call -> tools
        final -> END
      }
    }

    node tools {
      kind tool_executor
      next agent
    }
  }

Constraints for YOUR output:
- Use model "default" only.
- You may only reference these tools: "search_kb", "create_ticket".
- Keep it to at most three nodes.
"#;

/// Graph state for the materialised blueprint: just an execution trail.
#[derive(Clone, Debug, Default)]
struct BlueprintState {
    trail: Vec<String>,
}

/// A trivial node factory. Behaviour lives entirely in Rust — the `.rag` source
/// only chooses *which* nodes exist and how they are wired. Each node records
/// its name; routing is resolved so the run always terminates: linear (`next`)
/// and terminal nodes commit a whole-state update (static edges route them),
/// and conditional nodes end immediately by routing to `END` (the demo always
/// takes the `final -> END` route).
struct TrailFactory;

impl NodeFactory<BlueprintState> for TrailFactory {
    fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<BlueprintState>> {
        let name = spec.name.clone();
        let routing = spec.routing.clone();
        Ok(Arc::new(
            move |mut state: BlueprintState, _ctx: NodeContext| -> NodeFuture<BlueprintState> {
                let name = name.clone();
                let routing = routing.clone();
                Box::pin(async move {
                    state.trail.push(name);
                    let result = match &routing {
                        // Static edges (Next/Terminal) route these; commit the
                        // whole-state update.
                        Routing::Next(_) | Routing::Terminal => NodeResult::Update(state),
                        // Conditional nodes are command-routed: end the demo run
                        // by routing explicitly to END.
                        Routing::Conditional(_) => {
                            NodeResult::Command(Command::goto([END]).with_update(state))
                        }
                    };
                    Ok(result)
                })
            },
        ))
    }
}

/// Extracts `.rag` source from a model reply, stripping a single ``` fenced
/// block (with optional language tag) when present.
fn extract_rag(reply: &str) -> String {
    let trimmed = reply.trim();
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        // Drop an optional language tag on the rest of the opening line.
        let after = match after.find('\n') {
            Some(nl) => &after[nl + 1..],
            None => after,
        };
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
        return after.trim().to_string();
    }
    trimmed.to_string()
}

/// Runs the safe pipeline over `source`, printing each stage. Returns `Ok(())`
/// even on parse/compile failure (the diagnostic is printed) so a bad model
/// output does not abort the example with a panic.
async fn run_pipeline(source: &str) -> Result<()> {
    println!("\n--- stage 1: parse + compile ---");
    let program = match parse_str(source) {
        Ok(program) => program,
        Err(error) => {
            eprintln!("parse failed: {error}");
            eprintln!("offending source:\n{source}");
            return Ok(());
        }
    };

    let mut blueprints = match compile(&program) {
        Ok(blueprints) => blueprints,
        Err(error) => {
            eprintln!("compile failed: {error}");
            eprintln!("offending source:\n{source}");
            return Ok(());
        }
    };
    if blueprints.is_empty() {
        eprintln!("compile produced no blueprints");
        return Ok(());
    }
    let blueprint = blueprints.remove(0);

    println!("\n--- stage 2: blueprint ---");
    println!("graph : {}", blueprint.graph_id);
    println!("start : {}", blueprint.start);
    for node in &blueprint.nodes {
        println!(
            "node  : {} (kind {}, model {:?}, tools {:?}) routing {:?}",
            node.name, node.kind, node.model, node.tools, node.routing
        );
    }

    println!("\n--- stage 3: capability binding (policy gate) ---");
    // Only these models/tools are allowed. Anything the model invented outside
    // this allowlist is rejected here — the safety boundary.
    let resolver = CapabilityResolver::new()
        .allow_model("default")
        .allow_tool("search_kb")
        .allow_tool("create_ticket");
    match bind_capabilities(&blueprint, &resolver) {
        Ok(()) => println!("binding OK: every referenced model/tool is allowlisted"),
        Err(error) => {
            eprintln!("binding rejected by the allowlist: {error}");
            eprintln!("(this is the safety gate doing its job)");
            return Ok(());
        }
    }

    println!("\n--- stage 4: build graph + run to END ---");
    let graph: CompiledGraph<BlueprintState, BlueprintState> =
        build_graph(&blueprint, &TrailFactory)?;
    let run = graph.run(BlueprintState::default()).await?;
    println!("visited: {:?}", run.visited);
    println!("trail  : {:?}", run.state.trail);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let model = OpenAiModel::from_env()?;
    println!("=== OpenAI self-authored blueprint ===");
    println!("model: {}", model.model());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("openai", Arc::new(model))
        .set_default_model("openai");

    let task = "Author a `.rag` graph for a customer-support agent that can \
                search the knowledge base and create a ticket.";
    println!("task : {task}\n");

    let run = harness
        .invoke_default(
            &(),
            vec![Message::system(SYSTEM_PROMPT), Message::user(task)],
        )
        .await?;

    let reply = run.text().unwrap_or_default();
    println!("--- stage 0: model-authored reply ---");
    println!("{reply}");

    let source = extract_rag(&reply);
    run_pipeline(&source).await?;

    Ok(())
}
