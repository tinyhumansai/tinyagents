//! Compiles the spec's `support_agent` `.rag` blueprint and binds its
//! capabilities.
//!
//! Parses the expressive-language source into a `Program`, compiles it into a
//! [`Blueprint`], prints the node/edge/route structure, then binds the
//! blueprint's referenced model and tools against a [`CapabilityResolver`].
//!
//! Run with:
//!
//! ```text
//! cargo run --example rag_blueprint
//! ```

use tinyagents::Result;
use tinyagents::language::capability_resolver::{CapabilityResolver, bind_capabilities};
use tinyagents::language::compiler::compile;
use tinyagents::language::parser::parse_str;
use tinyagents::language::types::Routing;

const SUPPORT_AGENT: &str = r#"
// A support workflow with a tool loop.
graph support_agent {
  start agent

  defaults {
    recursion_limit 50
    backoff "exponential"
    checkpoint inherit
  }

  channel messages messages
  channel tool_calls append

  node agent {
    kind agent
    model "default"
    system "Resolve support requests using tools when useful."
    tools ["lookup_user", "create_ticket"]
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
"#;

fn main() -> Result<()> {
    // source -> lexer -> parser -> AST -> compiler -> Blueprint
    let program = parse_str(SUPPORT_AGENT)?;
    let blueprint = compile(&program)?.remove(0);

    println!("=== Blueprint: {} ===", blueprint.graph_id);
    println!("start node : {}", blueprint.start);
    println!("channels   : {:?}", blueprint.channels);
    println!("edges      : {:?}", blueprint.edges);

    println!("nodes:");
    for node in &blueprint.nodes {
        print!("  - {} (kind {})", node.name, node.kind);
        if let Some(model) = &node.model {
            print!(", model {model:?}");
        }
        if !node.tools.is_empty() {
            print!(", tools {:?}", node.tools);
        }
        println!();
        match &node.routing {
            Routing::Next(target) => println!("      next -> {target}"),
            Routing::Conditional(routes) => {
                for (label, target) in routes {
                    println!("      route {label} -> {target}");
                }
            }
            Routing::Terminal => println!("      (terminal)"),
        }
    }

    // Bind the capabilities the blueprint references against an allowlist.
    let resolver = CapabilityResolver::new()
        .allow_model("default")
        .allow_tool("lookup_user")
        .allow_tool("create_ticket");

    bind_capabilities(&blueprint, &resolver)?;
    println!("\ncapability binding: OK (model + tools resolved against the allowlist)");

    Ok(())
}
