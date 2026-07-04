//! Tests for the expressive language pipeline: lexer, parser, compiler,
//! capability binding, and graph materialisation.

use std::sync::Arc;

use crate::graph::{Command, NodeContext, NodeFuture, NodeResult};
use crate::language::capability_resolver::{CapabilityResolver, bind_capabilities};
use crate::language::compiler::{BoxedNode, NodeFactory, build_graph, compile};
use crate::language::lexer::tokenize;
use crate::language::parser::{parse, parse_str};
use crate::language::types::{Literal, NodeSpec, Routing, Token};

/// The `support_agent` fixture from the module spec: an agent node with a tool
/// loop plus conditional routing to `END`.
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

mod capability_binding;
mod compiler;
mod diagnostics;
mod extended_grammar;
mod graph_materialisation;
mod lexer;
mod parser;
mod provenance_diff_testkit;
mod registry_binding;
mod resolver;

use graph_materialisation::TestState;
use registry_binding::{FULL_SOURCE, full_registry};
