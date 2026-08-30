//! Tests for the expressive language pipeline: lexer, parser, compiler,
//! capability binding, and graph materialisation.

use crate::capability_resolver::{CapabilityResolver, bind_capabilities};
use crate::compiler::compile;
use crate::lexer::tokenize;
use crate::parser::{parse, parse_str};
use crate::types::{Literal, Routing, Token};

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
mod lexer;
mod parser;
mod provenance_diff_testkit;
