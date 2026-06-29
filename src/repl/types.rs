//! REPL command and session types for the `.ragsh` interactive language.
//!
//! All public types for the REPL skeleton live here.  Logic (parsing) lives in
//! [`super`]; tests live in `test.rs`.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

// ── Command model ─────────────────────────────────────────────────────────────

/// The set of commands understood by the `.ragsh` REPL.
///
/// Each variant maps to one command verb.  Serde is derived so that command
/// values can be logged or replayed as JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum ReplCommand {
    /// Print command help listing all verbs and their signatures.
    Help,

    /// Load a `.rag` blueprint from the given file path into the session.
    Load {
        /// Filesystem path to the `.rag` source file.
        path: String,
    },

    /// Compile a named blueprint that has already been loaded into the session.
    Compile {
        /// Name of the blueprint to compile.
        name: String,
    },

    /// Run a named compiled graph with a JSON-encoded input payload.
    ///
    /// In the skeleton this produces a [`ReplOutcome::Planned`]; live wiring
    /// to the graph runtime is a follow-up milestone.
    Run {
        /// Name of the registered compiled graph.
        graph: String,
        /// JSON-encoded input payload to pass to the graph.
        input: String,
    },

    /// Set a named session variable to a string value.
    ///
    /// The value is stored internally as a [`serde_json::Value::String`].
    /// Use [`ReplSession::set`] directly for richer JSON values.
    Set {
        /// Variable name.
        key: String,
        /// String representation of the value.
        value: String,
    },

    /// Retrieve a named session variable and return its value.
    Get {
        /// Variable name to look up.
        key: String,
    },

    /// Show session information.
    ///
    /// Recognised subjects: `vars`, `graphs`, `status`.
    Show {
        /// The subject to display (`vars`, `graphs`, or `status`).
        what: String,
    },

    /// Invoke a registered capability by name with a JSON argument object.
    ///
    /// In the skeleton this is policy-checked and returned as
    /// [`ReplOutcome::Planned`] rather than executed immediately.
    Call {
        /// Name of the registered capability (must be on the [`CapabilityPolicy`]
        /// allowlist).
        capability: String,
        /// Arbitrary JSON arguments forwarded to the capability.
        args: serde_json::Value,
    },

    /// Exit the REPL session.
    Quit,
}

impl ReplCommand {
    /// Returns the canonical command verb name used in the grammar.
    pub fn name(&self) -> &'static str {
        match self {
            ReplCommand::Help => "help",
            ReplCommand::Load { .. } => "load",
            ReplCommand::Compile { .. } => "compile",
            ReplCommand::Run { .. } => "run",
            ReplCommand::Set { .. } => "set",
            ReplCommand::Get { .. } => "get",
            ReplCommand::Show { .. } => "show",
            ReplCommand::Call { .. } => "call",
            ReplCommand::Quit => "quit",
        }
    }
}

// ── Outcome ───────────────────────────────────────────────────────────────────

/// The result produced by executing a [`ReplCommand`] in a [`ReplSession`].
///
/// Uses adjacent tagging (`tag = "kind", content = "data"`) so that newtype
/// variants containing non-map values (such as `Message` holding a `String`)
/// serialize correctly alongside struct variants like `Planned`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum ReplOutcome {
    /// A human-readable message from a side-effect-free command.
    Message(String),

    /// A JSON value retrieved from the session namespace.
    Value(serde_json::Value),

    /// The command was policy-checked and recorded; live harness/graph
    /// execution is deferred until the REPL skeleton is wired to a runtime
    /// (milestones R2–R6 in the design document).
    Planned {
        /// Short label of the intended action (e.g. `"graph_run"`).
        action: String,
        /// Structured parameters describing the planned call.
        detail: serde_json::Value,
    },

    /// The session has been asked to terminate.
    Quit,
}

// ── Capability policy ─────────────────────────────────────────────────────────

/// An allowlist that controls which capability names a [`ReplSession`] may
/// invoke.
///
/// By default nothing is allowed.  Use [`CapabilityPolicy::allow`] or
/// [`CapabilityPolicy::from_list`] to grant access.  Attempting to invoke a
/// capability that is not on the list produces a
/// [`crate::error::RustAgentsError::Capability`] error.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityPolicy {
    allowed: HashSet<String>,
}

impl CapabilityPolicy {
    /// Create an empty policy (no capabilities allowed).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a capability name to the allowlist.
    pub fn allow(&mut self, name: impl Into<String>) -> &mut Self {
        self.allowed.insert(name.into());
        self
    }

    /// Returns `true` if the given capability name is on the allowlist.
    pub fn is_allowed(&self, name: &str) -> bool {
        self.allowed.contains(name)
    }

    /// Build a policy from an iterable of allowed names.
    pub fn from_list<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut policy = Self::new();
        for name in names {
            policy.allow(name.into());
        }
        policy
    }

    /// Returns the number of capabilities currently on the allowlist.
    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    /// Returns `true` if no capabilities are allowed.
    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

/// An interactive REPL session holding session variables, a capability policy,
/// and a command history.
///
/// `ReplSession` is the primary entry point for driving the `.ragsh` skeleton.
///
/// ## Execution model
///
/// Side-effect-free commands (`Set`, `Get`, `Show`, `Help`, `Quit`) run fully
/// inside `execute`.  Commands that need live harness/graph integration
/// (`Load`, `Compile`, `Run`, `Call`) are policy-checked first — a
/// [`crate::error::RustAgentsError::Capability`] error is returned immediately
/// if the operation is not on the allowlist — and, when allowed, the method
/// returns [`ReplOutcome::Planned`] describing the intended action without
/// performing it.  The wiring to the live runtime is a follow-up milestone
/// (R2–R6 in the design document).
///
/// ## Example
///
/// ```rust
/// use rustagents::repl::{ReplSession, CapabilityPolicy, ReplOutcome};
///
/// let policy = CapabilityPolicy::from_list(["my_tool"]);
/// let mut session = ReplSession::new().with_policy(policy);
///
/// session.set("x", serde_json::json!(42));
/// assert_eq!(session.get("x"), Some(&serde_json::json!(42)));
/// ```
pub struct ReplSession {
    /// Session-scoped variables, keyed by name and stored as JSON values.
    variables: HashMap<String, serde_json::Value>,
    /// The capability allowlist governing this session.
    policy: CapabilityPolicy,
    /// Ordered history of every command submitted to this session.
    pub history: Vec<ReplCommand>,
}

impl ReplSession {
    /// Create a new session with an empty namespace and a deny-all policy.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            policy: CapabilityPolicy::new(),
            history: Vec::new(),
        }
    }

    /// Replace the session's capability policy, returning the updated session.
    pub fn with_policy(mut self, policy: CapabilityPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set a session variable to any JSON value.
    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.variables.insert(key.into(), value);
    }

    /// Get a session variable by name.  Returns `None` if it has not been set.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.variables.get(key)
    }

    /// Return a reference to the full variable map.
    pub fn vars(&self) -> &HashMap<String, serde_json::Value> {
        &self.variables
    }

    /// Execute a command against this session and return a [`ReplOutcome`].
    ///
    /// The command is always appended to [`ReplSession::history`] before
    /// execution begins.
    ///
    /// # Errors
    ///
    /// * [`crate::error::RustAgentsError::Capability`] — the command requires
    ///   a capability that is not on the allowlist.
    /// * [`crate::error::RustAgentsError::Serialization`] — an internal
    ///   serialization step failed (e.g. serialising variables for `show vars`).
    pub fn execute(&mut self, cmd: ReplCommand) -> crate::error::Result<ReplOutcome> {
        self.history.push(cmd.clone());

        match cmd {
            ReplCommand::Help => {
                let text = concat!(
                    "Commands:\n",
                    "  help                        — show this help\n",
                    "  load <path>                 — load a .rag blueprint\n",
                    "  compile <name>              — compile a loaded blueprint\n",
                    "  run <graph> <input>         — run a compiled graph\n",
                    "  set <key> <value>           — set a session variable\n",
                    "  get <key>                   — retrieve a session variable\n",
                    "  show <vars|graphs|status>   — show session info\n",
                    "  call <capability> <json>    — invoke a registered capability\n",
                    "  quit                        — exit the session",
                );
                Ok(ReplOutcome::Message(text.to_string()))
            }

            ReplCommand::Quit => Ok(ReplOutcome::Quit),

            ReplCommand::Set { key, value } => {
                self.variables.insert(key, serde_json::Value::String(value));
                Ok(ReplOutcome::Message("ok".to_string()))
            }

            ReplCommand::Get { key } => {
                let val = self
                    .variables
                    .get(&key)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                Ok(ReplOutcome::Value(val))
            }

            ReplCommand::Show { what } => match what.as_str() {
                "vars" => {
                    let map = serde_json::to_value(&self.variables)?;
                    Ok(ReplOutcome::Value(map))
                }
                "graphs" => Ok(ReplOutcome::Message(
                    "(graph registry not yet wired in skeleton)".to_string(),
                )),
                "status" => {
                    let status = serde_json::json!({
                        "variables": self.variables.len(),
                        "history": self.history.len(),
                        "policy_allowed": self.policy.len(),
                    });
                    Ok(ReplOutcome::Value(status))
                }
                other => Ok(ReplOutcome::Message(format!(
                    "unknown show subject `{other}`; recognised subjects: vars, graphs, status"
                ))),
            },

            ReplCommand::Load { path } => {
                self.check_capability("load")?;
                Ok(ReplOutcome::Planned {
                    action: "load".to_string(),
                    detail: serde_json::json!({ "path": path }),
                })
            }

            ReplCommand::Compile { name } => {
                self.check_capability("compile")?;
                Ok(ReplOutcome::Planned {
                    action: "compile".to_string(),
                    detail: serde_json::json!({ "name": name }),
                })
            }

            ReplCommand::Run { graph, input } => {
                self.check_capability("run")?;
                Ok(ReplOutcome::Planned {
                    action: "graph_run".to_string(),
                    detail: serde_json::json!({ "graph": graph, "input": input }),
                })
            }

            ReplCommand::Call { capability, args } => {
                self.check_capability(&capability)?;
                Ok(ReplOutcome::Planned {
                    action: "capability_call".to_string(),
                    detail: serde_json::json!({ "capability": capability, "args": args }),
                })
            }
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn check_capability(&self, name: &str) -> crate::error::Result<()> {
        if self.policy.is_allowed(name) {
            Ok(())
        } else {
            Err(crate::error::RustAgentsError::Capability(format!(
                "capability `{name}` is not in the session allowlist"
            )))
        }
    }
}

impl Default for ReplSession {
    fn default() -> Self {
        Self::new()
    }
}
