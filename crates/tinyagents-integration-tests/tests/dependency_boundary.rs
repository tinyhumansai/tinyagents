//! Dependency-direction guard for host-independent TinyAgents crates.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

// Product types need not carry an `OpenHuman` prefix. Keep this inventory in
// step with the OpenHuman architecture check when product wire types are added.
const OPENHUMAN_DOMAIN_TYPES: &[&str] = &[
    "AgentDefinitionDisplay",
    "AgentDefinitionModel",
    "AgentProfile",
    "AgentProfileStore",
    "AgentProgress",
    "ApprovalDecision",
    "ChatMessage",
    "ConversationMessage",
    "ConversationMessagePatch",
    "ConversationMessageRecord",
    "DomainEvent",
    "IntegrationAccount",
    "OpenHumanConfig",
    "OpenHumanRunContext",
    "ToolExecutionResult",
    "ToolResultMessage",
    "TurnOrigin",
    "UserProfileHook",
    "WorkspacePolicy",
];

// Exact temporary debt for TinyAgents' crate-local Claude Code wire record.
// Physical file-and-line debt is enforced exclusively here; it is deliberately
// line-specific, so a new, deleted, or moved bare `ChatMessage` reference
// fails instead of being hidden by a module or path exclusion.
const KNOWN_GENERIC_CLAUDE_CODE_CHAT_MESSAGE_DEBT: &[(&str, usize)] = &[
    (
        "crates/tinyagents-harness/src/providers/claude_code/bridge.rs",
        10,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/bridge.rs",
        15,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/driver.rs",
        71,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/driver.rs",
        237,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder.rs",
        15,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder.rs",
        25,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder.rs",
        35,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder.rs",
        93,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder.rs",
        116,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        6,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        8,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        9,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        10,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        11,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        148,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        167,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        178,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        179,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        180,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        197,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        198,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        199,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        200,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        210,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/input_builder_tests.rs",
        225,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/mod.rs",
        55,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/mod.rs",
        222,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/mod.rs",
        292,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/mod.rs",
        335,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/mod.rs",
        358,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/mod_tests.rs",
        52,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/mod_tests.rs",
        53,
    ),
    (
        "crates/tinyagents-harness/src/providers/claude_code/mod_tests.rs",
        54,
    ),
];

fn visit_files(root: &Path, extension: &str, found: &mut Vec<PathBuf>) {
    for entry in
        fs::read_dir(root).unwrap_or_else(|error| panic!("read {}: {error}", root.display()))
    {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            visit_files(&path, extension, found);
        } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            found.push(path);
        }
    }
}

fn dependency_manifests(workspace: &Path) -> Vec<PathBuf> {
    let mut manifests = vec![workspace.join("Cargo.toml")];
    visit_files(&workspace.join("crates"), "toml", &mut manifests);
    manifests.retain(|manifest| {
        manifest.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml")
    });
    manifests
}

fn is_rust_char_literal(bytes: &[u8], start: usize) -> bool {
    let Some(&first) = bytes.get(start + 1) else {
        return false;
    };
    if first == b'\\' {
        let Some(&escape) = bytes.get(start + 2) else {
            return false;
        };
        return match escape {
            b'\\' | b'\'' | b'\"' | b'n' | b'r' | b't' | b'0' => {
                bytes.get(start + 3) == Some(&b'\'')
            }
            b'x' => bytes.get(start + 3..start + 6).is_some_and(|tail| {
                tail[0].is_ascii_hexdigit() && tail[1].is_ascii_hexdigit() && tail[2] == b'\''
            }),
            b'u' => {
                let Some(end) = bytes[start + 3..].iter().position(|byte| *byte == b'}') else {
                    return false;
                };
                let end = start + 3 + end;
                bytes.get(start + 3) == Some(&b'{')
                    && bytes[start + 4..end]
                        .iter()
                        .all(|byte| byte.is_ascii_hexdigit() || *byte == b'_')
                    && bytes.get(end + 1) == Some(&b'\'')
            }
            _ => false,
        };
    }
    let rest = std::str::from_utf8(&bytes[start + 1..]).unwrap_or_default();
    rest.chars()
        .next()
        .is_some_and(|character| bytes.get(start + 1 + character.len_utf8()) == Some(&b'\''))
}

fn strip_rust_comments_and_literals(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut result = String::with_capacity(source.len());
    let mut index = 0;
    let mut block_depth = 0;
    let mut state = "code";
    let mut raw_hashes = 0;

    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        let blank = |byte: u8| {
            if byte == b'\n' || byte == b'\r' {
                byte as char
            } else {
                ' '
            }
        };
        match state {
            "line" => {
                result.push(blank(byte));
                if byte == b'\n' {
                    state = "code";
                }
                index += 1;
            }
            "block" => {
                if byte == b'/' && next == Some(b'*') {
                    block_depth += 1;
                    result.push_str("  ");
                    index += 2;
                } else if byte == b'*' && next == Some(b'/') {
                    block_depth -= 1;
                    result.push_str("  ");
                    index += 2;
                    if block_depth == 0 {
                        state = "code";
                    }
                } else {
                    result.push(blank(byte));
                    index += 1;
                }
            }
            "quoted" | "character" => {
                result.push(blank(byte));
                if byte == b'\\' {
                    if let Some(next) = next {
                        result.push(blank(next));
                        index += 2;
                    } else {
                        index += 1;
                    }
                } else {
                    if (state == "quoted" && byte == b'\"')
                        || (state == "character" && byte == b'\'')
                    {
                        state = "code";
                    }
                    index += 1;
                }
            }
            "raw" => {
                if byte == b'\"'
                    && bytes[index + 1..]
                        .iter()
                        .take(raw_hashes)
                        .all(|candidate| *candidate == b'#')
                {
                    result.push_str(&" ".repeat(raw_hashes + 1));
                    index += raw_hashes + 1;
                    state = "code";
                } else {
                    result.push(blank(byte));
                    index += 1;
                }
            }
            _ => {
                if byte == b'/' && next == Some(b'/') {
                    result.push_str("  ");
                    index += 2;
                    state = "line";
                } else if byte == b'/' && next == Some(b'*') {
                    result.push_str("  ");
                    index += 2;
                    block_depth = 1;
                    state = "block";
                } else if byte == b'\"' {
                    result.push(' ');
                    index += 1;
                    state = "quoted";
                } else if byte == b'\'' && is_rust_char_literal(bytes, index) {
                    result.push(' ');
                    index += 1;
                    state = "character";
                } else if byte == b'r' {
                    let mut cursor = index + 1;
                    while bytes.get(cursor) == Some(&b'#') {
                        cursor += 1;
                    }
                    if bytes.get(cursor) == Some(&b'\"') {
                        raw_hashes = cursor - index - 1;
                        result.push_str(&" ".repeat(cursor - index + 1));
                        index = cursor + 1;
                        state = "raw";
                    } else {
                        result.push(byte as char);
                        index += 1;
                    }
                } else {
                    result.push(byte as char);
                    index += 1;
                }
            }
        }
    }
    result
}

fn dependency_table(header: &str) -> bool {
    let header = header.trim();
    header == "dependencies"
        || header == "dev-dependencies"
        || header == "build-dependencies"
        || header.ends_with(".dependencies")
        || header.ends_with(".dev-dependencies")
        || header.ends_with(".build-dependencies")
}

fn nested_dependency_name(header: &str) -> Option<&str> {
    let (table, name) = header.rsplit_once('.')?;
    dependency_table(table).then_some(name.trim())
}

fn is_openhuman_package(value: &str) -> bool {
    let value = value.trim_matches(&['\"', '\''][..]);
    value == "openhuman" || value.starts_with("openhuman-") || value.starts_with("openhuman_")
}

fn manifest_declares_openhuman(source: &str) -> bool {
    let mut table = String::new();
    let mut nested_dependency = None::<String>;
    for raw in source.lines() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.starts_with('[') && line.ends_with(']') {
            table = line
                .trim_matches(&['[', ']'][..])
                .replace(char::is_whitespace, "");
            nested_dependency = nested_dependency_name(&table).map(str::to_owned);
            if nested_dependency
                .as_deref()
                .is_some_and(is_openhuman_package)
            {
                return true;
            }
            continue;
        }
        if !dependency_table(&table) && nested_dependency.is_none() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if is_openhuman_package(key.trim())
            || (nested_dependency.is_some()
                && key.trim() == "package"
                && is_openhuman_package(value.trim()))
            || value
                .split(|character: char| {
                    !character.is_ascii_alphanumeric() && character != '_' && character != '-'
                })
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .windows(2)
                .any(|parts| parts[0] == "package" && is_openhuman_package(parts[1]))
        {
            return true;
        }
    }
    false
}

fn compact_paths(source: &str) -> String {
    source
        .split("::")
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("::")
}

fn names_openhuman_domain_type(source: &str) -> bool {
    source
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|word| {
            OPENHUMAN_DOMAIN_TYPES.contains(&word)
                || word
                    .strip_prefix("OpenHuman")
                    .is_some_and(|suffix| suffix.chars().next().is_some_and(char::is_uppercase))
        })
}

fn has_openhuman_crate_path(source: &str) -> bool {
    let mut remaining = source;
    while let Some(index) = remaining.find("openhuman_") {
        let after_prefix = &remaining[index + "openhuman_".len()..];
        let name_end = after_prefix
            .find(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .unwrap_or(after_prefix.len());
        if after_prefix[name_end..].starts_with("::") {
            return true;
        }
        remaining = after_prefix;
    }
    false
}

fn source_names_openhuman(source: &str) -> bool {
    let source = compact_paths(&strip_rust_comments_and_literals(source));
    source.contains("openhuman::")
        || has_openhuman_crate_path(&source)
        || names_openhuman_domain_type(&source)
}

fn domain_reference_lines(source: &str) -> Vec<usize> {
    strip_rust_comments_and_literals(source)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = compact_paths(line);
            (line.contains("openhuman::")
                || has_openhuman_crate_path(&line)
                || names_openhuman_domain_type(&line))
            .then_some(index + 1)
        })
        .collect()
}

fn is_known_generic_claude_code_chat_message_debt(path: &str, line: usize) -> bool {
    KNOWN_GENERIC_CLAUDE_CODE_CHAT_MESSAGE_DEBT.contains(&(path, line))
}

fn known_generic_claude_code_chat_message_debt() -> BTreeSet<(String, usize)> {
    KNOWN_GENERIC_CLAUDE_CODE_CHAT_MESSAGE_DEBT
        .iter()
        .map(|(path, line)| ((*path).to_owned(), *line))
        .collect()
}

fn check_generic_claude_code_chat_message_debt(
    observed: &BTreeSet<(String, usize)>,
) -> Result<(), String> {
    let expected = known_generic_claude_code_chat_message_debt();
    let unexpected = observed.difference(&expected).collect::<Vec<_>>();
    let stale = expected.difference(observed).collect::<Vec<_>>();
    if unexpected.is_empty() && stale.is_empty() {
        return Ok(());
    }
    let render = |entries: Vec<&(String, usize)>| {
        entries
            .into_iter()
            .map(|(path, line)| format!("  {path}:{line}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut message = String::from("generic Claude Code ChatMessage debt changed.");
    if !unexpected.is_empty() {
        message.push_str("\nUnexpected references:\n");
        message.push_str(&render(unexpected));
    }
    if !stale.is_empty() {
        message.push_str("\nStale baseline entries:\n");
        message.push_str(&render(stale));
    }
    Err(message)
}

#[test]
fn host_independent_crates_do_not_depend_on_or_name_openhuman_types() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("integration crate is nested under the TinyAgents workspace");
    let crates = workspace.join("crates");

    for manifest in dependency_manifests(workspace) {
        let source = fs::read_to_string(&manifest).expect("manifest is readable");
        assert!(
            !manifest_declares_openhuman(&source),
            "{} declares an OpenHuman dependency",
            manifest.display(),
        );
    }

    let mut rust_sources = Vec::new();
    visit_files(&crates, "rs", &mut rust_sources);
    let mut observed_generic_chat_message_debt = BTreeSet::new();
    for source_path in rust_sources {
        let source = fs::read_to_string(&source_path).expect("Rust source is readable");
        if source_names_openhuman(&source) {
            let relative = source_path
                .strip_prefix(workspace)
                .expect("source belongs to TinyAgents workspace")
                .to_string_lossy();
            let references = domain_reference_lines(&source);
            assert!(
                !references.is_empty(),
                "{} has an OpenHuman reference split across lines",
                source_path.display(),
            );
            for line in references {
                observed_generic_chat_message_debt.insert((relative.to_string(), line));
            }
        }
    }
    if let Err(diff) =
        check_generic_claude_code_chat_message_debt(&observed_generic_chat_message_debt)
    {
        panic!("{diff}");
    }
}

#[test]
fn workspace_root_manifest_is_inside_the_dependency_guard() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("integration crate is nested under the TinyAgents workspace");
    let manifests = dependency_manifests(workspace);

    assert!(manifests.contains(&workspace.join("Cargo.toml")));
    assert!(manifest_declares_openhuman(
        "[workspace.dependencies]\nhost = { package = \"openhuman-core\", version = \"1\" }",
    ));
}

#[test]
fn dependency_forms_cannot_hide_an_openhuman_package() {
    for manifest in [
        "[dependencies.openhuman]\nversion = \"1\"",
        "[target.'cfg(unix)'.dependencies]\nhost = { package = \"openhuman-core\", version = \"1\" }",
        "[workspace.dependencies]\nopenhuman_tools = \"1\"",
        "[build-dependencies.host]\npackage = \"openhuman\"",
        "[dev-dependencies]\nhost = { package = \"openhuman\", version = \"1\" }",
        "[dependencies.host]\npackage = 'openhuman'",
    ] {
        assert!(
            manifest_declares_openhuman(manifest),
            "missed manifest: {manifest}"
        );
    }
}

#[test]
fn whitespace_and_inventory_cannot_hide_openhuman_domain_references() {
    assert!(source_names_openhuman("use openhuman :: AgentProgress;"));
    assert!(source_names_openhuman(
        "fn project(message: ConversationMessage) {}"
    ));
    assert!(source_names_openhuman(
        "fn retain(message: ChatMessage) -> ToolResultMessage { todo!() }"
    ));
    assert!(!source_names_openhuman(
        "// openhuman :: AgentProgress\nlet label = \"ConversationMessage\";"
    ));
}

#[test]
fn lifetimes_and_labels_do_not_hide_code_but_literals_and_comments_do() {
    let lifetime_source = "fn borrow<'a>(message: &'a ChatMessage) { 'retry: loop { let _ = openhuman :: AgentProgress; break 'retry; } }";
    let stripped = strip_rust_comments_and_literals(lifetime_source);
    assert!(stripped.contains("&'a ChatMessage"));
    assert!(compact_paths(&stripped).contains("openhuman::AgentProgress"));
    assert!(source_names_openhuman(lifetime_source));

    let ignored = r#"let marker = '\u{2764}'; let text = "openhuman :: AgentProgress"; /* ConversationMessage */"#;
    assert!(!source_names_openhuman(ignored));
}

#[test]
fn generic_chat_message_debt_is_exact_not_a_path_exclusion() {
    assert!(is_known_generic_claude_code_chat_message_debt(
        "crates/tinyagents-harness/src/providers/claude_code/bridge.rs",
        10,
    ));
    assert!(!is_known_generic_claude_code_chat_message_debt(
        "crates/tinyagents-harness/src/providers/claude_code/bridge.rs",
        11,
    ));
    assert_eq!(
        domain_reference_lines("fn unexpected(message: ChatMessage) {}"),
        vec![1],
    );
    let expected = known_generic_claude_code_chat_message_debt();
    assert!(check_generic_claude_code_chat_message_debt(&expected).is_ok());
    let mut neighboring_new_line = expected.clone();
    neighboring_new_line.insert((
        "crates/tinyagents-harness/src/providers/claude_code/bridge.rs".to_owned(),
        11,
    ));
    assert!(
        check_generic_claude_code_chat_message_debt(&neighboring_new_line)
            .expect_err("a neighboring new reference must fail")
            .contains("Unexpected references")
    );
    let mut stale_entry = expected;
    stale_entry.remove(&(
        "crates/tinyagents-harness/src/providers/claude_code/bridge.rs".to_owned(),
        10,
    ));
    assert!(
        check_generic_claude_code_chat_message_debt(&stale_entry)
            .expect_err("a stale physical debt entry must fail")
            .contains("Stale baseline entries")
    );
}
