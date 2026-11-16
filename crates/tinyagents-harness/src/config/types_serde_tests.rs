//! Serde and JSON-schema parity tests for [`RequiredOutput`].
//!
//! OpenHuman reads this contract from `[agent.required_output]` in its TOML
//! config through its own mirror type, `RequiredOutputContract`. The mirror is
//! reproduced verbatim below so these tests pin that the upstream derives read
//! the same files, write the same bytes and describe the same JSON schema —
//! the guarantee a host needs to swap its mirror for this type.

use super::*;
use serde_json::json;

/// OpenHuman's `config::schema::agent::RequiredOutputContract`, verbatim
/// (derives, attributes and field docs), as the parity oracle.
#[derive(
    Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(default)]
struct RequiredOutputContract {
    /// The JSON key that identifies the required block (e.g. `"thoughts"`). The
    /// contract is satisfied when the reply contains a JSON object carrying this
    /// key — plus every key in [`required_keys`](Self::required_keys) — with a
    /// non-null value. A blank `block_key` makes the contract inert (enforcement
    /// is skipped), so it is a safe no-op default.
    pub block_key: String,
    /// Additional sibling keys that must also be present and non-null in the
    /// same block (e.g. `["next_action"]`). The `block_key` is always required
    /// and need not be repeated here.
    pub required_keys: Vec<String>,
}

/// How a host embeds the contract: an optional field of its agent section.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct HostAgent<T> {
    #[serde(default)]
    required_output: Option<T>,
}

/// TOML inputs a host config may carry for the contract.
const TOML_CASES: &[&str] = &[
    "[required_output]\nblock_key = \"thoughts\"\nrequired_keys = [\"next_action\"]\n",
    "[required_output]\nblock_key = \"thoughts\"\n",
    "[required_output]\nrequired_keys = [\"next_action\"]\n",
    "[required_output]\n",
    "",
];

#[test]
fn reads_a_full_toml_contract() {
    let agent: HostAgent<RequiredOutput> = toml::from_str(TOML_CASES[0]).unwrap();
    let contract = agent.required_output.expect("section present");
    assert_eq!(contract.block_key, "thoughts");
    assert_eq!(contract.required_keys, vec!["next_action".to_string()]);
    assert!(contract.is_active());
}

#[test]
fn omitted_fields_take_the_inert_defaults() {
    let only_key: RequiredOutput =
        serde_json::from_value(json!({"block_key": "thoughts"})).unwrap();
    assert_eq!(only_key, RequiredOutput::new("thoughts"));

    // A missing `block_key` is no longer a parse error: it is the inert blank
    // key, exactly as OpenHuman's mirror reads it.
    let no_key: RequiredOutput =
        serde_json::from_value(json!({"required_keys": ["next_action"]})).unwrap();
    assert_eq!(no_key.block_key, "");
    assert_eq!(no_key.required_keys, vec!["next_action".to_string()]);
    assert!(!no_key.is_active());

    let empty: RequiredOutput = serde_json::from_value(json!({})).unwrap();
    assert_eq!(empty, RequiredOutput::default());

    let absent: HostAgent<RequiredOutput> = toml::from_str("").unwrap();
    assert!(absent.required_output.is_none());
}

#[test]
fn serializes_every_field() {
    assert_eq!(
        serde_json::to_value(RequiredOutput::default()).unwrap(),
        json!({"block_key": "", "required_keys": []})
    );
    assert_eq!(
        toml::to_string(&RequiredOutput {
            block_key: "thoughts".into(),
            required_keys: vec!["next_action".into()],
        })
        .unwrap(),
        "block_key = \"thoughts\"\nrequired_keys = [\"next_action\"]\n"
    );
}

#[test]
fn toml_and_json_match_openhumans_mirror_byte_for_byte() {
    for input in TOML_CASES {
        let ours: HostAgent<RequiredOutput> = toml::from_str(input).unwrap();
        let theirs: HostAgent<RequiredOutputContract> = toml::from_str(input).unwrap();
        assert_eq!(
            ours.required_output
                .as_ref()
                .map(|c| (&c.block_key, &c.required_keys)),
            theirs
                .required_output
                .as_ref()
                .map(|c| (&c.block_key, &c.required_keys)),
            "parsed values differ for {input:?}"
        );
        assert_eq!(
            toml::to_string(&ours).unwrap(),
            toml::to_string(&theirs).unwrap(),
            "TOML output differs for {input:?}"
        );
        assert_eq!(
            serde_json::to_string(&ours).unwrap(),
            serde_json::to_string(&theirs).unwrap(),
            "JSON output differs for {input:?}"
        );
        // The re-serialized TOML reads back to the same value.
        let again: HostAgent<RequiredOutput> =
            toml::from_str(&toml::to_string(&ours).unwrap()).unwrap();
        assert_eq!(again.required_output, ours.required_output);
    }
}

/// The root schema of `T` without the type-level `title`/`description`, which
/// name the Rust type rather than describe the wire shape.
fn wire_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    let mut schema = serde_json::to_value(schemars::schema_for!(T)).unwrap();
    let object = schema.as_object_mut().unwrap();
    object.remove("title");
    object.remove("description");
    schema
}

#[test]
fn json_schema_matches_openhumans_mirror() {
    let ours = wire_schema::<RequiredOutput>();
    assert_eq!(ours, wire_schema::<RequiredOutputContract>());
    assert_eq!(ours["type"], json!("object"));
    assert!(ours.get("required").is_none(), "no field is required");
    assert_eq!(ours["properties"]["block_key"]["default"], json!(""));
    assert_eq!(ours["properties"]["required_keys"]["default"], json!([]));
    assert_eq!(
        schemars::schema_for!(RequiredOutput).as_value()["title"],
        json!("RequiredOutput")
    );
}
