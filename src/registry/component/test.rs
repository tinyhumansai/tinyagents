use super::*;

#[test]
fn component_id_new_and_display() {
    let id = ComponentId::new("gpt-4o");
    assert_eq!(id.as_str(), "gpt-4o");
    assert_eq!(id.to_string(), "gpt-4o");
    assert_eq!(ComponentId::from("x"), ComponentId("x".to_owned()));
}

#[test]
fn component_kind_as_str_matches_serde() {
    for kind in ComponentKind::ALL {
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, format!("\"{}\"", kind.as_str()));
        assert_eq!(kind.to_string(), kind.as_str());
    }
}

#[test]
fn component_metadata_builder() {
    let meta = ComponentMetadata::new("lookup_user", ComponentKind::Tool)
        .with_description("looks up a user")
        .with_tag("crm");
    assert_eq!(meta.name(), "lookup_user");
    assert_eq!(meta.kind, ComponentKind::Tool);
    assert_eq!(meta.description.as_deref(), Some("looks up a user"));
    assert_eq!(meta.tags, vec!["crm".to_string()]);
    assert!(meta.aliases.is_empty());
}

#[test]
fn component_metadata_round_trips() {
    let meta = ComponentMetadata::new("default", ComponentKind::Model).with_tag("fast");
    let json = serde_json::to_string(&meta).unwrap();
    let back: ComponentMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(meta, back);
}
