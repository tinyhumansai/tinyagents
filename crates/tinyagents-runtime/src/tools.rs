use std::collections::BTreeMap;

use tinytools::ToolSpec;

use crate::RuntimeError;

/// An immutable, model-visible tool declaration set for one session turn.
///
/// It records declarations only; choosing which tools are permitted and wiring
/// their executors remains a host/driver responsibility.
#[derive(Clone, Debug, Default)]
pub struct ToolSnapshot {
    specs: Vec<ToolSpec>,
}

impl ToolSnapshot {
    /// Validates and freezes a tool declaration set.
    ///
    /// Identical repeated declarations are deduplicated.  A shared name with
    /// different contents is rejected rather than silently choosing one.
    pub fn new(specs: Vec<ToolSpec>) -> Result<Self, RuntimeError> {
        let mut names = BTreeMap::<String, ToolSpec>::new();
        for spec in specs {
            if let Some(existing) = names.get(&spec.name) {
                let same = existing.description == spec.description
                    && existing.parameters == spec.parameters;
                if !same {
                    return Err(RuntimeError::ToolNameCollision(spec.name));
                }
                continue;
            }
            names.insert(spec.name.clone(), spec);
        }
        Ok(Self {
            specs: names.into_values().collect(),
        })
    }

    /// Returns the frozen declarations in stable name order.
    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }
}
