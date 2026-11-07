use std::{
    any::TypeId,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    sync::Arc,
};

use tinytools::ToolSpec;

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod tests;

use thiserror::Error;

/// Errors while freezing or restoring a tool declaration snapshot.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ToolSnapshotError {
    /// Two different declarations used the same name.
    #[error("tool snapshot has conflicting declarations for `{0}`")]
    NameCollision(String),
    /// Stored declaration JSON could not be decoded.
    #[error("decode recorded tool declarations: {0}")]
    Decode(String),
}

/// Type-erased host-owned state carried alongside a tool snapshot.
///
/// Extensions are retained by in-memory snapshot copies and are excluded from
/// the serialized tool declarations.
pub trait ToolSnapshotExtension: std::any::Any + fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: std::any::Any + fmt::Debug + Send + Sync> ToolSnapshotExtension for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Default)]
pub struct ToolSnapshotExtensions {
    values: HashMap<TypeId, Arc<dyn ToolSnapshotExtension>>,
}

impl fmt::Debug for ToolSnapshotExtensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolSnapshotExtensions")
            .field("len", &self.values.len())
            .finish()
    }
}

impl ToolSnapshotExtensions {
    pub fn insert<T: ToolSnapshotExtension>(&mut self, value: T) {
        self.values.insert(TypeId::of::<T>(), Arc::new(value));
    }

    pub fn get<T: ToolSnapshotExtension>(&self) -> Option<&T> {
        self.values
            .get(&TypeId::of::<T>())?
            .as_ref()
            .as_any()
            .downcast_ref::<T>()
    }
}

/// An immutable, model-visible tool declaration set for one session turn.
///
/// It records declarations only; choosing which tools are permitted and wiring
/// their executors remains a host/driver responsibility. Typed extensions stay
/// in memory and do not appear in the serialized transcript value.
#[derive(Clone, Debug, Default)]
pub struct ToolSnapshot {
    specs: Vec<ToolSpec>,
    exact: bool,
    /// Names the host still offers without declaring them in `specs` (a
    /// deferred catalogue the model discovers). In memory only.
    offered: BTreeSet<String>,
    extensions: ToolSnapshotExtensions,
}

impl ToolSnapshot {
    /// Validates and freezes a tool declaration set.
    ///
    /// Identical repeated declarations are deduplicated.  A shared name with
    /// different contents is rejected rather than silently choosing one.
    pub fn new(specs: Vec<ToolSpec>) -> Result<Self, ToolSnapshotError> {
        let mut names = BTreeMap::<String, ToolSpec>::new();
        for spec in specs {
            if let Some(existing) = names.get(&spec.name) {
                let same = existing.description == spec.description
                    && existing.parameters == spec.parameters;
                if !same {
                    return Err(ToolSnapshotError::NameCollision(spec.name));
                }
                continue;
            }
            names.insert(spec.name.clone(), spec);
        }
        Ok(Self {
            specs: names.into_values().collect(),
            exact: false,
            offered: BTreeSet::new(),
            extensions: ToolSnapshotExtensions::default(),
        })
    }

    /// Marks this snapshot as a one-off declaration set. It is sent exactly
    /// as supplied and is not retained for a later turn.
    pub fn exact(mut self) -> Self {
        self.exact = true;
        self
    }

    pub fn is_exact(&self) -> bool {
        self.exact
    }

    /// Returns the frozen declarations in stable name order.
    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    /// Names tools the host still offers without declaring them here, such as
    /// a deferred catalogue the model discovers. They are never serialized;
    /// they only tell [`Self::with_retained_offered`] which recorded
    /// declarations are not retired.
    pub fn with_offered_names<N: Into<String>>(
        mut self,
        names: impl IntoIterator<Item = N>,
    ) -> Self {
        self.offered.extend(names.into_iter().map(Into::into));
        self
    }

    /// Whether the host still offers `name`: declared here or named through
    /// [`Self::with_offered_names`].
    pub fn offers(&self, name: &str) -> bool {
        self.offered.contains(name) || self.specs.iter().any(|spec| spec.name == name)
    }

    /// Attaches an immutable, host-defined value keyed by its Rust type.
    pub fn with_extension<T: ToolSnapshotExtension>(mut self, extension: T) -> Self {
        self.extensions.insert(extension);
        self
    }

    /// Reads a host-defined extension without serializing it into tool specs.
    pub fn extension<T: ToolSnapshotExtension>(&self) -> Option<&T> {
        self.extensions.get::<T>()
    }

    /// A narrowed copy holding only the declarations `keep` accepts, with the
    /// one-off ([`Self::exact`]) marker preserved. The source is untouched.
    pub fn retaining(&self, keep: impl Fn(&ToolSpec) -> bool) -> Self {
        Self {
            specs: self
                .specs
                .iter()
                .filter(|spec| keep(spec))
                .cloned()
                .collect(),
            exact: self.exact,
            offered: self.offered.clone(),
            extensions: self.extensions.clone(),
        }
    }

    /// The declarations as the JSON a transcript `tools` record stores.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.specs).unwrap_or(serde_json::Value::Array(Vec::new()))
    }

    /// Restores a snapshot from a transcript `tools` record.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, ToolSnapshotError> {
        let specs: Vec<ToolSpec> = serde_json::from_value(value.clone())
            .map_err(|error| ToolSnapshotError::Decode(error.to_string()))?;
        Self::new(specs)
    }

    /// This snapshot plus every declaration of `recorded` whose name it does
    /// not already carry. A name present in both keeps this snapshot's
    /// declaration: the live host is authoritative for a tool it still
    /// supplies. Returns the merged snapshot and how many were retained.
    pub fn with_retained(
        &self,
        recorded: &ToolSnapshot,
    ) -> Result<(Self, usize), ToolSnapshotError> {
        self.merge_recorded(recorded, |_| true)
    }

    /// Like [`Self::with_retained`], but a recorded declaration is kept only
    /// if the host still offers its name through
    /// [`Self::with_offered_names`]. A name the host neither declares nor
    /// offers is retired and dropped.
    pub fn with_retained_offered(
        &self,
        recorded: &ToolSnapshot,
    ) -> Result<(Self, usize), ToolSnapshotError> {
        self.merge_recorded(recorded, |spec| self.offered.contains(&spec.name))
    }

    fn merge_recorded(
        &self,
        recorded: &ToolSnapshot,
        keep: impl Fn(&ToolSpec) -> bool,
    ) -> Result<(Self, usize), ToolSnapshotError> {
        if self.exact {
            return Ok((self.clone(), 0));
        }
        let retained: Vec<ToolSpec> = recorded
            .specs
            .iter()
            .filter(|spec| !self.specs.iter().any(|live| live.name == spec.name))
            .filter(|spec| keep(spec))
            .cloned()
            .collect();
        if retained.is_empty() {
            return Ok((self.clone(), 0));
        }
        let count = retained.len();
        let mut merged = Self::new(self.specs.iter().cloned().chain(retained).collect())?;
        merged.offered = self.offered.clone();
        merged.extensions = self.extensions.clone();
        Ok((merged, count))
    }
}
