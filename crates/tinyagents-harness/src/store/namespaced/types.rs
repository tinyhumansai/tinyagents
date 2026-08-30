//! Types for the hierarchical, batch-oriented long-term store.
//!
//! See the module docs on [`super`] for why this exists alongside the flat
//! [`Store`](crate::store::Store) trait.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, TinyAgentsError};

/// A hierarchical namespace: an ordered tuple of path segments.
///
/// `("users", "alice", "memories")` is a child of `("users", "alice")`, which
/// is what makes prefix search and namespace listing meaningful. The flat
/// `&str` namespace the original [`Store`](crate::store::Store) trait
/// uses is the degenerate one-segment case, and converts freely.
///
/// # Validation
///
/// A namespace must be non-empty, every segment must be non-empty, no segment
/// may contain the separator `.`, and the first segment may not be the reserved
/// word `langgraph` (kept reserved for compatibility with stores shared with
/// that ecosystem). Segments are otherwise opaque.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Namespace(pub Vec<String>);

/// Separator used when rendering a namespace as a single string.
pub const NAMESPACE_SEPARATOR: char = '.';

/// Reserved first segment.
const RESERVED_ROOT: &str = "langgraph";

impl Namespace {
    /// Builds a namespace from any iterator of segments, validating it.
    pub fn new<I, S>(segments: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let ns = Namespace(segments.into_iter().map(Into::into).collect());
        ns.validate()?;
        Ok(ns)
    }

    /// The segments, in order.
    pub fn segments(&self) -> &[String] {
        &self.0
    }

    /// Whether this namespace is `prefix` or lies beneath it.
    pub fn starts_with(&self, prefix: &[String]) -> bool {
        self.0.len() >= prefix.len() && self.0[..prefix.len()] == *prefix
    }

    /// Whether this namespace ends with `suffix`.
    pub fn ends_with(&self, suffix: &[String]) -> bool {
        self.0.len() >= suffix.len() && self.0[self.0.len() - suffix.len()..] == *suffix
    }

    /// Rejects namespaces that would break addressing or collide with the
    /// reserved root.
    pub fn validate(&self) -> Result<()> {
        if self.0.is_empty() {
            return Err(TinyAgentsError::Validation(
                "store namespace must have at least one segment".into(),
            ));
        }
        for segment in &self.0 {
            if segment.is_empty() {
                return Err(TinyAgentsError::Validation(
                    "store namespace segments must not be empty".into(),
                ));
            }
            if segment.contains(NAMESPACE_SEPARATOR) {
                return Err(TinyAgentsError::Validation(format!(
                    "store namespace segment {segment:?} must not contain \
                     {NAMESPACE_SEPARATOR:?}"
                )));
            }
        }
        if self.0[0] == RESERVED_ROOT {
            return Err(TinyAgentsError::Validation(format!(
                "{RESERVED_ROOT:?} is a reserved root namespace"
            )));
        }
        Ok(())
    }
}

impl From<&str> for Namespace {
    fn from(value: &str) -> Self {
        Namespace(vec![value.to_string()])
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.join(&NAMESPACE_SEPARATOR.to_string()))
    }
}

/// Time-to-live policy for stored items.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TtlConfig {
    /// Lifetime applied to items written without an explicit TTL, in minutes.
    ///
    /// `None` means items never expire, which is the pre-TTL behaviour and the
    /// default.
    pub default_ttl_minutes: Option<f64>,
    /// Whether reading an item extends its life by the default TTL.
    ///
    /// A sliding window suits caches and working memory; leave it off for a
    /// hard retention bound (which is what makes TTL usable as the answer to
    /// unbounded store growth).
    pub refresh_on_read: bool,
}

/// A stored value together with its address and timestamps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Item {
    /// Namespace the item lives in.
    pub namespace: Namespace,
    /// Key within the namespace.
    pub key: String,
    /// The stored payload.
    pub value: Value,
    /// Unix-epoch milliseconds at first write.
    pub created_at_ms: u64,
    /// Unix-epoch milliseconds at the most recent write (or TTL refresh).
    pub updated_at_ms: u64,
    /// Unix-epoch milliseconds after which the item is invisible, if any.
    pub expires_at_ms: Option<u64>,
}

impl Item {
    /// Whether the item is expired as of `now_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at_ms.is_some_and(|expiry| now_ms >= expiry)
    }
}

/// One comparison in a [`SearchQuery`] filter.
///
/// The operator set mirrors LangGraph's: equality by default, plus explicit
/// `$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`/`$in`/`$exists`.
#[derive(Clone, Debug, PartialEq)]
pub enum FilterOp {
    /// Field equals the value.
    Eq(Value),
    /// Field does not equal the value.
    Ne(Value),
    /// Field is numerically greater than the value.
    Gt(f64),
    /// Field is numerically greater than or equal to the value.
    Gte(f64),
    /// Field is numerically less than the value.
    Lt(f64),
    /// Field is numerically less than or equal to the value.
    Lte(f64),
    /// Field equals one of the values.
    In(Vec<Value>),
    /// Field is present (`true`) or absent (`false`).
    Exists(bool),
}

impl FilterOp {
    /// Evaluates the operator against a field that may be absent.
    pub fn matches(&self, field: Option<&Value>) -> bool {
        match self {
            FilterOp::Exists(expected) => field.is_some() == *expected,
            FilterOp::Eq(expected) => field == Some(expected),
            FilterOp::Ne(expected) => field != Some(expected),
            FilterOp::In(options) => field.is_some_and(|v| options.contains(v)),
            FilterOp::Gt(n) => field.and_then(Value::as_f64).is_some_and(|v| v > *n),
            FilterOp::Gte(n) => field.and_then(Value::as_f64).is_some_and(|v| v >= *n),
            FilterOp::Lt(n) => field.and_then(Value::as_f64).is_some_and(|v| v < *n),
            FilterOp::Lte(n) => field.and_then(Value::as_f64).is_some_and(|v| v <= *n),
        }
    }
}

/// A namespace-prefixed search over stored items.
#[derive(Clone, Debug, Default)]
pub struct SearchQuery {
    /// Only items in this namespace or beneath it are considered.
    pub namespace_prefix: Vec<String>,
    /// Field-path → condition. Every entry must match (conjunction). Paths are
    /// dotted, so `"meta.kind"` addresses a nested object field.
    pub filter: HashMap<String, FilterOp>,
    /// Optional substring match applied to the item's rendered JSON.
    ///
    /// This is the seam where semantic/vector search belongs. There is no
    /// embedding dependency in this crate, so the built-in backends do a plain
    /// case-insensitive substring scan; a backend with an index is free to
    /// interpret it properly, and callers should treat result *order* as
    /// relevance-defined rather than guaranteed.
    pub query: Option<String>,
    /// Maximum items to return. `None` means unlimited.
    pub limit: Option<usize>,
    /// Items to skip before collecting results.
    pub offset: usize,
}

/// A namespace listing query with prefix/suffix wildcards.
#[derive(Clone, Debug, Default)]
pub struct ListNamespacesQuery {
    /// Match namespaces starting with these segments. `*` matches one segment.
    pub prefix: Option<Vec<String>>,
    /// Match namespaces ending with these segments. `*` matches one segment.
    pub suffix: Option<Vec<String>>,
    /// Truncate returned namespaces to at most this many segments, then
    /// deduplicate — the way to enumerate one level of the hierarchy.
    pub max_depth: Option<usize>,
    /// Maximum namespaces to return.
    pub limit: Option<usize>,
    /// Namespaces to skip.
    pub offset: usize,
}

/// One operation in a [`NamespacedStore::batch`] request.
#[derive(Clone, Debug)]
pub enum StoreOp {
    /// Read one item.
    Get {
        /// Namespace to read from.
        namespace: Namespace,
        /// Key to read.
        key: String,
        /// Whether to extend the item's TTL on this read, overriding
        /// [`TtlConfig::refresh_on_read`].
        refresh_ttl: Option<bool>,
    },
    /// Write or delete one item.
    Put {
        /// Namespace to write to.
        namespace: Namespace,
        /// Key to write.
        key: String,
        /// The value, or `None` to delete the item.
        value: Option<Value>,
        /// Lifetime in minutes, overriding [`TtlConfig::default_ttl_minutes`].
        ttl_minutes: Option<f64>,
    },
    /// Search a namespace subtree.
    Search(SearchQuery),
    /// List namespaces.
    ListNamespaces(ListNamespacesQuery),
}

/// The result of one [`StoreOp`], positionally aligned with the request.
#[derive(Clone, Debug, PartialEq)]
pub enum StoreResult {
    /// Result of a `Get`: the item, or `None` if missing or expired.
    Item(Option<Box<Item>>),
    /// Result of a `Put`: nothing to return.
    Ack,
    /// Result of a `Search`.
    Items(Vec<Item>),
    /// Result of a `ListNamespaces`.
    Namespaces(Vec<Namespace>),
}

impl StoreResult {
    /// Unwraps a `Get` result, erroring if the batch returned a different shape.
    pub fn into_item(self) -> Result<Option<Item>> {
        match self {
            StoreResult::Item(item) => Ok(item.map(|b| *b)),
            other => Err(mismatch("Item", &other)),
        }
    }

    /// Unwraps a `Search` result.
    pub fn into_items(self) -> Result<Vec<Item>> {
        match self {
            StoreResult::Items(items) => Ok(items),
            other => Err(mismatch("Items", &other)),
        }
    }

    /// Unwraps a `ListNamespaces` result.
    pub fn into_namespaces(self) -> Result<Vec<Namespace>> {
        match self {
            StoreResult::Namespaces(ns) => Ok(ns),
            other => Err(mismatch("Namespaces", &other)),
        }
    }
}

fn mismatch(expected: &str, got: &StoreResult) -> TinyAgentsError {
    TinyAgentsError::Validation(format!(
        "store batch returned a {got:?} result where {expected} was expected — \
         a backend must return results positionally aligned with the request"
    ))
}
