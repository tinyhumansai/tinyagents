//! Feature-gated tracing macros shared by the TinyAgents crates.

#[cfg(feature = "tracing")]
pub use tracing::{debug, error, info, trace, warn};

#[cfg(not(feature = "tracing"))]
#[doc(hidden)]
#[macro_export]
macro_rules! debug {
    ($($token:tt)*) => {{ let _ = stringify!($($token)*); }};
}

#[cfg(not(feature = "tracing"))]
#[doc(hidden)]
#[macro_export]
macro_rules! info {
    ($($token:tt)*) => {{ let _ = stringify!($($token)*); }};
}

#[cfg(not(feature = "tracing"))]
#[doc(hidden)]
#[macro_export]
macro_rules! warn {
    ($($token:tt)*) => {{ let _ = stringify!($($token)*); }};
}

#[cfg(not(feature = "tracing"))]
#[doc(hidden)]
#[macro_export]
macro_rules! error {
    ($($token:tt)*) => {{ let _ = stringify!($($token)*); }};
}

#[cfg(not(feature = "tracing"))]
#[doc(hidden)]
#[macro_export]
macro_rules! trace {
    ($($token:tt)*) => {{ let _ = stringify!($($token)*); }};
}
