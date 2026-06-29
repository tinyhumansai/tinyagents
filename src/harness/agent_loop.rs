//! Model-tool-model agent loop.
//!
//! Owns the default harness execution loop: build a model request, invoke the
//! model, execute requested tools, append tool results, and repeat until the run
//! reaches a final assistant response or a configured limit.
