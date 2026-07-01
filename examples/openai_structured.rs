//! Structured output (JSON Schema) against the real OpenAI API.
//!
//! Sets [`RunPolicy::default_response_format`] to a
//! [`ResponseFormat::json_schema`] describing a `{sentiment, score}` object.
//! The harness attaches it to every model request and, on the final response,
//! extracts the parsed JSON into [`AgentRun::structured`].
//!
//! Run with:
//!
//! ```text
//! cargo run --example openai_structured
//! ```

use std::sync::Arc;

use serde_json::json;

use tinyagents::Result;
use tinyagents::harness::message::Message;
use tinyagents::harness::model::ResponseFormat;
use tinyagents::harness::providers::openai::OpenAiModel;
use tinyagents::harness::runtime::{AgentHarness, RunPolicy};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let model = OpenAiModel::from_env()?;
    println!("=== OpenAI structured output ===");
    println!("model: {}", model.model());

    // Constrain the model to a strict JSON schema.
    let schema = ResponseFormat::json_schema(
        "sentiment",
        json!({
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"]
                },
                "score": {
                    "type": "number",
                    "description": "Confidence between 0 and 1."
                }
            },
            "required": ["sentiment", "score"],
            "additionalProperties": false
        }),
    );

    let policy = RunPolicy {
        default_response_format: Some(schema),
        ..RunPolicy::default()
    };

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("openai", Arc::new(model))
        .set_default_model("openai")
        .with_policy(policy);

    let review = "Honestly the best purchase I've made all year — works flawlessly!";
    println!("review: {review}\n");

    let run = harness
        .invoke_default(
            &(),
            vec![
                Message::system(
                    "Classify the sentiment of the user's product review. \
                     Respond only with the requested JSON object.",
                ),
                Message::user(review),
            ],
        )
        .await?;

    println!("raw text  : {}", run.text().unwrap_or_default());
    match &run.structured {
        Some(value) => {
            println!("structured: {value}");
            println!(
                "  sentiment = {}, score = {}",
                value.get("sentiment").unwrap_or(&json!(null)),
                value.get("score").unwrap_or(&json!(null)),
            );
        }
        None => println!("structured: <none extracted>"),
    }

    Ok(())
}
