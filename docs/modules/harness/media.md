# Media generation tools

Feature: `media` on `tinyagents-harness`. Module: `tinyagents_harness::media`.

The harness exposes image and video generation as ordinary `tinytools::Tool`s
over TinyInference's provider-neutral generators:

| Tool | Generator trait | Default model-visible name |
| --- | --- | --- |
| `GenerateImageTool` | `tinyinference_image::ImageGenerator` | `generate_image` |
| `GenerateVideoTool` | `tinyinference_video::VideoGenerator` | `generate_video` |

`tinyinference_image` and `tinyinference_video` are re-exported from the harness
when the feature is on, so hosts reach generator types through
`tinyagents_harness::tinyinference_image::…` rather than adding their own
dependency.

## Who owns what

- **TinyInference** owns the wire: OpenRouter's `POST /images`, `POST /videos`,
  `GET /videos/{id}`, `GET /videos/{id}/content`; reference inlining (URL,
  `data:` URL, bytes, local path → content parts); output-shape normalization
  (`"16x9"`, `"landscape"`, `"full hd"`); capability pre-flight checks; the
  billing-aware retry policy; and the submit → poll → download job loop.
- **The harness** owns the tool contract: argument parsing (including the loose
  spellings and legacy camelCase aliases models emit), artifact persistence into
  the run's workspace, result wording, and tool policy metadata.
- **The host** owns the generator (and so the credential and endpoint), the
  tool's visible name, the fallback output root, and whether a local reference
  file may leave the machine (`MediaOutput::with_reference_policy`).

## Using OpenRouter directly or through a backend

```rust
use std::sync::Arc;
use tinyagents_harness::media::{GenerateImageTool, MediaOutput};
use tinyagents_harness::tinyinference_image::{MediaAuth, MediaTransport, OpenRouterImageGenerator};

// Direct: the caller's OpenRouter key.
let direct = OpenRouterImageGenerator::new(MediaAuth::ApiKey(key));

// Proxied: a backend that forwards OpenRouter's media routes verbatim.
let proxied = OpenRouterImageGenerator::with_transport(
    MediaTransport::new(MediaAuth::Bearer(Arc::new(|| session_token())))
        .with_base_url("https://api.example.com/agent-integrations/openrouter"),
);

let tool = GenerateImageTool::new(Arc::new(proxied), MediaOutput::new(fallback_root))
    .with_name("media_generate_image");
```

## Billing safety

Generation is billed on submit, and a model that retries a failure pays again.
The tools are built so that cannot happen quietly:

- The tools reject an empty successful image or video response as a billed
  non-delivery and tell the model not to retry.
- A video job whose status reads `completed` before its outputs exist keeps
  polling instead of failing.
- A timed-out video job names its id and can be collected with `resume_job_id`
  without a new submit. Image-generation errors cannot be resumed.
- Tool policy declares `payment`, `network`, `external_service` and
  `writes_files`, and `idempotent: false`, so hosts never replay a call after a
  crash.

## Local references

Reference paths are canonicalized and resolved against the run's workspace root.
Without a host policy they must stay inside it (no `..`, no absolute paths
elsewhere, and no symlinks pointing outside it), so a model cannot read an
arbitrary file and upload it to a third party. A custom
`MediaOutput::with_reference_policy` can override this to permit out-of-workspace
references when the host policy allows.
