# Security Policy

TinyAgents is an agent orchestration framework. Security-sensitive areas
include tool execution, model-generated workflow source, registry binding,
policy checks, sandbox boundaries, prompt and context handling, stores, and
credentials passed through runtime context.

## Supported Versions

TinyAgents is pre-1.0. Security fixes target the `main` branch until the project
starts maintaining release branches.

## Reporting A Vulnerability

Please do not open a public issue for a suspected vulnerability.

Report security concerns by emailing `security@tinyhumans.ai` with:

- a description of the issue
- affected versions or commits
- reproduction steps or proof of concept
- impact assessment
- any suggested fix or mitigation

We will acknowledge reports as quickly as practical and coordinate disclosure
before publishing details.

## Scope

Examples of in-scope issues:

- generated `.rag` or `.ragsh` source bypassing registry or policy checks
- unintended tool, model, store, or filesystem access
- unsafe handling of credentials or secrets
- prompt/context leakage across runs, forks, or sub-agents
- checkpoint, cache, or store isolation failures
- dependency vulnerabilities with a practical exploit path in TinyAgents

Examples generally out of scope:

- model hallucination or low-quality model output by itself
- unsafe workflows caused by intentionally granting a tool broad authority
- vulnerabilities in downstream applications that use TinyAgents incorrectly

## Security Design Direction

TinyAgents should treat generated workflow source as untrusted. Declarative
workflow input should be parsed, resolved, bound to registries, checked against
policy, and compiled before execution. Runtime code should not install
model-generated topology directly.
