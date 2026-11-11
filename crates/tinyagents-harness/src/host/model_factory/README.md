# Host model factory

This module assembles a turn's primary model, optional workload-tier models,
and summarizer from host-owned construction policy. It keeps provider
selection and authorization at the host boundary while giving the harness a
stable description of each requested route.

## Public surface

`types.rs` defines the serde-compatible `BuildOpts`, `ModelPurpose`, and
`ModelBuildRequest` values, plus the `ModelBuildGate` and `ModelFactory`
contracts. Hosts implement the gate to authorize each construction request and
the async factory to return a model with its resolved provider/model route.
Both adapters are `Sync`, and factory futures are `Send`, so turn assembly can
run inside the host's sendable task boundaries.

`assembly.rs` owns `TurnModelSetBuilder` and `TurnModelSet`. It builds the
primary route, independently builds workload tiers (omitting unavailable or
gated tiers), builds the summarizer, and records the concrete routes and error
slots for later resolution. Await `TurnModelSetBuilder::build` or
`TurnModelSet::build`; each primary, workload, and summary factory request is
awaited before assembly continues. A primary pin does not implicitly pin
workload routes. An explicit summarization role with no summary pin uses that
role's configured default.

`subprocess.rs` adapts Claude Code and Claude Agent SDK route values to the
existing provider implementations. Hosts must run `ModelBuildGate` before
constructing these subprocess-backed models. The injected constructor supports
tests and hosts that own credential or process setup themselves.

## Operational constraints

- This module describes construction requests; it does not own product
  configuration, credentials, or provider policy.
- Workload failures are omitted from the turn model set; primary and
  summarization construction failures are returned to the host.
- Route aliases and concrete provider/model identities remain distinct. Use
  `ResolvedModelRoute` for the concrete result.
- Keep value types limited to serde and standard-library data so hosts can
  persist or transport requests without importing runtime-specific types.
