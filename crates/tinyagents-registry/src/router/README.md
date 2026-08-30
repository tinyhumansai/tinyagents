# `registry::router` — high-level model router

The declarative **workload-tier** layer over the named model registry. Where
[`CapabilityRegistry`] resolves a model *by name* and the agent loop fails over
across a [`FallbackPolicy`], `ModelRouter` owns the *policy* that a host's tiers
(`chat-v1`, `reasoning-v1`, `vision-v1`, …) resolve to concrete models, with the
per-tier capability gates and same-family fallback ordering that go with them.

## Public surface

- **`WorkloadRoute`** — one named tier: `alias` → `model`, an optional
  [`CapabilitySet`] gate, and an ordered list of same-family fallback aliases.
  Pure metadata; names a model rather than owning one.
- **`ModelRouter`** — an insertion-ordered table of routes plus an optional
  default alias. Answers the three questions turn assembly needs:
  - `target_model(alias)` — which registered model does this alias forward to?
  - `fallback_policy(alias)` — the `[alias, fallbacks…]` [`FallbackPolicy`] for a
    turn whose primary is `alias` (`None` when there are no alternates).
  - `required_capabilities(alias)` — the [`CapabilitySet`] to stamp on requests
    routed here (`None` for an ungated text tier).

Build with the infallible last-write-wins builder (`with_route`/`with_default`)
or the duplicate-rejecting `register`.

## Design constraints

- **Holds no models, drives no I/O.** It is cheap, cloneable policy read while
  wiring a registry + `RunPolicy`. *What routes where* stays decoupled from *how
  a model is built* (host provider/factory territory).
- **Insertion order is preserved** so `routes()` / `aliases()` iterate
  deterministically — callers projecting the table onto a registry rely on it.
- **Fallback chains lead with the primary** because the crate's
  [`FallbackPolicy::next_after`] expects the current name first and yields each
  subsequent alternate.

## Why (Phase 3 / issue #4249)

This is the crate-owned home for the projection hosts have re-implemented by
hand (OpenHuman's `RouterProvider` + `routes.rs`): register a model per tier
alias, build a fallback policy, stamp a per-turn capability requirement. Moving
the *declaration* of the tier table into the crate lets a host describe its
tiered routing once and hand it over, instead of open-coding alias resolution,
fallback ordering, and capability gating at the turn-assembly boundary.

[`CapabilityRegistry`]: ../capability/index.html
[`CapabilitySet`]: ../../harness/model/struct.CapabilitySet.html
[`FallbackPolicy`]: ../../harness/retry/struct.FallbackPolicy.html
[`FallbackPolicy::next_after`]: ../../harness/retry/struct.FallbackPolicy.html#method.next_after
