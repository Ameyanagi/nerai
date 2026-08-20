# Roadmap

## v0.1 — Foundation

- Ship dense unconstrained `Float64` nonlinear least squares with
  Levenberg-Marquardt.
- Support forward finite-difference Jacobians, observation weights, linear,
  Huber, and soft-L1 losses, and full-rank covariance estimates.
- Report termination, cost, optimality, iterations, and callback counts through
  explicit values.
- Pass the unit, reference, invariant, end-to-end, package, and installed-artifact
  gates in the [v0.1 implementation plan](implementation-plan.md).

## v0.2 — Usability

- Add ergonomic APIs only after v0.1 usage demonstrates repeated friction.
- Expand examples and integration fixtures.
- Publish the first modular-community recipe when the package is useful alone.

## v0.3 — Performance

- Add reproducible benchmarks and representative datasets.
- Optimize measured bottlenecks without weakening correctness or API clarity.
- Add SIMD or specialized backends only behind the same semantic contract.

## v1.0 — Stability

- Document every public symbol and error contract.
- Provide a compatibility and deprecation policy.
- Support the declared OS and architecture matrix in CI.
- Require downstream proof from at least one independent consumer.

## Not planned

Bounds, sparse systems, automatic differentiation, general minimizers, global
optimization, GPU backends, plotting, interpolation, and domain-specific models
are outside v0.1. The complete boundary is maintained in the
[implementation plan](implementation-plan.md#v01-non-goals).
