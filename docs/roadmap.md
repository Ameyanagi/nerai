# Roadmap

## v0.1 — Foundation

- Ship dense `Float64` nonlinear least squares with Levenberg-Marquardt and
  optional parameter-wise box bounds.
- Support bound-aware forward and central finite-difference Jacobians,
  observation weights, explicit parameter scaling, linear, Huber, and soft-L1
  losses, adaptive QR stabilization, and full-rank covariance estimates.
- Provide `CurveFit` for paired data while preserving the small residual-model
  API for expert use.
- Report termination, cost, optimality, iterations, and callback counts through
  explicit values.
- Pass the unit, reference, invariant, end-to-end, package, and installed-artifact
  gates in the [v0.1 implementation plan](implementation-plan.md).

## v0.2 — Usability

- Refine the existing front doors only when downstream use demonstrates
  repeated friction.
- Expand examples, diagnostic guidance, and integration fixtures.
- Publish the first modular-community recipe when the package is useful alone.

## v0.3 — Performance

- Extend the reproducible least-squares profile matrix to larger parameter
  counts and real downstream models.
- Continue optimizing measured bottlenecks without weakening correctness or API
  clarity.
- Extend internal SIMD or specialized backends only when scalar differential
  tests and isolated measurements demonstrate a win.

## v1.0 — Stability

- Document every public symbol and error contract.
- Provide a compatibility and deprecation policy.
- Support the declared OS and architecture matrix in CI.
- Require downstream proof from at least one independent consumer.

## Not planned

Sparse systems, automatic differentiation, general minimizers, global
optimization, GPU backends, plotting, interpolation, and domain-specific models
are outside v0.1. The complete boundary is maintained in the
[implementation plan](implementation-plan.md#v01-non-goals).
