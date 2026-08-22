# Architecture

Nerai owns Optimization problems, residuals, Jacobians, robust losses, solver state, termination reasons, results, and uncertainty estimates.

## Dependency boundary

Allowed ecosystem dependencies: Mojo standard library only until a narrowly justified linear-algebra dependency is selected.
Expected downstream consumers: Scientific fitting, calibration, inverse-problem, and parameter-estimation applications.

Dependencies point from applications and higher-level packages toward smaller
foundations. This repository must never import a downstream consumer. New
dependencies require a documented need and must not force unrelated users to
install an application, renderer, language layer, or scientific stack.

## Layers

Implemented areas are problem and result contracts, termination, robust losses,
the private dense numerical kernel, bound-aware forward and central
finite-difference Jacobians, the Levenberg-Marquardt solve loop, and covariance
estimation. General minimizers remain outside the repository boundary.

The package root exports only the small documented public surface. Algorithms,
generated tables, platform details, and backend implementations remain in
their owning modules. Generic Mojo-native buffers, spans, strings, and
collections are preferred over an ecosystem-specific universal container.

## Data flow

Input validation occurs at the public boundary. Internal layers operate on
explicit typed values, produce deterministic outputs for deterministic inputs,
and report invalid state rather than silently replacing it with a default.
I/O, clocks, randomness, terminal queries, filesystem access, and accelerator
selection stay at explicit effect or backend boundaries.

## v0.1 solver boundary

The [v0.1 implementation plan](implementation-plan.md) is the source of truth
for the objective equation, callback and Jacobian conventions, termination
precedence, counters, numerical gates, and issue dependency order. Architectural
changes to the solver update that contract before implementation.

## Implemented problem boundary

`LeastSquaresProblem[M: ResidualModel]` owns one concrete, potentially move-only
model and dispatches without an FFI boundary, heap-erased callback, or runtime
type switch. The model exposes its currently configured residual count and a
raising `mut self` evaluation method, so instrumented or cached models can
update their own state. Model errors propagate to the caller. Nerai validates
initial parameters, observation weights, configuration, model-declared shape,
and every returned residual before later numerical layers receive them.

Mojo 1.0 public fields make coherent direct mutation an explicit problem
reconfiguration between standalone evaluations or before a solve. Each
evaluation snapshots the validated entry residual count and checks the model's
declaration again after callback return. A callback cannot change its dimension
during one call. `least_squares()` separately captures the solve-entry dimension
and enforces it for the entire solve, even if the public problem was reconfigured
between standalone calls.

The problem uses Mojo `List[Float64]` values and does not create a public matrix
or array abstraction. `least_squares()` builds a private column-major Jacobian
so finite-difference columns, transpose products, and QR scans remain
contiguous. Normal equations retain the common fast path; a small Cholesky
pivot switches to a reusable pivoted Householder-QR workspace over the augmented
damped system. This stability choice does not add a public solver-mode option.
Only the documented `LeastSquaresResult` crosses the package boundary.
