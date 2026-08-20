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

Planned implementation areas: problem and result contracts, termination, least-squares solvers, Jacobians, covariance, robust losses, and later BFGS/L-BFGS minimizers.

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

`LeastSquaresProblem[M: ResidualModel]` owns one concrete model and dispatches
without an FFI boundary, heap-erased callback, or runtime type switch. The
model exposes a fixed residual count and a raising `mut self` evaluation method,
so instrumented or cached models can update their own state. Model errors
propagate to the caller. Nerai validates initial parameters, observation
weights, configuration, model-declared shape, and every returned residual
before later numerical layers receive them.

The problem uses Mojo `List[Float64]` values and does not create a public matrix
or array abstraction. The current public method evaluates residuals only; it
does not yet calculate a Jacobian, objective, step, or solution.
