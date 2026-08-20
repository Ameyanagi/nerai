# Design

## Principles

- Mojo is the runtime implementation language.
- Prefer pure Mojo and safe standard-library APIs.
- Keep the root API small, typed, documented, and testable.
- Separate semantic contracts from optimized CPU, SIMD, GPU, terminal, or
  rendering backends.
- Establish correctness and reference fixtures before optimization.
- Make invalid public configuration unrepresentable when practical; otherwise
  reject it explicitly.
- Preserve source mappings, numerical tolerances, ownership, and provenance as
  first-class data when the domain requires them.
- Do not add a framework-wide array, executor, renderer, or application model.

## Tradeoffs

The project accepts a narrower initial feature set in exchange for reviewable
contracts and sparse dependencies. Generated tables are acceptable when their
sources, Unicode or data version, licenses, checksums, and deterministic update
procedure are committed. Consumers must not need the generator toolchain.

For v0.1, Nerai chooses dense unconstrained `Float64` least squares and a small
private numerical kernel. This keeps callback, loss, termination, and numerical
failure semantics reviewable before considering an external linear-algebra
dependency or a broader optimizer family. Exact equations and acceptance gates
live in the [implementation plan](implementation-plan.md).

Robust costs use loss-specific scaled formulas. Linear and inlier Huber costs
halve before squaring; Huber outliers use `C * (abs(r) - C / 2)`; soft-L1 uses
bounded ratios selected by the relative sizes of `abs(r)` and `C`. Thus a
representable cost does not depend on an unrepresentable normalized square or
scale square.

## Mojo 1.0 mutation and invariants

Mojo 1.0 does not make an underscore-prefixed struct field private. A caller
with a mutable value can assign that field directly. Nerai therefore uses total
representations for finite semantic states:

- `LossKind` uses `Optional[Bool]`, whose three states map exactly to linear,
  Huber, and soft-L1.
- `TerminationReason` uses `Bool` times `Optional[Bool]`, whose six states map
  exactly to the three convergence reasons, two budget limits, and numerical
  failure.

Integer discriminants and unchecked constructor flags are excluded. Direct
field mutation can select another documented state but cannot create an unknown
loss or termination reason.

`LossEvaluation` and `LeastSquaresResult` are mutable numeric snapshots. Their
floating-point and collection fields cannot encode finiteness or cross-field
relationships in their storage types. `LeastSquaresResult` validates its
constructor inputs and provides `validate()` for callers that mutate a report.
Its only query, `converged()`, depends solely on the total termination value.
`LossEvaluation` has no operation that consumes its mutable fields. Future APIs
that accept either snapshot must revalidate every numeric and shape invariant at
their boundary.

`LeastSquaresOptions` and `LeastSquaresProblem` also contain writable numeric
and collection fields. Options construction and `validate()` enforce finite
positive enabled tolerances, loss scale, damping values, and finite-difference
step; positive budgets; and ordered damping bounds. A problem owns its model,
initial parameters, resolved weights, declared residual count, and options.
Both residual-evaluation methods call `validate()` before the model, then check
the returned length and finiteness. Consequently mutation through safe public
field and collection operations cannot reach a numerical kernel or be silently
interpreted as another configuration. A caller can still mutate a value into an
invalid snapshot, so passing it to a later solver or evaluator can raise.

The stateful residual callback uses static generic dispatch. `mut self` permits
instrumentation and model-local caches, while a read-only parameter list keeps
parameter ownership with the problem or caller. The returned list is owned by
the caller. Callback `Error` values propagate; Nerai does not translate a model
domain error into solver convergence or numerical failure.

## Out of scope

Interpolation, plotting, domain-specific models, automatic differentiation, global optimization, and a broad minimizer catalog are outside v0.1.
