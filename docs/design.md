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

For v0.1, Nerai chooses dense `Float64` least squares, optional parameter-wise
box bounds, and a small private numerical kernel. This keeps callback, loss,
termination, constraint, and numerical-failure semantics reviewable before
considering an external linear-algebra dependency or a broader optimizer
family. Exact equations and acceptance gates live in the
[implementation plan](implementation-plan.md).

Robust costs use loss-specific scaled formulas. Linear and inlier Huber costs
halve before squaring; Huber outliers use `C * (abs(r) - C / 2)`; soft-L1 uses
bounded ratios selected by the relative sizes of `abs(r)` and `C`. Thus a
representable cost does not depend on an unrepresentable normalized square or
scale square.

The implemented `least_squares()` surface validates reachable problem state
once at solve entry, then evaluates the trusted model directly inside the loop
while enforcing the captured residual dimension and finite outputs. It builds
bound-aware forward or central finite-difference Jacobians and the weighted
robust objective, applies optional explicit parameter scaling and the LM
damping policy fixed in the implementation plan, and maps tolerance, budget,
and numerical-breakdown outcomes to `TerminationReason`. The usual normal-
equation path is backed by a pivoted, scaled Householder-QR fallback without a
public linear-solver switch.

## Mojo 1.0 mutation and invariants

Finite semantic states follow the standard-library nominal-enum pattern:

- `LossKind` is an `Int`-backed struct with constants for linear, Huber, and
  soft-L1.
- `TerminationReason` is an `Int`-backed struct with constants for the three
  convergence reasons, two budget limits, and numerical failure.

The integer discriminants are underscore-prefixed and constructors are used
privately by convention. Direct field mutation or raw integer construction is
outside the public contract.

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
initial parameters, resolved weights, configured residual count, and options.
Both residual-evaluation methods call `validate()` before the model, snapshot
the entry residual count, then check the post-callback declaration, returned
length, and finiteness.

An incoherent or invalid mutation therefore raises before a residual result can
reach a numerical kernel. Coordinated mutation of the model declaration,
problem count, and weights into another valid state is explicit problem
reconfiguration between standalone evaluations or solves; it is not described
as an impossible lifetime-fixed invariant. Each evaluation enforces its entry
dimension. During `least_squares()`, the solver captures the solve-entry
dimension and rejects any change until the solve returns.

The stateful residual callback uses static generic dispatch and requires only a
movable model, not a copyable one. `mut self` permits instrumentation and
model-local caches, while a read-only parameter list keeps parameter ownership
with the problem or caller. The returned list is owned by the caller. Callback
`Error` values propagate; Nerai does not translate a model domain error into
solver convergence or numerical failure.

## Out of scope

Interpolation, plotting, domain-specific models, automatic differentiation, global optimization, and a broad minimizer catalog are outside v0.1.
