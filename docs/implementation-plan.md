# Nonlinear least-squares v0.1 implementation plan

This is the execution plan for Nerai v0.1. Each `NERAI-###` section is one
reviewable issue. Work proceeds in dependency order; an issue is complete only
when every acceptance check in that section is green.

## Release contract

Nerai v0.1 solves deterministic, unconstrained, dense nonlinear least-squares
problems in `Float64` using Levenberg-Marquardt. It provides finite-difference
Jacobians, observation weights, linear, Huber, and soft-L1 losses, covariance
estimation for full-rank solutions, and explicit termination reports.

For parameters `x`, raw residuals `r_i(x)`, observation weights `w_i`, and loss
scale `C`, the objective is

```text
F(x) = 1/2 sum_i C^2 rho(((w_i r_i(x)) / C)^2)
```

The contract has these consequences:

- A weight multiplies its residual. All weights are finite and non-negative,
  and at least one is positive. A zero weight excludes one observation.
- `C` is finite and positive. Linear loss always reduces to
  `1/2 sum_i (w_i r_i)^2`, independent of `C`.
- Per-residual loss arithmetic accepts every finite residual and positive finite
  scale. It returns a finite cost whenever the mathematical cost is within the
  finite `Float64` range and raises only when that result overflows. Stable
  piecewise formulas do not require `(r / C)^2` or `C^2` to be representable.
- A residual callback returns the same non-empty length throughout one solve.
  Each standalone evaluation snapshots its validated entry dimension and
  rejects a declaration change during that callback. Every returned residual
  is finite.
- For fixed model configuration, equal parameter values produce equal residual
  values throughout a solve. `mut self` may update caches and counters only when
  those changes are observationally irrelevant to the mathematical residual
  mapping. This semantic-determinism obligation is documented rather than
  guessed from repeated calls.
- Coherent direct mutation may explicitly reconfigure a problem between
  standalone evaluations or before a solve. A solver captures the solve-entry
  dimension and rejects changes until that solve returns.
- v0.1 accepts at least one parameter and requires the residual count to be at
  least the parameter count.
- A user-supplied or finite-difference Jacobian describes the raw residual:
  `J[i, j] = d r_i / d x_j`. Weighting and robust-loss transformations belong
  to the solver, not to the callback.
- Dense matrices are private row-major implementation values. Nerai does not
  introduce a public universal array type.

Invalid problem configuration, a callback error, a changed residual length, or
non-finite callback values at any point raise `Error`. Nerai never converts a
user/model contract violation into solver status. An internal arithmetic
breakdown after a complete valid accepted state returns that state with
`TerminationReason.NUMERICAL_FAILURE`; an initial internal breakdown raises
because no valid result exists yet. A future recoverable domain-invalid model
outcome requires an explicit typed callback result and is not inferred from
NaN or infinity.

## Termination and reporting

The solver checks a newly accepted state in this order:

1. gradient tolerance;
2. step tolerance;
3. cost tolerance;
4. iteration budget;
5. residual-evaluation budget.

Convergence therefore wins when it occurs on the final allowed evaluation.
After a rejected trial, the accepted state and its already-checked tolerances
are unchanged. The solver increments `iterations`, applies the damping update,
then returns `MAX_ITERATIONS` if that budget is reached before it forms another
proposal or reserves more residual calls. Otherwise it checks whether the next
complete trial reservation fits and returns `MAX_EVALUATIONS` if it does not.
The v0.1 tolerance definitions are:

```text
gradient: ||J^T f||_inf <= gtol
step:     ||step||_2 <= xtol * (xtol + ||x||_2)
cost:     |F_previous - F_current| <= ftol * max(1, F_previous)
```

Here `f` and `J` mean the weighted, robustified residual and Jacobian used by
the LM model. Each tolerance is `Optional[Float64]`; a present value is finite
and positive, while `None` disables that convergence test.

The v0.1 defaults and fixed policy are:

| Setting | Default or rule |
| --- | --- |
| `ftol`, `xtol`, `gtol` | `1e-8` each |
| Maximum iterations | `100` |
| Maximum residual evaluations | `1000` |
| Initial damping | `1e-3` |
| Damping bounds | `[1e-15, 1e15]` |
| Damping scale | `d_j = ||J[:,j]||_2`; conceptually `D[j,j] = d_j^2` |
| Accept trial | actual reduction is positive and ratio is greater than zero |
| Strong trial | ratio greater than `0.75`: divide damping by `3` |
| Weak trial | ratio below `0.25`: multiply damping by `2` |
| Otherwise | retain damping |

A damping update is clamped to its bounds. A rejected trial uses the weak-trial
update. Failure to produce a finite descent proposal at maximum damping is a
numerical failure.

One iteration is one completed LM trial, whether the trial is accepted or
rejected. Counters report completed calls:

- `residual_evaluations` includes the initial call, trial calls, and calls made
  by finite differences;
- `jacobian_evaluations` increments once for each complete analytic or
  finite-difference Jacobian;
- `iterations` increments after a trial residual is evaluated and classified.

Every result includes the initial residual call. Consequently,
`iterations <= residual_evaluations - 1`, and every complete Jacobian evaluation
is paired with an evaluated residual state, so
`jacobian_evaluations <= residual_evaluations`. Construction and revalidation
enforce both relationships.

Before a finite-difference Jacobian, the solver reserves enough evaluation
budget for all parameter columns. If the complete Jacobian does not fit, the
solver returns `MAX_EVALUATIONS` without starting a partial Jacobian.

`LeastSquaresResult.parameters` is the last valid accepted point. `cost` is
`F(x)`, and `optimality` is `||J^T f||_inf`. Tolerance reasons are convergence;
budget reasons preserve a valid iterate without claiming convergence; numerical
failure reports an unusable next step while preserving the last valid iterate.

## Fixed dense QR and LM step

For the weighted, robustified model Jacobian `J` and residual `f`, compute
stable column norms `d_j = ||J[:,j]||_2` and solve

```text
min_s ||[J; sqrt(lambda) diag(d)] s - [-f; 0]||_2
```

with private column-pivoted Householder QR. The implementation never forms
`J^T J`, `lambda * d_j^2`, or an absolute diagonal floor. A zero column is a
private rank breakdown unless an enabled gradient tolerance has already stopped
at the accepted state.

Factor `A P = Q R`. Recompute stable trailing column norms after each reflector;
choose the largest norm and break an exact tie by lowest original column index.
For nonzero active column `x`, use
`alpha = -copysign(||x||_2, x[0])`, treating zero `x[0]` as positive. With

```text
tau = 64 * max(rows, columns) * 2^-52
```

rank is zero when `R[0,0] == 0`; otherwise diagonal `j` is independent exactly
when `abs(R[j,j]) / abs(R[0,0]) > tau`. Equality is deficient. The same pivot,
sign, and relative-rank policy factors final unfloored `J_covariance`.
Triangular solves require

```text
||R s_permuted - (Q^T b)[0:n]||_inf
  <= 128 max(rows, n) epsilon
     * max(1, ||(Q^T b)[0:n]||_inf, ||R||_inf ||s_permuted||_inf)
```

using stable norm and product helpers.

For `g = J^T f`, the trial model uses

```text
predicted_reduction = -(g^T s + 1/2 ||J s||_2^2)
```

and independently checks the equivalent
`1/2 * (||sqrt(lambda) * d * s||_2^2 - g^T s)` identity. The damping term
selects the step but is not part of the user objective.

## Issue sequence

### NERAI-001 — Loss and report semantic foundation

**Outcome:** establish loss equations and nominal result states before solver
control flow depends on them.

**Scope:**

- `LossKind`, `LossEvaluation`, `evaluate_loss()`, and `robust_cost()`;
- `TerminationReason` with convergence, budget, and failure categories;
- `LeastSquaresResult` with last parameters, cost, optimality, counters, and
  termination reason;
- root exports, focused reference/property tests, and one executable example.

**Complete when:** linear, Huber, and soft-L1 reference values and derivatives
pass; Huber is continuous at `z=1`; cost is even in the residual; invalid
numeric inputs are rejected; representable extreme-scale costs and derivatives
remain finite; every termination category and result invariant is tested;
`pixi run check` passes.

**Status:** implemented in the current working tree.

### NERAI-002 — Problem and configuration contracts

**Depends on:** NERAI-001.

**Outcome:** one public, statically dispatched residual-callback contract and a
validated `LeastSquaresOptions` value.

**Scope:** define callback ownership and `raises` behavior; initial parameters;
loss and weights; positive evaluation and iteration budgets; optional positive
`ftol`, `xtol`, and `gtol`; damping bounds; optional finite-difference step.

**Complete when:** the compiler accepts a stateful model callback without FFI or
dynamic dispatch; invalid dimensions, weights, tolerances, damping, and budgets
have focused error tests; an example evaluates a two-parameter problem through
the public contract; `pixi run check` passes.

**Status:** implemented in the current working tree. `ResidualModel` is an
owned, potentially move-only, statically dispatched raising callback. Each
standalone evaluation enforces its validated entry residual count across the
callback. `LeastSquaresProblem` and `LeastSquaresOptions` revalidate all
reachable numeric, shape, weight, and configuration state before residual
evaluation; coherent between-call mutation is explicit reconfiguration.
During a solve, cache and counter mutation must preserve identical residual
values for identical parameters under fixed model configuration.

### NERAI-003 — Private dense numerical kernel

**Depends on:** NERAI-002.

**Outcome:** the minimum private row-major operations needed by LM, with no
public array abstraction.

**Scope:** checked matrix shape/indexing; dot products, stable norms, `J^T f`,
and matrix-vector products; deterministic column-pivoted Householder QR for
rectangular matrices; application of reflectors; checked triangular solves;
one relative numerical-rank policy. Do not form `J^T J`.

**Complete when:** hand-computed square and overdetermined systems match
reference solutions; `A P = Q R`, orthogonality, transpose/product, triangular
residual, scale, permutation, exact-tie, and rank-threshold identities hold on
fixed generated cases; shape mismatches and deficient pivots are explicit
errors; all outputs remain finite for the declared fixture range;
`pixi run check` passes.

### NERAI-004 — Forward finite-difference Jacobian

**Depends on:** NERAI-002 and NERAI-003.

**Outcome:** deterministic numerical Jacobians with exact evaluation accounting.

**Scope:** forward differences of raw residuals; default
`h_j = sqrt(epsilon) * max(1, abs(x_j))`; validated user relative step; a
representable perturbation for every finite parameter; fixed residual shape;
all callback errors, non-finite values, and dimension violations remain raised
model/contract errors in initial, trial, and perturbation phases.

**Complete when:** constant, affine, quadratic, and coupled analytic fixtures
meet their stated absolute/relative tolerances; constant columns are exactly
zero; the callback count is `n + 1` when a base residual is not supplied and
`n` when it is supplied; callback errors, non-finite outputs, and changed shape
are tested; `pixi run check` passes.

### NERAI-005 — Weighted and robust objective model

**Depends on:** NERAI-001, NERAI-003, and NERAI-004.

**Outcome:** one internal transformation produces cost, model residuals, model
Jacobian, and gradient under the release objective.

**Scope:** apply observation weights once; apply loss scaling and derivatives;
protect the robust Jacobian scaling near zero; compute finite cost and
optimality.

**Complete when:** linear loss equals direct weighted least squares; zero-weight
observations make no contribution; Huber and soft-L1 match independently
calculated fixtures; repeated residual sign changes preserve cost; finite-
difference gradients of the scalar objective agree with the model gradient on
fixed cases; `pixi run check` passes.

### NERAI-006 — Levenberg-Marquardt step

**Depends on:** NERAI-003 and NERAI-005.

**Outcome:** compute a checked LM proposal and its predicted reduction without
owning iteration policy.

**Scope:** compute stable positive column scales `d_j = ||J[:,j]||_2`, then
solve the augmented least-squares problem
`min ||[J; sqrt(lambda) diag(d)] step - [-f; 0]||_2` with the private pivoted-QR
kernel. Do not form `J^T J`, `lambda * d_j^2`, or a unit-sensitive absolute
floor. Calculate predicted reduction from `J * step`; classify deficient,
non-finite, and non-descent proposals.

**Complete when:** one-dimensional and linear multi-parameter fixtures match
hand calculations; increasing damping monotonically reduces step norm for the
fixture family; predicted reduction is positive for accepted descent fixtures;
the QR solution satisfies its triangular residual and the equivalent damping
identity; zero columns and breakdowns return an explicit internal status;
`pixi run check` passes.

### NERAI-007 — LM iteration and damping policy

**Depends on:** NERAI-006.

**Outcome:** deterministic accept/reject control flow advances only valid states.

**Scope:** actual/predicted reduction ratio; damping increase/decrease bounds;
accepted-state replacement; rejected-trial preservation; exact iteration and
callback accounting; an explicit `MAX_ITERATIONS` check after every rejected
trial and before another proposal or budget reservation.

**Complete when:** fixtures exercise accepted and rejected trials; rejected
trials preserve parameters and cost; damping remains finite and within bounds;
repeated rejection stops at exactly the iteration budget even when residual
budget remains; no subsequent proposal or callback begins after that stop;
the same input produces identical decisions and counters on repeated runs;
`pixi run check` passes.

### NERAI-008 — Public solve loop and termination

**Depends on:** NERAI-002, NERAI-004, and NERAI-007.

**Outcome:** `least_squares()` returns the explicit report defined above.

**Scope:** initial validation/evaluation; convergence precedence; iteration and
evaluation budgets; last-valid-state handling; uniform propagation of every
callback error/non-finite return; root export and user example.

**Complete when:** each termination reason is reached by a focused fixture;
simultaneous convergence/budget cases follow the documented precedence; no
callback starts after the evaluation budget is exhausted; result counters equal
instrumented callback counters; invalid initial state raises while later
breakdown returns numerical failure; `pixi run check` passes.

### NERAI-009 — Covariance estimate

**Depends on:** NERAI-003 and NERAI-008.

**Outcome:** an opt-in covariance report for a converged, locally full-rank
solution.

**Scope:** inverse local curvature from the final unfloored `J_covariance` QR
factor using checked triangular solves; residual variance scaling with
documented degrees of freedom; explicit unavailable reasons for rank deficiency
or insufficient degrees of freedom; result-level validation of covariance
dimension, termination, and availability status after public mutation.

**Complete when:** linear-regression fixtures match independently calculated
covariance values; output is symmetric within tolerance with non-negative
diagonal; rank-deficient and `m <= n` cases report unavailable instead of
inventing finite values; mutation cannot leave an available covariance with a
non-converged result, a mismatched parameter dimension, or a contradictory
unavailability reason; `pixi run check` passes.

### NERAI-010 — End-to-end numerical corpus

**Depends on:** NERAI-008 and NERAI-009.

**Outcome:** release-level evidence for solver correctness and robustness.

**Scope:** exact affine fit; Rosenbrock residual system; weighted line fit;
fixed nonlinear curve fit; contaminated-data comparison of linear, Huber, and
soft-L1 loss; rank-deficient failure; exact budget accounting.

**Complete when:** every fixture records equation, input, expected value,
tolerance, and provenance; solutions meet the numerical gates below on all CI
platforms; robust fixtures improve parameter error over linear loss for the
committed contaminated dataset; `pixi run check` and `pixi run package` pass.

### NERAI-011 — v0.1 usability and release audit

**Depends on:** NERAI-010.

**Outcome:** a small documented public surface that is independently useful.

**Scope:** API docs; fitting example; benchmark methodology and fixed datasets;
compatibility matrix; changelog; Conda installed-package smoke test.

**Complete when:** every root export is documented and used by a test or
example; README commands work in a clean clone; benchmarks report environment,
warmup, iterations, and metric without superiority claims; package smoke tests
import and solve through only installed artifacts; no v0.2 symbol leaks through
the root; release CI is green.

## Numerical gates

Every algorithm issue carries unit, reference-value, and invariant coverage.
The release corpus additionally satisfies these gates:

| Gate | Requirement |
| --- | --- |
| Finite state | Every successful or budget-limited result field is finite. |
| Affine recovery | Exact full-rank affine fixtures recover parameters to `1e-10` relative error and cost below `1e-20`. |
| Jacobian | Forward differences match analytic fixtures to `1e-6` relative or `1e-8` absolute error. |
| Nonlinear solve | Rosenbrock and curve-fit parameters meet fixture-specific tolerances on every supported target. |
| Robustness | On the committed outlier fixture, Huber and soft-L1 parameter error are each below linear-loss error. |
| Accounting | Instrumented callback counts equal every reported counter exactly, including budget boundaries. |
| Breakdown | Rank loss and internal overflow never produce a converged result or fabricated covariance; non-finite callback returns always raise. |
| Determinism | Repeated runs on one target produce the same termination reason, counters, and accepted-state sequence. |

Tolerance values belong beside each fixture, not in a global permissive helper.
Reference values must be independently generated from documented equations;
external datasets record source, license, checksum, and generation command in
`docs/data-provenance.md`.

## v0.1 non-goals

- bounds, equality constraints, and inequality constraints;
- underdetermined systems and rank-deficient pseudoinverse solutions;
- sparse Jacobians or sparse linear algebra;
- automatic differentiation and symbolic Jacobians;
- local or global general minimizers and an optimizer catalog;
- complex parameters, mixed precision, SIMD-specific APIs, GPU backends, or FFI;
- asynchronous callbacks, cancellation hooks, and user-stop callbacks;
- domain-specific model, dataframe, plotting, interpolation, or file-I/O types.

These are candidates for later roadmaps only after the dense `Float64` solver is
released and exercised by an independent downstream consumer.
