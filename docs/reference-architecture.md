# Nonlinear least-squares reference architecture

This document turns a source review of four established solver projects into
an executable architecture for Nerai. It is subordinate to the v0.1 contract
in [implementation-plan.md](implementation-plan.md): where this review finds a
contract gap, the implementation plan must be amended before the affected
issue begins.

The scope is intentionally narrow: deterministic, unconstrained, dense
`Float64` nonlinear least squares. This research does not authorize general
minimizers, bounds, sparse matrices, automatic differentiation, foreign
runtime dependencies, or source translation.

## Reproducible reference ledger

The repositories below were shallow-cloned for study under
`/Users/ryuichi/dev/reference-libraries/nerai/`. That directory is outside the
Nerai repository and is not a build input. No reference source or fixture is
copied into Nerai.

| Project | Reviewed commit | Upstream license | Useful organization |
| --- | --- | --- | --- |
| [argmin](https://github.com/argmin-rs/argmin/tree/b3002851e70ee25ec03b626b2661f1c5bf033fae) | `b3002851e70ee25ec03b626b2661f1c5bf033fae` | MIT OR Apache-2.0 | Operation traits, owned problem wrapper, generic iteration state, result and termination orchestration |
| [rust-cv/levenberg-marquardt](https://github.com/rust-cv/levenberg-marquardt/tree/e67066cf1351e91e054c30fcb2ff49d005db1e50) | `e67066cf1351e91e054c30fcb2ff49d005db1e50` | MIT | Stateful least-squares problem, focused LM state, pivoted-QR trust subproblem, MINPACK-derived reference tests |
| [Ceres Solver](https://github.com/ceres-solver/ceres-solver/tree/fe351d5ab8cbc574f33456376bf5aa90d3162d5d) | `fe351d5ab8cbc574f33456376bf5aa90d3162d5d` | BSD-3-Clause for Ceres; its `LICENSE` also carries notices for bundled Apache-2.0 and MIT material | Cost/loss/problem contracts, trust-region strategy, accepted/trial state, summary counters and covariance policy |
| [SciPy](https://github.com/scipy/scipy/tree/a2c4d68b3cab98a0e5773ea0016e7c4109da6511) | `a2c4d68b3cab98a0e5773ea0016e7c4109da6511` | BSD-3-Clause; bundled components have separate notices | `least_squares`, robust residual/Jacobian scaling, trust-radius updates, `curve_fit` weighting/covariance and numerical fixtures |

The commits were resolved from each upstream default branch on 2026-08-20 and
verified with `git rev-parse HEAD` and
`git rev-parse --is-shallow-repository`. Primary files reviewed include:

- argmin's
  [`Problem` and operation traits](https://github.com/argmin-rs/argmin/blob/b3002851e70ee25ec03b626b2661f1c5bf033fae/crates/argmin/src/core/problem.rs),
  [`IterState`](https://github.com/argmin-rs/argmin/blob/b3002851e70ee25ec03b626b2661f1c5bf033fae/crates/argmin/src/core/state/iterstate.rs),
  [termination](https://github.com/argmin-rs/argmin/blob/b3002851e70ee25ec03b626b2661f1c5bf033fae/crates/argmin/src/core/termination.rs),
  and [Gauss-Newton solver](https://github.com/argmin-rs/argmin/blob/b3002851e70ee25ec03b626b2661f1c5bf033fae/crates/argmin/src/solver/gaussnewton/gaussnewton_method.rs);
- rust-cv's
  [problem trait](https://github.com/rust-cv/levenberg-marquardt/blob/e67066cf1351e91e054c30fcb2ff49d005db1e50/src/problem.rs),
  [LM state and loop](https://github.com/rust-cv/levenberg-marquardt/blob/e67066cf1351e91e054c30fcb2ff49d005db1e50/src/lm.rs),
  [trust subproblem](https://github.com/rust-cv/levenberg-marquardt/blob/e67066cf1351e91e054c30fcb2ff49d005db1e50/src/trust_region.rs),
  and [numerical derivative checker](https://github.com/rust-cv/levenberg-marquardt/blob/e67066cf1351e91e054c30fcb2ff49d005db1e50/src/utils.rs);
- Ceres'
  [`CostFunction`](https://github.com/ceres-solver/ceres-solver/blob/fe351d5ab8cbc574f33456376bf5aa90d3162d5d/include/ceres/cost_function.h),
  [`LossFunction`](https://github.com/ceres-solver/ceres-solver/blob/fe351d5ab8cbc574f33456376bf5aa90d3162d5d/include/ceres/loss_function.h),
  [`Solver::Options` and `Summary`](https://github.com/ceres-solver/ceres-solver/blob/fe351d5ab8cbc574f33456376bf5aa90d3162d5d/include/ceres/solver.h),
  [LM strategy](https://github.com/ceres-solver/ceres-solver/blob/fe351d5ab8cbc574f33456376bf5aa90d3162d5d/internal/ceres/levenberg_marquardt_strategy.cc),
  [trust-region loop](https://github.com/ceres-solver/ceres-solver/blob/fe351d5ab8cbc574f33456376bf5aa90d3162d5d/internal/ceres/trust_region_minimizer.cc),
  and [covariance contract](https://github.com/ceres-solver/ceres-solver/blob/fe351d5ab8cbc574f33456376bf5aa90d3162d5d/include/ceres/covariance.h);
- SciPy's
  [`least_squares`](https://github.com/scipy/scipy/blob/a2c4d68b3cab98a0e5773ea0016e7c4109da6511/scipy/optimize/_lsq/least_squares.py),
  [shared trust-region and robust-scaling helpers](https://github.com/scipy/scipy/blob/a2c4d68b3cab98a0e5773ea0016e7c4109da6511/scipy/optimize/_lsq/common.py),
  [TRF loop](https://github.com/scipy/scipy/blob/a2c4d68b3cab98a0e5773ea0016e7c4109da6511/scipy/optimize/_lsq/trf.py),
  [`curve_fit`](https://github.com/scipy/scipy/blob/a2c4d68b3cab98a0e5773ea0016e7c4109da6511/scipy/optimize/_minpack_py.py),
  and [least-squares tests](https://github.com/scipy/scipy/blob/a2c4d68b3cab98a0e5773ea0016e7c4109da6511/scipy/optimize/tests/test_least_squares.py).

These links are provenance, not implementation dependencies. Reference-value
fixtures committed later must state their equation, independently generated
expected values, tolerance, generation command, and license in
`docs/data-provenance.md`.

## What the references establish

### Argmin

Argmin separates residual-like `Operator` calls from `Jacobian` calls and
counts operations in a problem wrapper. Its executor owns a very broad
`IterState` containing current, previous, and best values plus generic
derivative slots and string-keyed counters. Its Gauss-Newton implementation
forms an explicit inverse of `J^T J` and reports the residual norm as cost.

Nerai adopts static operation boundaries, central accounting, and explicit
current/previous separation. It rejects the universal executor/state, generic
counter map, arbitrary string termination, explicit normal-matrix inversion,
and expansion into a general optimizer framework.

### rust-cv/levenberg-marquardt

This crate exposes a focused problem with mutable current parameters and
separate residual/Jacobian calls. Its solver owns the problem, keeps accepted
and temporary parameters, and restores the accepted model parameters before
returning from a rejected termination. Ordinary rejected trials keep the
accepted numerical `x` separate; the next trial overwrites the model's temporary
parameters. Numerically it uses pivoted QR, scale-aware diagonals, a trust
radius, and an iterated damping solve rather than explicitly inverting `J^T J`.

Nerai adopts a private, least-squares-specific solver state; accepted/trial
separation; dimension checks at each callback boundary; scale-aware damping;
and isolated step tests. It rejects model-owned current parameters, mandatory
analytic Jacobians, `Option`-only callback failures, panic-based option
validation, its residual-only evaluation count, and returning the model as part
of the solver result.

### Ceres Solver

Ceres combines required residual output and optionally requested Jacobian output
in `CostFunction`, organizes residual/parameter blocks in `Problem`, applies
loss functions using `rho`, `rho'`, and `rho''`, and records named
accepted/rejected and evaluation counters. Its trust-region minimizer evaluates
candidates without replacing the accepted state, and its LM strategy treats
radius as inverse damping.
Covariance assumes appropriately whitened residuals and reports limitations for
rank-deficient problems.

Nerai adopts candidate-state isolation, named counters, the three-value loss
convention, explicit covariance assumptions, and a private step-policy
boundary. It rejects block graphs, pointer ownership policies, manifolds,
bounds, sparse solvers, callbacks, logging, threading, plugin-like public
strategies, and Ceres' much larger surface.

### SciPy

SciPy makes Jacobian selection explicit, keeps accepted state separate in its
trust-region loops, and implements robust loss by scaling residual and Jacobian
models while retaining raw residuals in the public result. `curve_fit` gives a
useful distinction between known observation uncertainty and residual-variance
scaled covariance. Its `method="lm"` delegates to MINPACK, requires `m >= n`,
and does not support bounds or nonlinear loss; the TRF code is a state-machine
and robust-transform reference here, not an LM implementation model.

Nerai adopts independently testable robust model transforms, explicit
finite-difference policy, trust-ratio boundary fixtures, and explicit
covariance scaling. It rejects the giant keyword surface, bounds and sparse
branches, silent algorithm switching, pseudoinverse covariance in v0.1, and
SciPy's accounting convention that excludes finite-difference residual calls
from `nfev`.

## Public contracts

### Residual model

Keep the implemented contract:

```mojo
trait ResidualModel(Deinitable, Movable):
    def residual_count(self) -> Int: ...
    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]: ...
```

The explicit parameter argument is preferable to a `set_params()`/`residuals()`
pair: a rejected candidate never becomes hidden mutable model state. `mut self`
still permits counters and caches. `LeastSquaresProblem[M]` owns `M`, initial
parameters, weights, options, and the captured residual count. A solve takes a
mutable borrow of the problem so the caller retains its stateful model after
return:

```mojo
def least_squares[M: ResidualModel](
    mut problem: LeastSquaresProblem[M]
) raises -> LeastSquaresResult: ...
```

That is the complete v0.1 solve surface. Do not add an `Executor`, public
solver-state object, callback registry, runtime-selected solver, or public
matrix family.

At solve entry, validate options, finite parameters and weights, fixed `n >= 1`,
fixed `m >= n`, and `model.residual_count() == m`. Capture `m` and `n` for the
entire solve. Each callback must recheck the model declaration, returned
length, and finite values before numerical code receives them.

### Jacobian providers

The provider boundary is private in v0.1. The default
`_ForwardDifferenceProvider` consumes an already evaluated base residual and
produces the raw-residual Jacobian in private row-major storage. It owns no
model; it mutably borrows the problem for each perturbation and updates a
single `_EvaluationCounters` value.

The callback convention is nevertheless frozen for a future analytic provider:
an analytic Jacobian differentiates the raw residual, not a weighted or
robustified residual. A future static refinement can look conceptually like:

```mojo
trait AnalyticResidualModel(ResidualModel):
    # The final Mojo-native output-buffer shape is deliberately not frozen yet.
    def raw_jacobian(...): ...
```

Do not export this trait in v0.1. First prove a checked Mojo-native borrowed
output buffer that does not expose Nerai's private dense type. Do not use a
runtime enum containing generic providers, do not fall back from a failed
analytic callback to finite differences, and do not count derivative-checking
calls as anything other than residual evaluations.

The finite-difference option is a relative step. Its default is
`sqrt(epsilon)`, where binary64 `epsilon = 2^-52`. For parameter `x_j`, construct
one deterministic perturbation as follows:

1. Let `scale = max(1, abs(x_j))`. Compute the positive nominal magnitude
   `h = relative_step * scale` with a checked, saturating product; cap an
   overflowing mathematical product at the largest finite `Float64`, and
   replace a zero underflow with the smallest positive subnormal.
2. Try `x_j + h` first. If it is finite and distinct, use it. Otherwise try
   `x_j - h`.
3. If neither is finite and distinct, double `h` with saturation and repeat.
   Binary64 requires at most 1075 magnitudes before a finite input has a finite
   distinct neighbor in one direction. Failure after that bound is an internal
   numerical breakdown.
4. Divide the residual difference by the actual signed denominator
   `x_perturbed - x_j`, never the nominal `h`.

This rule fixes direction and rounding behavior for maximum finite values,
subnormal option values, and `x_j + h == x_j`. A complete `m * n` Jacobian
increments `jacobian_evaluations` exactly once; partial work never does.

### Options

Retain `LeastSquaresOptions` as a validated value. `None` disables a tolerance;
present tolerances and scales are finite and positive; budgets are positive;
damping bounds satisfy `min <= initial <= max`. Invalid configuration raises
before any model callback. Mutable public fields are revalidated at solve
entry. At least one of `ftol`, `xtol`, or `gtol` must be enabled for a solve;
an all-disabled options value may still be constructed and used for standalone
problem evaluation, but `least_squares()` rejects it before the first callback.

Algorithm policy remains fixed in v0.1 rather than becoming options: forward
difference, damped normal-equation step, ratio thresholds, termination order,
and dense `Float64`. This makes identical inputs reproducible and keeps the
public configuration from outrunning tested implementation choices.

### Result and termination

Retain `LeastSquaresResult` as the last valid accepted point:

```text
parameters
cost
optimality = ||J^T f||_inf at those parameters
iterations
residual_evaluations
jacobian_evaluations
termination
```

All numeric fields are finite. `cost` and `optimality` correspond to the same
accepted parameters; a stale gradient from an earlier state is never reported.
Tolerance reasons are success. Budget reasons preserve a usable accepted point
without claiming convergence. `NUMERICAL_FAILURE` means the attempted next
numerical operation was unusable, not that the returned point contains invalid
values.

Do not reduce termination to `success: Bool`, and do not add arbitrary strings
to the status enum. Human-readable formatting can be derived later without
weakening exhaustive matching.

## Private solver state and data flow

Use narrow private values rather than one universal state:

```text
_AcceptedState
  parameters
  raw_residuals
  raw_jacobian when covariance is requested
  model_residuals
  model_jacobian
  gradient
  cost
  optimality

_TrialState
  parameters
  raw_residuals
  cost
  step_norm
  predicted_reduction
  actual_reduction
  ratio

_LMState
  accepted
  damping
  completed_iterations
  counters

_EvaluationCounters
  residual_evaluations
  jacobian_evaluations
```

The flow for one solve is:

```text
validate and capture m/n
  -> evaluate initial raw residuals
  -> compute complete raw Jacobian
  -> apply weights/loss to build accepted model
  -> check initial gradient convergence
  -> form damped step from accepted model
  -> check enabled step tolerance and positive predicted reduction
  -> evaluate isolated trial residuals
  -> compute and classify actual/predicted reduction
     -> reject: preserve accepted state, increase/retain damping
     -> accept: compute complete Jacobian and model at trial,
                then atomically replace accepted state
  -> check tolerances and budgets in documented order
```

No pointer or shared reference from `_TrialState` aliases `_AcceptedState`.
Copying the small parameter vector is preferable to rollback-sensitive model
mutation. A rejected or non-finite trial must not alter result parameters,
cost, gradient, or Jacobian.

### Evaluation accounting and reservation

Counters count completed user calls, including finite-difference calls. A
residual call that returns a vector has completed even if validation then finds
non-finite data or the wrong shape. A callback that raises propagates its
`Error`; because no result is returned, the solve aborts and no report needs to
expose a partial count. `jacobian_evaluations` increments only after all
columns and validations complete. `iterations` increments once a trial
residual has returned and the trial is classified, accepted or rejected.

During the initial Jacobian, a perturbation callback error or wrong shape raises
as a callback-contract failure; a non-finite perturbation return raises because
no complete accepted state exists yet. During the Jacobian for a promising
trial, callback errors and wrong shapes still raise, while a non-finite
perturbation return yields `NUMERICAL_FAILURE` and preserves the previous
accepted state. In every case a returned perturbation vector increments
`residual_evaluations`, and an incomplete Jacobian does not increment
`jacobian_evaluations`.

The default finite-difference solver must not create an accepted point without
a matching gradient. Therefore:

- before the initial callback, require enough remaining residual budget for
  the initial residual plus its full `n`-column Jacobian;
- before every trial callback, reserve `1 + n` residual calls: one for the
  trial and enough to differentiate it if accepted;
- a rejected trial consumes only its one completed call; unused reservation is
  released;
- if the reservation does not fit, return `MAX_EVALUATIONS` at the current
  accepted state without starting the trial.

This is intentionally conservative. It resolves the otherwise impossible
case where a final-budget trial is accepted but `LeastSquaresResult.optimality`
cannot be evaluated at its parameters. No partial Jacobian may begin.
For the v0.1 forward-difference entry point,
`max_residual_evaluations < 1 + n` is therefore an invalid configuration and
raises before the first callback; without an initial Jacobian, the solver
cannot construct the valid accepted state required by `LeastSquaresResult`.

Named counters are preferable to argmin's string map. Accepted/rejected counts,
linear-solve counts, final damping, and trust radius are useful internal test
diagnostics, but they are not part of the v0.1 public result unless an
implementation issue demonstrates a stable user need.

## LM step, damping, and trial policy

The reference projects show that pivoted QR and trust-radius LM are stronger
long-term numerical designs than explicit normal-equation inversion. Nerai
v0.1 nevertheless keeps its already scoped step:

```text
(J^T J + lambda D) step = -J^T f
D[j,j] = max((J^T J)[j,j], 1e-15)
```

Let `B = J^T J`, `g = J^T f`, and let the checked solve return `s`. The trial and
reduction equations are fixed as:

```text
x_trial             = x + s
actual_reduction    = F(x) - F(x_trial)
predicted_reduction = -(g^T s + 1/2 s^T B s)
ratio               = actual_reduction / predicted_reduction
```

The prediction is the reduction in the undamped local least-squares model; the
damping term chooses the step but is not counted as objective value. For an
accurate linear solve, `predicted_reduction` also equals
`1/2 s^T (lambda D s - g)`; tests use that identity as a solve check, not as a
second definition. Dot products use the stable private accumulator.

The private kernel factors and solves the damped symmetric system; it never
computes `inverse(J^T J)`. To reduce scale sensitivity in an already formed
positive-diagonal matrix `A`, it symmetrically equilibrates it:

```text
scale_j = sqrt(A[j,j])
C[i,j]  = (A[i,j] / scale_i) / scale_j
tau     = 64 * max(1, n) * 2^-52
```

It solves the correlation-scale system `C y = b / scale` and returns
`x_j = y_j / scale_j`. A non-finite or non-positive diagonal is a breakdown.
Cholesky uses symmetric diagonal pivoting: at each column choose the largest
remaining Schur-complement diagonal, break an exact tie by the lowest original
column index, swap both matrix axes and the right-hand side, and unpermute the
solution. During factorization, `p < -tau` is indefinite and
`-tau <= p <= tau` is singular at the v0.1 precision contract. The same
pivoting, equilibration, and `tau` define full column rank for covariance.
Sequential division avoids overflowing `scale_i * scale_j`; every factor and
solution entry must remain finite. The checked solution also requires

```text
||C y - b_scaled||_inf
    <= 128 n epsilon * max(1, ||b_scaled||_inf, ||C||_inf ||y||_inf)
```

using stable norm/product helpers. Otherwise the solve is a numerical
breakdown rather than a usable step.

Equilibration makes the factor decision insensitive to exact positive diagonal
congruence of the supplied matrix when no intermediate underflows or overflows.
It does not make the complete LM proposal invariant to parameter units: forming
`B + lambda D` uses the absolute `1e-15` damping floor, so a rescaling that
crosses that floor can change the proposal. Kernel rescaling tests stay above
the floor; separate solver tests cover crossings explicitly.

The rank label means numerical rank under this pivot policy, threshold, and
input column order. Symmetric pivoting reduces order sensitivity but cannot
make a threshold-boundary matrix permutation invariant under floating-point
rounding; an exact diagonal tie uses the rule above. Permutation fixtures must
agree away from `tau` and explicitly record allowed boundary sensitivity.

Before evaluating the trial, check an enabled `xtol` against `||s||` using the
documented step equation; success returns `STEP_TOLERANCE` at the current
accepted state without consuming an iteration. Then require a finite positive
prediction. A nonpositive prediction increases damping and retries without a
callback or iteration. If no positive proposal exists at maximum damping,
return `NUMERICAL_FAILURE`. In particular, when `gtol` and `xtol` are disabled,
an exact stationary step cannot silently claim convergence; it follows this
no-progress failure path.

For a finite positive prediction, evaluate `F(x_trial)`. A trial is accepted
only when actual reduction and ratio are positive. Ratio above `0.75` divides
damping by three; ratio below `0.25` doubles it; other accepted ratios retain
damping; rejected trials use the weak update. Updates clamp to the documented
finite bounds without overflowing. The normal post-acceptance termination
order remains gradient, step, cost, iteration budget, then evaluation budget.

Do not expose a trust radius in v0.1: the current policy is parameterized by
damping. Use the rust-cv and Ceres trust-region implementations as adversarial
references for tests of scale, rank, rejection, and predicted reduction, not
as code to translate. Pivoted QR is the first candidate replacement after the
dense v0.1 contract is proven; it is not an excuse to expand into sparse or
general trust-region solvers.

## Weights and robust losses

For raw residual `r_i`, observation multiplier `w_i`, and loss scale `C`, keep
the implemented objective:

```text
q_i = w_i r_i
z_i = (q_i / C)^2
F   = 1/2 sum_i C^2 rho(z_i)
```

Weights multiply both residuals and raw Jacobian rows exactly once. A zero
weight excludes the observation. The robust transform owns cost, model
residuals, model Jacobian, gradient, and numerical safeguards. User callbacks
never apply weights or loss themselves.

For one row, let `rho1 = rho'(z)` and
`A = rho'(z) + 2 z rho''(z)`. Use the fixed curvature floor
`A_floor = 2^-52`, then construct:

```text
sqrt_A       = sqrt(max(A, A_floor))
f_model[i]   = q_i * rho1 / sqrt_A
J_model[i,j] = J_raw[i,j] * (w_i * sqrt_A)
```

This preserves the exact first-order model identity
`J_model^T f_model = dF/dx` before floating-point rounding. The displayed
multiplication order applies the bounded `sqrt_A` to the weight before the raw
Jacobian entry. Cost is computed separately by the stable scalar `robust_cost`;
`1/2 ||f_model||^2` is not the robust objective. Preserve raw residuals
separately and never feed an already robustified residual into this transform
again.

The transform must not form an overflowing `z` merely to obtain derivatives.
For `a = abs(q_i)`, compute `rho1` and `A` with these piecewise stable formulas:

| Loss and branch | `rho1` | `A` before floor |
| --- | --- | --- |
| linear | `1` | `1` |
| Huber, `a <= C` | `1` | `1` |
| Huber, `a > C` | `C / a` | `0` |
| soft-L1, `a <= C`, `t = a/C` | `1 / sqrt(1 + t^2)` | `rho1^3` |
| soft-L1, `a > C`, `u = C/a` | `u / sqrt(1 + u^2)` | `rho1^3` |

Every squared ratio in this table is at most one. The implementation handles
`a == 0` through the first branch. Any
unrepresentable weighted residual, model row, model residual, gradient, normal
entry, or cost raises at the initial state and becomes `NUMERICAL_FAILURE`
after a complete accepted state exists.

`A_floor` stabilizes the solver model only. When covariance is requested,
retain the final raw Jacobian and construct a separate statistical-curvature
row:

```text
J_covariance[i,j] = J_raw[i,j] * (w_i * sqrt(max(A, 0)))
```

This path has no curvature floor. A Huber outlier has `A = 0` and contributes
no rank; an underflowed soft-L1 curvature also contributes a zero row. The
covariance rank decision therefore cannot be manufactured by the solver's
regularization floor.

For each built-in loss, test the scalar cost and first two derivatives
independently before testing this model transform. The floor must keep `f` and
`J` finite while the model gradient agrees with a finite difference of the
scalar objective within the fixture tolerance.

Required cross-layer invariants are:

- linear loss exactly reduces to ordinary weighted least squares and ignores
  `C`;
- zero-weight rows contribute zero cost, gradient, and normal matrix;
- flipping one residual's sign preserves its cost and flips its gradient
  contribution;
- finite differences of the scalar robust objective agree with `J^T f`;
- extreme but representable cost cases do not fail because an intermediate
  square overflows.

## Covariance contract

Covariance is opt-in and never implied by convergence. For v0.1 it is a local
curvature approximation formed from the separate, unfloored final
`J_covariance` above. It is not a general robust sandwich estimator and must be
documented as approximate when a nonlinear loss is active.

Observation weights are multipliers, not variances. For relative standard
deviations `sigma_i`, callers use weights proportional to `1 / sigma_i`; the
v0.1 report then estimates one common residual-variance scale. Known absolute
uncertainty would omit that scale and is not exposed in v0.1. Correlated
observation covariance requires caller-side whitening and is also out of scope.

Let `m_eff` be the count of strictly positive weights. Residual-variance scaling
has degrees of freedom `m_eff - n`, not `m - n`, because zero-weight
observations are excluded. When uncertainty is relative rather than known,
scale the local inverse normal matrix by `2F / (m_eff - n)`. If known absolute
uncertainty is introduced later, expose the absolute-versus-relative choice
explicitly rather than inferring it from weight values.

Covariance is unavailable, with a specific reason, when:

- `m_eff <= n`;
- the final model Jacobian is not numerically full column rank under the
  equilibrated `tau` pivot rule above;
- any factorization, solve, variance scale, or output entry is non-finite;
- termination is not `GRADIENT_TOLERANCE`, `STEP_TOLERANCE`, or
  `COST_TOLERANCE`.

Compute inverse-normal columns through checked solves. Let `cmax` be the largest
absolute output entry; require pairwise asymmetry no greater than
`128 n epsilon * max(1, cmax)`, then replace each off-diagonal pair by its
average. Larger asymmetry makes covariance unavailable as non-finite/unstable
arithmetic. Do not use a pseudoinverse in v0.1, return infinities, or call a
rank-deficient matrix a covariance.

## Ownership and error taxonomy

The problem owns its concrete model, but `least_squares(mut problem)` only
borrows it for the solve. This keeps model counters/caches inspectable and
avoids rust-cv's need to return the model. Private matrices and states own their
storage; no result borrows solver scratch memory.

The public boundary distinguishes three outcomes:

1. **Raise `Error`:** invalid entry configuration; model-raised error at any
   call; changed model declaration/returned shape; or another violated public
   callback contract. These are not convergence statuses.
2. **Return convergence or budget termination:** the reported accepted state
   and its cost/gradient are valid and mutually consistent.
3. **Return `NUMERICAL_FAILURE`:** a complete accepted state exists, but a later
   trial, candidate Jacobian/model transform, factorization, proposal, or
   damping progression cannot continue safely. The last accepted state remains
   valid.

To preserve this distinction, the solver must call the model and validate its
returned value in separate internal steps. Do not catch and erase a
model-originated `Error` into `NUMERICAL_FAILURE`. A non-finite initial return
raises; a later non-finite trial return is numerical failure after its completed
call is counted. A changed dimension is always a callback-contract `Error`.
The provider-specific initial/candidate Jacobian rules above apply the same
last-complete-state boundary to perturbation returns.

## Minimal package surface

Through NERAI-008, the v0.1 root remains limited to:

```mojo
from nerai import (
    LeastSquaresOptions,
    LeastSquaresProblem,
    LeastSquaresResult,
    LossEvaluation,
    LossKind,
    ResidualModel,
    TerminationReason,
    evaluate_loss,
    least_squares,
    robust_cost,
)
```

`least_squares` is added only in NERAI-008. NERAI-009 then adds
`CovarianceEstimate`, `CovarianceReport`, and `CovarianceUnavailableReason` to
the root, adds `estimate_covariance: Bool = False` to
`LeastSquaresOptions`, and adds one `covariance: CovarianceReport` field to
`LeastSquaresResult`.

`CovarianceReport` has exactly three states: not requested, available, or
unavailable with one reason. Reasons are insufficient degrees of freedom,
rank deficiency, unstable/non-finite covariance arithmetic, and non-converged
termination, represented by the nominal constants
`INSUFFICIENT_DEGREES_OF_FREEDOM`, `RANK_DEFICIENT`,
`UNSTABLE_ARITHMETIC`, and `NONCONVERGED_TERMINATION`.
`CovarianceUnavailableReason` uses a total two-`Bool` nominal representation,
so every reachable mutation still denotes one of those four reasons.

Mojo 1.0 cannot hide stored struct fields. `CovarianceEstimate` therefore
acknowledges public, mutable `values: List[Float64]`, `parameter_count: Int`, and
`residual_variance: Float64` fields. Its `validate()` checks positive dimension,
exactly a checked `n*n` count of finite entries, finite non-negative variance,
symmetry tolerance, and non-negative diagonal. Its checked
`value(row, column)` revalidates before every read. This is a problem-specific
snapshot, not a universal matrix.

`CovarianceReport` uses public
`estimate_value: Optional[CovarianceEstimate]` and
`unavailable_value: Optional[CovarianceUnavailableReason]`. Neither present is
not requested; only an estimate present is available; only a reason present is
unavailable; both present is invalid. `validate()`, `is_requested()`,
`is_available()`, `estimate()`, and `unavailable_reason()` all raise and
revalidate at operation entry. `LeastSquaresResult.validate()` also validates
the report. Coherent public mutation is supported; invalid combinations or
invalid nested snapshots are rejected by every library operation that consumes
them.

Covariance is computed from the retained final raw Jacobian and unfloored
curvature factors only when requested. It never reevaluates the user's model
after termination.
Internal matrix, provider, state, damping, factorization, and robust-transform
values do not leak through `__init__.mojo`.

Do not expose BFGS/L-BFGS placeholders, a universal array, a solver class with
mutable lifetime, a public backend trait, or unimplemented analytic Jacobian
symbols in v0.1.

## Test architecture

Each numerical layer needs unit, independent reference-value, and invariant
tests. Fixed cases use deterministic source checked into the test; randomized
property families use a committed deterministic generator or seed and print
the failing case.

### Contracts and providers

- zero parameters, `m < n`, zero/all-negative/non-finite weights, mutated
  options, and residual-count changes;
- all tolerances disabled at solve entry, proving rejection before a callback;
- stateful/move-only model operation and propagation of its exact error;
- affine, quadratic, coupled, constant-column, extreme-parameter, and
  representable-next-value finite differences;
- maximum finite parameters, subnormal relative steps, preferred positive and
  fallback negative perturbations, and the actual signed denominator;
- exact residual/Jacobian counts with and without a supplied base residual;
- budget one short of a full initial or accepted-state Jacobian, proving no
  partial callback begins;
- later NaN/Inf trial versus later wrong-size trial, proving failure versus
  raised contract error.

### Dense kernel and LM step

- hand-computed `1x1`, `2x2`, and `3x3` products and solves;
- scaled diagonal, zero column, near-rank-loss, ill-conditioned and singular
  cases;
- diagonal-rescaling equivalents proving the equilibrated pivot decision is
  stable for the supplied kernel matrix, plus parameter permutations and cases
  immediately around `tau`;
- solver-level rescalings that stay above and cross the absolute damping floor,
  documenting where unit invariance does and does not hold;
- two-sided solve residual checks instead of comparison only to parameters;
- monotonic step-norm reduction as damping increases;
- direct and damping-identity predicted-reduction agreement, finite ratio
  boundaries at `0`, `0.25`, and `0.75`, damping clamps, enabled pre-trial
  `xtol`, disabled no-progress tolerance, and maximum-damping failure;
- rejected trial preserves every accepted-state field, not only parameters.

### Objective, loop, and covariance

- linear/Huber/soft-L1 values plus scalar-objective gradient checks;
- weighted line fit and zero-weight equivalence to deleting an observation;
- exact affine, Rosenbrock residual, fixed curve fit, contaminated robust fit,
  rank-deficient, and repeated-rejection problems;
- every termination reason and simultaneous tolerance/budget precedence;
- instrumented callback counts equal the result exactly;
- covariance against hand-calculated linear regression, known weighting
  equivalence, relative variance scaling, zero-weight effective degrees of
  freedom, all-Huber-outlier zero-rank behavior, parameter permutations,
  symmetry, non-negative diagonal, public mutation revalidation, and every
  unavailable reason.

The rust-cv MINPACK-derived cases are a guide to problem selection only. Any
Nerai fixture derived from an external dataset needs explicit license and
provenance; independently regenerated values are preferred.

## Benchmark architecture

Benchmarks report methodology, not superiority. Record Mojo/compiler version,
commit, CPU, OS, build flags, warmup, iterations, fixture checksum, dimensions,
termination, accepted/rejected counts, and residual/Jacobian counts next to
wall time.

The first benchmark matrix is deliberately small:

| Dimension | Jacobian source | Residual cost | Trial behavior |
| --- | --- | --- | --- |
| `m=32, n=4` | forward difference | cheap affine | mostly accepted |
| `m=256, n=8` | forward difference | nonlinear | mixed acceptance |
| `m=4096, n=8` | forward difference | expensive synthetic callback | mostly accepted |
| fixed small rank-loss case | forward difference | cheap | repeated rejection/failure |

Measure the private dense kernel separately by `(m, n)` so callback cost does
not hide regressions. Analytic-versus-finite-difference comparison waits until
an analytic public contract exists. SIMD, GPU, sparse, mixed precision, and
cross-language headline comparisons are out of scope.

## Adopted and rejected design decisions

| Decision | Status | Reason |
| --- | --- | --- |
| Static, raising residual trait | Adopt | Small Mojo-native boundary; preserves model errors and stateful caches |
| Explicit parameter argument | Adopt | Trial evaluation does not mutate hidden accepted parameters |
| Private least-squares-specific state | Adopt | Easier invariants than a universal executor/state |
| Separate accepted and trial values | Adopt | Rejection and failure cannot corrupt the result |
| Named exact counters | Adopt | Budget and performance behavior remain auditable |
| Three-value robust loss transform | Adopt | Matches objective gradient/model construction needs |
| Private forward-difference provider | Adopt for v0.1 | Meets release scope without freezing a public matrix buffer |
| Analytic provider refinement | Defer | Raw-Jacobian convention is fixed, but Mojo output-buffer API is not proven |
| Damped checked normal-equation solve | Adopt for v0.1 only | Existing bounded scope; never explicitly invert; test aggressively |
| Pivoted QR trust subproblem | Defer, preferred next kernel | Better scale/rank behavior, but larger than current v0.1 step issue |
| General executor/minimizer framework | Reject | Expands scope before least-squares behavior is stable |
| Runtime solver/provider enums | Reject | Complicate static ownership and invite unsupported combinations |
| Callback failure fallback | Reject | Hides user errors and changes counts/results silently |
| Pseudoinverse covariance | Reject for v0.1 | Rank-deficient uncertainty requires a larger statistical contract |
| Bounds, blocks, sparse matrices, manifolds | Reject for v0.1 | Ceres/SciPy features outside Nerai's initial mission |

## Executable issue order

Proceed only when the previous gate is green:

1. **NERAI-001 and NERAI-002 — completed foundations.** Preserve the objective,
   result, options, ownership, mutation revalidation, and callback contract.
2. **NERAI-003 — private dense kernel.** Add checked row-major storage,
   accumulations, norms, products, damped symmetric factor/solve, pivot/rank
   policy with diagonal equilibration, symmetric diagonal pivoting, and the
   fixed `tau`, plus isolated numerical tests. No public matrix export.
3. **NERAI-004 — provider boundary and forward differences.** Implement private
   provider/counters, accepted base-residual reuse, the bounded
   positive-then-negative perturbation search, shape/finite phase rules,
   full-Jacobian atomic accounting, and conservative evaluation reservation.
   Do not export the future analytic trait.
4. **NERAI-005 — weights and robust model.** Produce cost, model residual,
   model Jacobian, gradient, and optimality with the fixed `A_floor` formulas in
   one internal operation; prove scalar-objective gradient agreement and
   extreme-value behavior.
5. **NERAI-006 — one checked LM proposal.** Solve the damped system, calculate
   the exact undamped-model predicted reduction above, cross-check the damping
   identity, and return typed private breakdown without iteration policy or
   callback ownership.
6. **NERAI-007 — accepted/trial state machine.** Add ratio/damping updates,
   atomic accept, rejection preservation, trial classification, exact counters,
   and non-finite-trial handling.
7. **NERAI-008 — public solve.** Add the single root solve function, solve-entry
   dimension/tolerance capture, initial/final provider reservation, pre-trial
   step handling, termination precedence, callback-error preservation, and
   last-valid-state result.
8. **NERAI-009 — covariance.** Add the option, mutation-revalidating
   estimate/report/reason values, unfloored statistical-curvature Jacobian,
   effective positive-weight degrees of freedom, allowed termination set,
   checked solves, and explicit approximate robust-loss semantics.
9. **NERAI-010 — end-to-end corpus.** Independently generate and document every
   fixture; run the full success, robustness, rank, budget, and determinism
   gates on all CI targets.
10. **NERAI-011 — release audit.** Add API/example/benchmark/package smoke
    evidence, prove every root export, and confirm no deferred API leaked.

Before NERAI-003 begins, update `implementation-plan.md` with the equilibrated
pivot rule. Before NERAI-004, add the exact perturbation, phase-specific failure,
and accepted-state Jacobian reservation rules. Before NERAI-005, add the stable
robust-model formulas and floor. Before NERAI-006/007, add the trial/reduction
equations and pre-trial no-progress policy. Before NERAI-008, require an enabled
tolerance at solve entry. Before NERAI-009, define the covariance API,
operation-time mutation validation, unfloored statistical Jacobian, termination
set, and positive-weight `m_eff` degrees of freedom. These are correctness
requirements discovered by this review, not optional enhancements.
