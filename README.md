# Nerai

> **Experimental — API not yet released.**

Optimization and nonlinear least squares for Mojo.

## Scope

Nerai begins with explicit, inspectable nonlinear least-squares contracts
instead of attempting a broad SciPy.optimize clone.

The implemented core is intentionally narrow: dense `Float64`
Levenberg-Marquardt, forward or central finite-difference Jacobians,
observation weights, linear, Huber, and soft-L1 losses, and explicit
termination reporting. Central differences cost `2n` residual evaluations per
Jacobian instead of forward differences' `n`.

Explicit positive `x_scale` values condition the LM system for parameters with
different natural magnitudes; there is no automatic Jacobian-derived mode.
Optional box bounds use one strictly feasible projected-LM strategy and report
active limits with a SciPy `active_mask`-style `-1`, `0`, or `+1` per parameter.
Outward-active coordinates are held while tangent coordinates finish converging.
The project is independently installable and does not require any application
from the wider ecosystem.

`LeastSquaresProblem` owns a statically dispatched, stateful residual model
behind validated parameters, weights, and options. `least_squares()` solves it
with the private dense kernel and returns the last valid accepted parameters,
cost, optimality, evaluation counters, and termination reason. Post-fit
covariance and standard-error estimation are available through
`fit_statistics()`. The release-level numerical corpus remains planned. See the
[issue-sized v0.1 implementation plan](docs/implementation-plan.md).

Residual models may be move-only. Each standalone evaluation snapshots its
validated residual dimension and rejects a callback that changes that
declaration before returning. Coherent direct mutation is explicit problem
reconfiguration between evaluations; `least_squares()` captures and enforces
one dimension for its complete solve.

On the fixed contaminated-data fixture, Huber and soft-L1 losses keep the fitted
parameters closer to the generating values than linear loss when one
observation has a gross outlier.

### Conventions

Cost is the robust objective value including the one-half factor:
`0.5 * sum(rho-scaled squared weighted residuals)`. For linear loss this is
exactly `0.5 * sum((w_i * r_i)^2)`. Optimality is the infinity norm of the
gradient.

Covariance is `(J^T W J)^-1 * reduced_chi_squared`, with degrees of freedom
`m_effective - n`; this follows SciPy `curve_fit(..., absolute_sigma=False)`
semantics. Weights multiply residuals once, and inverse-standard-deviation
weighting therefore uses `w_i = 1 / sigma_i`.

## Development

Install [Pixi](https://pixi.sh/), then run:

```sh
pixi install --locked
pixi run check
pixi run example
```

The exact stable Mojo compiler and all development dependencies are captured in
`pixi.lock`. Runtime and library code is Mojo-first and pure Mojo wherever
practical. Build-time data generation may use another language when justified,
but generated outputs must be deterministic, checksum-pinned, licensed, and
documented.

## Package

The Mojo import is `nerai`. The eventual Conda distribution is
`mojo-nerai`. Source lives under `src/nerai/`, whose
`__init__.mojo` defines the package boundary.

The current experimental root exports robust-loss, solver-report, and
fit-statistic values, the residual-model, problem, and option contracts, and
the solve and statistics entry points. These APIs are tested but do not carry a
source-compatibility promise before the first release.

This complete example fits exact observations from
`y(t) = 2.5 * exp(-0.7 * t)`:

```mojo
from nerai import (
    LeastSquaresProblem,
    ResidualModel,
    least_squares,
)
from std.math import exp


struct ExponentialDecayModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 6

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var amplitude = parameters[0]
        var rate = parameters[1]
        # Exact observations follow y(t)=2.5*exp(-0.7*t) at
        # t=[0, 0.5, 1, 1.5, 2, 2.5].
        return [
            amplitude - 2.5,
            amplitude * exp(-0.5 * rate) - 2.5 * exp(-0.35),
            amplitude * exp(-rate) - 2.5 * exp(-0.7),
            amplitude * exp(-1.5 * rate) - 2.5 * exp(-1.05),
            amplitude * exp(-2.0 * rate) - 2.5 * exp(-1.4),
            amplitude * exp(-2.5 * rate) - 2.5 * exp(-1.75),
        ]


def main() raises:
    var problem = LeastSquaresProblem(ExponentialDecayModel(), [1.5, 0.3])
    var result = least_squares(problem)

    print(result, end="")
```

## Repository map

- `src/nerai/`: library or application source
- `tests/`: TestSuite unit, reference-value, and invariant tests
- `examples/`: small compilable usage programs
- `benchmarks/`: reproducible methodology and later benchmark programs
- `docs/`: architecture, design, compatibility, roadmap, and release policy
- `conda.recipe/`: local Rattler build recipe

See [the architecture](docs/architecture.md), [design principles](docs/design.md),
and [roadmap](docs/roadmap.md) before proposing a new dependency or feature.

## License

Licensed under either Apache-2.0 or MIT, at your option.
