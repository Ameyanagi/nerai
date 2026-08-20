# Nerai

> **Experimental — API not yet released.**

Optimization and nonlinear least squares for Mojo.

## Scope

Nerai begins with explicit, inspectable nonlinear least-squares contracts
instead of attempting a broad SciPy.optimize clone.

The implemented core is intentionally narrow: dense `Float64`
Levenberg-Marquardt, forward finite-difference Jacobians, observation weights,
linear, Huber, and soft-L1 losses, and explicit termination reporting.
The project is independently installable and does not require any application
from the wider ecosystem.

`LeastSquaresProblem` owns a statically dispatched, stateful residual model
behind validated parameters, weights, and options. `least_squares()` solves it
with the private dense kernel and returns the last valid accepted parameters,
cost, optimality, evaluation counters, and termination reason. Covariance
estimation and the release-level numerical corpus remain planned. See the
[issue-sized v0.1 implementation plan](docs/implementation-plan.md).

Residual models may be move-only. Each standalone evaluation snapshots its
validated residual dimension and rejects a callback that changes that
declaration before returning. Coherent direct mutation is explicit problem
reconfiguration between evaluations; `least_squares()` captures and enforces
one dimension for its complete solve.

On the fixed contaminated-data fixture, Huber and soft-L1 losses keep the fitted
parameters closer to the generating values than linear loss when one
observation has a gross outlier.

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

The current experimental root exports robust-loss and solver-report values,
the residual-model, problem, and option contracts, and the `least_squares()`
solve entry point. These APIs are tested but do not carry a source-compatibility
promise before the first release.

This complete example fits exact observations from
`y(t) = 2.5 * exp(-0.7 * t)`:

```mojo
from nerai import (
    LeastSquaresProblem,
    ResidualModel,
    TerminationReason,
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


def termination_name(reason: TerminationReason) -> String:
    if reason == TerminationReason.GRADIENT_TOLERANCE:
        return "gradient tolerance"
    if reason == TerminationReason.STEP_TOLERANCE:
        return "step tolerance"
    if reason == TerminationReason.COST_TOLERANCE:
        return "cost tolerance"
    if reason == TerminationReason.MAX_ITERATIONS:
        return "maximum iterations"
    if reason == TerminationReason.MAX_EVALUATIONS:
        return "maximum residual evaluations"
    if reason == TerminationReason.NUMERICAL_FAILURE:
        return "numerical failure"
    return "unknown"


def main() raises:
    var problem = LeastSquaresProblem(ExponentialDecayModel(), [1.5, 0.3])
    var result = least_squares(problem)

    print("amplitude:", result.parameters[0])
    print("decay rate:", result.parameters[1])
    print("cost:", result.cost)
    print("iterations:", result.iterations)
    print("residual evaluations:", result.residual_evaluations)
    print("Jacobian evaluations:", result.jacobian_evaluations)
    print("termination:", termination_name(result.termination))
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
