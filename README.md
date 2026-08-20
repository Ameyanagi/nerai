# Nerai

> **Experimental — API not yet released.**

Optimization and nonlinear least squares for Mojo.

## Scope

Nerai begins with explicit, inspectable nonlinear least-squares contracts instead of attempting a broad SciPy.optimize clone.

The first implementation milestone is intentionally narrow: implement
Levenberg-Marquardt, finite-difference Jacobians, weighted residuals, robust
losses, covariance estimates, and explicit termination reporting.
The project is independently installable and does not require any application
from the wider ecosystem.

The first two public foundations are implemented: built-in loss functions use
an explicit normalized-squared-residual convention; solver results distinguish
successful convergence, budget exhaustion, and numerical failure; and
`LeastSquaresProblem` owns a statically dispatched, stateful residual model
behind validated `Float64` parameters, weights, and options. The solve loop,
Jacobian, and numerical kernel are not implemented. See the
[issue-sized v0.1 implementation plan](docs/implementation-plan.md).

Residual models may be move-only. Each standalone evaluation snapshots its
validated residual dimension and rejects a callback that changes that
declaration before returning. Coherent direct mutation is explicit problem
reconfiguration between evaluations; a future solver will capture and enforce
one dimension for its complete solve.

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

The current experimental root exports robust-loss and solver-report semantic
values plus the residual-model, problem, and option contracts. These APIs are
tested but do not carry a source-compatibility promise before the first release.

This complete example defines a stateful residual model and evaluates its
initial residuals with weights and a robust loss:

```mojo
from nerai import (
    LeastSquaresOptions,
    LeastSquaresProblem,
    LossKind,
    ResidualModel,
)


struct AffineModel(Copyable, ResidualModel):
    """Two equations in two parameters with an observable call count."""

    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [
            parameters[0] + 2.0 * parameters[1] - 5.0,
            3.0 * parameters[0] - parameters[1] - 4.0,
        ]


def main() raises:
    var problem = LeastSquaresProblem(
        AffineModel(),
        [1.0, 2.0],
        weights=[1.0, 0.5],
        options=LeastSquaresOptions(loss=LossKind.SOFT_L1),
    )
    var residuals = problem.evaluate_initial_residuals()

    print("residual 0:", residuals[0])
    print("residual 1:", residuals[1])
    print("model calls:", problem.model.calls)
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
