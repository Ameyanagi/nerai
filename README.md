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

The first public foundation is implemented: built-in loss functions use an
explicit normalized-squared-residual convention, and solver results distinguish
successful convergence, budget exhaustion, and numerical failure. The solve
loop is not implemented. See the
[issue-sized v0.1 implementation plan](docs/implementation-plan.md).

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
values. These APIs are tested but do not carry a source-compatibility promise
before the first release.

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
