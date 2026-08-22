# Nerai

> **Pre-1.0 — minor releases may change source compatibility.**

Optimization and nonlinear least squares for Mojo.

## Install

For a [Pixi](https://pixi.sh/) project, add the Nerai channel to
`[workspace].channels` in `pixi.toml`:

```toml
[workspace]
channels = [
    "https://ameyanagi.github.io/mojo-channel",
    "https://conda.modular.com/max",
    "conda-forge",
]
```

Then add the package:

```sh
pixi add mojo-nerai
```

Alternatively, run Nerai from a source checkout:

```sh
git clone https://github.com/Ameyanagi/nerai.git
cd nerai
pixi install --locked
```

Save your program in the checkout as `your_fit.mojo`, then run it with
`pixi run mojo run -I src your_fit.mojo`.

## Quickstart

Define the curve itself; `CurveFit` builds residuals, solves, and estimates
uncertainty:

```mojo
from nerai import CurveFit, CurveModel
from std.collections import List
from std.math import exp

struct Decay(CurveModel):
    # This empty initializer is required Mojo boilerplate for a fieldless struct.
    def __init__(out self):
        pass

    def values(
        mut self, p: List[Float64], t: Span[Float64, ...]
    ) raises -> List[Float64]:
        var y = List[Float64](length=len(t), fill=0.0)
        for i in range(len(t)):
            y[i] = p[0] * exp(-p[1] * t[i])
        return y^

def main() raises:
    var t = [0.0, 0.20833333333333334, 0.4166666666666667, 0.625,
             0.8333333333333334, 1.0416666666666667, 1.25,
             1.4583333333333335, 1.6666666666666667, 1.875,
             2.0833333333333335, 2.291666666666667, 2.5,
             2.7083333333333335, 2.916666666666667, 3.125,
             3.3333333333333335, 3.541666666666667, 3.75,
             3.9583333333333335, 4.166666666666667, 4.375,
             4.583333333333334, 4.791666666666667, 5.0]
    var y = [2.5152358539877215, 2.1087551483663853, 1.9050663105664043,
             1.6611495518892907, 1.2975361049924259, 1.1406678953850795,
             1.0485470693546353, 0.8849242942863328, 0.7776680019112797,
             0.6302136754442806, 0.6255290435363732, 0.5415321783848346,
             0.43773639350417365, 0.4318448652148168, 0.34790590925119713,
             0.23752760315669744, 0.26086745886513757,
             0.16158847170632618, 0.22502190765099225,
             0.1540282248418524, 0.126041297379791,
             0.08288007873970038, 0.16218689287408977,
             0.07961970944963043, 0.05407706744764089]
    var fit = CurveFit(
        Decay(),
        t,
        y,
        [1.0, 1.0],
        parameter_names=["amplitude", "rate"],
    )
    print(fit.solve(), end="")
```

Save this quickstart as `fit.mojo` in a checkout and run
`pixi run mojo run -I src fit.mojo`.

```text
termination           cost tolerance
converged             yes
cost                  0.020486345097035032
optimality            5.484938620439154e-07
iterations            19
residual evaluations  40
jacobian evaluations  10
amplitude             2.487 +/- 0.028
rate                  0.699 +/- 0.012
degrees of freedom    23
reduced chi-squared   0.0017814213127856553
```

Pass solver options inline, such as `options=LeastSquaresOptions(loss=...)`, or
transfer an existing variable with `options=options^`. Plain `options=options`
cannot be implicitly copied.

## Examples

The task-focused examples are
[`exponential_decay.mojo`](examples/exponential_decay.mojo),
[`gaussian_peak.mojo`](examples/gaussian_peak.mojo), and
[`bounded_decay.mojo`](examples/bounded_decay.mojo). The last one uses the raw
problem API to contrast an unbounded fit with a physically bounded fit.
[`ill_conditioned.mojo`](examples/ill_conditioned.mojo) demonstrates two
strongly correlated columns handled without adding a linear-solver option to
the public API.

## Scope

Nerai provides dense `Float64` nonlinear least squares with forward or central
bound-aware finite-difference Jacobians, observation weights, linear and robust
losses, parameter scaling, box bounds, adaptive QR stabilization, explicit
termination reports, and post-fit covariance estimates. Full-rank exact fits
with positive degrees of freedom report zero covariance and zero standard
errors rather than failing validation; correlation is then undefined and
`FitStatistics.correlation()` raises. `CurveFit` is the data-fitting front door;
`LeastSquaresProblem` and `least_squares()` remain the expert residual-model
API. See the [issue-sized v0.1 implementation plan](docs/implementation-plan.md).

### Conventions

Cost is the robust objective value including the one-half factor:
`0.5 * sum(rho-scaled squared weighted residuals)`. For linear loss this is
exactly `0.5 * sum((w_i * r_i)^2)`. Optimality is the infinity norm of the
gradient.

An initial guess on or outside a bound is nudged strictly inside, following
SciPy.

Covariance is `(J_model^T J_model)^-1 * reduced_chi_squared`, with degrees of
freedom `m_effective - n`; this follows SciPy
`curve_fit(..., absolute_sigma=False)` semantics. For linear loss,
`J_model = diag(w_i) J_raw`, so the normal matrix contains `w_i^2`. On the
`CurveFit` front door, `sigma` maps to `w_i = 1 / sigma_i`. Supplying `sigma`
with `absolute_sigma=True` skips the reduced-chi-squared rescaling, matching
SciPy's known-measurement-uncertainty convention.

## Development

From a source checkout, run:

```sh
pixi run check
pixi run example
pixi run bench
```

The exact stable Mojo compiler and all development dependencies are captured in
`pixi.lock`. Runtime and library code is Mojo-first and pure Mojo wherever
practical. Build-time data generation may use another language when justified,
but generated outputs must be deterministic, checksum-pinned, licensed, and
documented.

## Package

The Mojo import is `nerai`. The Conda distribution is
`mojo-nerai`. Source lives under `src/nerai/`, whose
`__init__.mojo` defines the package boundary.

The `0.x` root exports both curve-fitting and expert least-squares APIs. They are
tested, but minor releases may change source compatibility before 1.0.

## Repository map

- `src/nerai/`: library or application source
- `tests/`: TestSuite unit, reference-value, and invariant tests
- `examples/`: small compilable usage programs
- `benchmarks/`: reproducible methodology, profiler evidence, and benchmarks
- `docs/`: architecture, design, compatibility, roadmap, and release policy
- `conda.recipe/`: local Rattler build recipe

See [the architecture](docs/architecture.md), [design principles](docs/design.md),
and [roadmap](docs/roadmap.md) before proposing a new dependency or feature.

## License

Licensed under either Apache-2.0 or MIT, at your option.
