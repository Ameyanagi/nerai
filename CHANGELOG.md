# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and uses semantic versioning after the first public release.

## [Unreleased]

### Added

- Initial experimental repository scaffold.
- Robust-loss, termination, and least-squares result foundations.
- Statically dispatched stateful residual models with validated problem and
  solver-option contracts.
- Entry-snapshotted residual dimensions, explicit between-call problem
  reconfiguration, and positive move-only model coverage.
- Private row-major dense numerical kernel and forward finite-difference
  Jacobians with exact residual-evaluation accounting.
- Weighted linear, Huber, and soft-L1 objective construction for solver steps.
- Public Levenberg-Marquardt `least_squares()` with deterministic damping,
  explicit termination, and last-valid-state result reporting.
- Named `CurveFit` parameters with report labels and result lookup methods.
- Scalar-broadcast, one-sided, and nonnegative `Bounds` conveniences.

### Changed

- **Breaking:** `CurveFit.solve()` now returns converged fits with optional
  statistics and a reportable reason when uncertainty estimation fails.
- **Breaking:** `LeastSquaresResult` now records the raw residual vector at the
  returned parameters, and exact result equality includes it.
- Weighted `CurveFit` and `fit_statistics()` calls accept `absolute_sigma=True`
  to skip reduced-chi-squared covariance rescaling.
- Bounded initial guesses on or outside an endpoint are nudged strictly inside,
  following SciPy, while explicit `validate()` retains the strict checkpoint.
- **Format change:** curve-fit reports mark pinned parameter values instead of
  printing misleading uncertainty and include active-bound lines.
