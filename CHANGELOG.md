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
