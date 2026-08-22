# Least-squares profile — 2026-08-22

These results are development evidence, not portable performance promises. The
checked run reported:

```text
git_head=63482c51f4c0be3d878cf8777a0347a4050e8c2f
git_state=dirty
source_manifest_sha256=cde623700a58a587f686e316fabb70b965f0611b2c55804c9db341863ad1c372
timestamp_utc=2026-08-22T00:49:07Z..2026-08-22T00:49:27Z
os=Darwin 25.5.0
arch=arm64
cpu=Apple M4
power=AC Power
thermal=no thermal or performance warning recorded by pmset
mojo=Mojo 1.0.0 (ed45d567)
build=mojo build --optimization-level 3 -I src benchmarks/bench_least_squares.mojo -o .pixi/bench-least-squares
```

The source manifest hashes every compiled `src/nerai/*.mojo` file and the
benchmark source, so the dirty working-tree run is not attributed to the Git
revision alone. Earlier draft measurements did not capture a revision or
source manifest and are intentionally omitted rather than presented as a
reproducible before/after comparison.

## Latency

Three separate `pixi run bench` process campaigns each perform three warmups
and 31 measured samples. p50 and p95 are nearest-rank values at sorted indices
15 and 29. Each solver sample contains four complete solves. The table reports
the median campaign p50/p95 and the range of campaign p50 values so transient
process and power-management variation remains visible.

| Workload | Median p50 (ns) | Median p95 (ns) | Campaign p50 range (ns) |
| --- | ---: | ---: | ---: |
| Well-conditioned | 722,000 | 725,000 | 717,000–722,000 |
| Ill-conditioned | 718,000 | 723,000 | 718,000–722,000 |
| Bounded finite differences | 755,000 | 756,000 | 755,000–762,000 |
| Exact fit | 716,000 | 718,000 | 716,000–723,000 |

The ill-conditioned label describes correlated columns; it is not treated as
proof that every solve took the adaptive QR branch. Focused solver and kernel
tests separately force that branch, verify recovery of the independent
direction, and cover extreme finite rescaling and rank rejection. These raw
solver timings exclude `fit_statistics()` and `CurveFit.solve()` diagnostics.

## SIMD kernel evidence

Before timing, the executable rejects non-finite results and compares every
scalar and SIMD output element at a `2e-12` combined absolute/relative
tolerance. Each timed sample contains 256 calls over the same deterministic
4096-by-8 column-major matrix.

| Kernel | Scalar median p50/p95 (ns) | SIMD median p50/p95 (ns) | Scalar/SIMD p50 ranges (ns) | Median campaign speedup |
| --- | ---: | ---: | ---: | ---: |
| `J^T r` | 20,986,000 / 22,641,000 | 5,929,000 / 6,614,000 | 20,537,000–21,048,000 / 5,885,000–6,621,000 | 3.49x |
| `J^T J` | 99,704,000 / 112,756,000 | 29,582,000 / 29,802,000 | 77,673,000–113,436,000 / 23,235,000–31,253,000 | 3.34x |

This supports SIMD only for the contiguous column products. Bound selection
and the small, control-heavy Householder factorization remain scalar; this run
did not isolate a vectorized QR experiment and makes no speed claim for one.

## CPU profile

Every solver workload was sampled with the real macOS profiler at 1 ms
intervals for two seconds:

```sh
.pixi/bench-least-squares well_conditioned >.pixi/profile-stable-well_conditioned.out &
profile_pid=$!
sample "$profile_pid" 2 1 -file .pixi/profile-stable-well_conditioned.txt
wait "$profile_pid"
```

The other mode names are `ill_conditioned`, `bounded_finite_difference`, and
`exact_fit`. Main-thread sample counts and the leading exclusive symbols were:

| Workload | Main samples | Residual callback | Finite differences | Objective build | Objective cost | `J^T r` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Well-conditioned | 1,700 | 624 | 333 | 196 | 117 | 52 |
| Ill-conditioned | 1,691 | 625 | 338 | 168 | 120 | 38 |
| Bounded | 1,673 | 582 | 299 | 181 | 146 | 29 |
| Exact fit | 1,671 | 630 | 291 | 204 | 140 | 34 |

The ignored raw reports are identified here so a retained local capture can be
verified independently:

| Workload | Profiler timestamp (+0900) | Raw report SHA-256 |
| --- | --- | --- |
| Well-conditioned | 2026-08-22 09:49:48.052 | `21ada132a493fc3b56999c041b633d20d8905c917c557042427ebb8f89bb1198` |
| Ill-conditioned | 2026-08-22 09:49:50.692 | `2feeb52ab247fd84e8e586d703fc561f0a3371939b82bb13c9c3d283645f5861` |
| Bounded | 2026-08-22 09:49:53.754 | `449cd6cb53f8320fb51e10613786b63d52eac38e7592c5145fd66617e318c719` |
| Exact fit | 2026-08-22 09:49:57.392 | `373c24c4b19a7915b196c04af6aa9d974d590f000ae0e862b5f1a0c52d622957` |

The user callback, finite-difference callbacks, and objective construction are
the dominant family in every workload. The SIMD-backed products are measurable
but no longer dominate the full solve, so further work should target callback
and objective traffic only with new profile evidence. Raw profiler reports are
kept under ignored `.pixi/` paths rather than committed as portable results.
