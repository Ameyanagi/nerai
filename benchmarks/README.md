# Least-squares benchmarks

`pixi run bench` builds an optimized native executable before measuring four
deterministic 512-residual, four-parameter solver workloads: well-conditioned,
ill-conditioned, upper/lower bounded, and exact-fit. Each reported sample is
four complete solves. Three warmups precede 31 samples; p50 and p95 are
nearest-rank values at sorted indices 15 and 29. Checksums retain solver output
parameters, counters, bound masks, cost, and optimality and reject drift. These
raw-solver cases intentionally exclude the extra callbacks and matrix inversion
performed by `fit_statistics()` and `CurveFit.solve()`.

The same executable measures scalar and native-SIMD `J^T r` and `J^T J` over a
4096-by-8 column-major matrix. Each sample contains 256 calls. Before any timing,
every scalar and SIMD result is compared element-by-element at a `2e-12`
combined absolute/relative tolerance. Unit tests separately cover zero rows
and native-width tails with deterministic scalar differentials.

The benchmark wrapper reports the Git revision and dirty state, a SHA-256
manifest of all compiled Nerai and benchmark sources, CPU, OS, architecture,
compiler version, and exact optimized build command. A dirty run remains
content-identifiable by the source manifest rather than being attributed to the
Git revision alone.

The executable accepts one workload name for a long-running profiling loop:

```sh
pixi run mojo build --optimization-level 3 -I src \
  benchmarks/bench_least_squares.mojo \
  -o .pixi/bench-least-squares
.pixi/bench-least-squares ill_conditioned >.pixi/profile-output.txt &
profile_pid=$!
sample "$profile_pid" 3 1 -file .pixi/profile-ill-conditioned.txt
wait "$profile_pid"
```

Full commands, profiler findings, hardware, and current measurements are
recorded in [PROFILE.md](PROFILE.md). Results are development evidence, not
portable performance promises.
