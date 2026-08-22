"""Compiled nonlinear least-squares latency and profiling workloads."""

from nerai import Bounds, LeastSquaresProblem, ResidualModel, least_squares
from nerai._kernel import (
    _DenseMatrix,
    _jt_j,
    _jt_j_scalar,
    _jt_residual,
    _jt_residual_scalar,
)
from std.benchmark import keep
from std.collections import List
from std.sys import argv
from std.time import perf_counter_ns
from std.utils.numerics import inf, isfinite


comptime _SAMPLES = 31
comptime _WARMUPS = 3
comptime _SOLVES_PER_SAMPLE = 4


struct ProfileModel(Copyable, ResidualModel):
    """Deterministic linear-in-parameters workload with selectable conditioning."""

    var kind: Int

    def __init__(out self, kind: Int):
        self.kind = kind

    def residual_count(self) -> Int:
        return 512

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var residuals = List[Float64](length=512, fill=0.0)
        for row in range(512):
            var x = (Float64(row) - 255.5) / 256.0
            var first = 1.0
            var second = x
            var third = x * x
            var fourth = third * x
            if self.kind == 1:
                # Deliberately correlated, while remaining full rank.
                second = 1.0 + 1.0e-4 * x
                third = 1.0 - 1.0e-4 * x + 1.0e-7 * x * x
                fourth = 1.0 + 2.0e-4 * x + 1.0e-7 * x * x * x

            var target = (
                1.25 * first - 0.75 * second + 0.5 * third - 0.25 * fourth
            )
            if self.kind == 0:
                target += 1.0e-3 * Float64((row * 17) % 13 - 6)
            elif self.kind == 2:
                # The optimum is inside [0, 1]^4, but the initial point is
                # close enough to upper bounds to exercise bounded differences.
                target = (
                    0.2 * first + 0.3 * second + 0.4 * third + 0.1 * fourth
                )
            residuals[row] = (
                parameters[0] * first
                + parameters[1] * second
                + parameters[2] * third
                + parameters[3] * fourth
                - target
            )
        return residuals^


def _problem(kind: Int) raises -> LeastSquaresProblem[ProfileModel]:
    if kind == 2:
        return LeastSquaresProblem(
            ProfileModel(kind),
            [1.0 - 1.0e-12, 0.8, 0.7, 0.6],
            bounds=Bounds([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        )
    return LeastSquaresProblem(ProfileModel(kind), [0.1, -0.1, 0.2, -0.2])


def _batch(kind: Int) raises -> Float64:
    var checksum = 0.0
    for _ in range(_SOLVES_PER_SAMPLE):
        var problem = _problem(kind)
        var result = least_squares(problem)
        if not result.converged():
            raise Error("benchmark workload did not converge")
        checksum += result.cost + result.optimality
        checksum += Float64(result.iterations * 3)
        checksum += Float64(result.residual_evaluations * 5)
        checksum += Float64(result.jacobian_evaluations * 7)
        for index in range(len(result.parameters)):
            checksum += Float64(index + 1) * result.parameters[index]
            checksum += Float64(index + 11) * Float64(result.active_bounds[index])
    keep(checksum)
    return checksum


def _sort(mut values: List[Int]):
    for index in range(1, len(values)):
        var value = values[index]
        var destination = index
        while destination > 0 and values[destination - 1] > value:
            values[destination] = values[destination - 1]
            destination -= 1
        values[destination] = value


def _measure(name: StringLiteral, kind: Int) raises:
    var expected = _batch(kind)
    for _ in range(_WARMUPS):
        if _batch(kind) != expected:
            raise Error("least-squares benchmark checksum changed during warmup")

    var timings = List[Int](capacity=_SAMPLES)
    for _ in range(_SAMPLES):
        var started = perf_counter_ns()
        var checksum = _batch(kind)
        timings.append(perf_counter_ns() - started)
        if checksum != expected:
            raise Error("least-squares benchmark checksum changed")
    _sort(timings)
    print(
        "case=", name,
        " residuals=512 parameters=4 solves_per_sample=4 samples=31",
        " p50_ns=", timings[15],
        " p95_ns=", timings[29],
        " checksum=", expected,
        sep="",
    )


def _kernel_batch(
    jacobian: _DenseMatrix, residuals: List[Float64], simd: Bool
) raises -> Float64:
    var checksum = 0.0
    for _ in range(256):
        var product = (
            _jt_residual(jacobian, residuals)
            if simd
            else _jt_residual_scalar(jacobian, residuals)
        )
        for value in product:
            checksum += value
    keep(checksum)
    return checksum


def _normal_kernel_batch(jacobian: _DenseMatrix, simd: Bool) raises -> Float64:
    var checksum = 0.0
    for _ in range(256):
        var product = _jt_j(jacobian) if simd else _jt_j_scalar(jacobian)
        for value in product._values:
            checksum += value
    keep(checksum)
    return checksum


def _fill_kernel_fixture(
    mut jacobian: _DenseMatrix, mut residuals: List[Float64]
) raises:
    for row in range(jacobian.rows):
        residuals[row] = Float64((row * 29) % 31 - 15) / 17.0
        for col in range(jacobian.cols):
            jacobian.set(
                row,
                col,
                Float64((row * 13 + col * 19) % 37 - 18) / 11.0,
            )


def _verify_kernel_equivalence(
    jacobian: _DenseMatrix, residuals: List[Float64]
) raises:
    var scalar_residual = _jt_residual_scalar(jacobian, residuals)
    var simd_residual = _jt_residual(jacobian, residuals)
    for index in range(len(scalar_residual)):
        var tolerance = 2.0e-12 * max(1.0, abs(scalar_residual[index]))
        if (
            not isfinite(scalar_residual[index])
            or not isfinite(simd_residual[index])
            or abs(simd_residual[index] - scalar_residual[index]) > tolerance
        ):
            raise Error("SIMD J^T r differs from the scalar reference")

    var scalar_normal = _jt_j_scalar(jacobian)
    var simd_normal = _jt_j(jacobian)
    for index in range(len(scalar_normal._values)):
        var tolerance = 2.0e-12 * max(1.0, abs(scalar_normal._values[index]))
        if (
            not isfinite(scalar_normal._values[index])
            or not isfinite(simd_normal._values[index])
            or abs(simd_normal._values[index] - scalar_normal._values[index])
            > tolerance
        ):
            raise Error("SIMD J^T J differs from the scalar reference")


def _measure_kernel(name: StringLiteral, simd: Bool) raises:
    var jacobian = _DenseMatrix(4096, 8)
    var residuals = List[Float64](length=4096, fill=0.0)
    _fill_kernel_fixture(jacobian, residuals)
    var expected = _kernel_batch(jacobian, residuals, simd)
    for _ in range(_WARMUPS):
        if _kernel_batch(jacobian, residuals, simd) != expected:
            raise Error("transpose-product checksum changed during warmup")
    var timings = List[Int](capacity=_SAMPLES)
    for _ in range(_SAMPLES):
        var started = perf_counter_ns()
        var checksum = _kernel_batch(jacobian, residuals, simd)
        timings.append(perf_counter_ns() - started)
        if checksum != expected:
            raise Error("transpose-product checksum changed")
    _sort(timings)
    print(
        "case=jt_residual_", name,
        " rows=4096 columns=8 calls_per_sample=256 samples=31",
        " p50_ns=", timings[15],
        " p95_ns=", timings[29],
        " checksum=", expected,
        sep="",
    )


def _measure_normal_kernel(name: StringLiteral, simd: Bool) raises:
    var jacobian = _DenseMatrix(4096, 8)
    var residuals = List[Float64](length=4096, fill=0.0)
    _fill_kernel_fixture(jacobian, residuals)
    var expected = _normal_kernel_batch(jacobian, simd)
    for _ in range(_WARMUPS):
        if _normal_kernel_batch(jacobian, simd) != expected:
            raise Error("normal-matrix checksum changed during warmup")
    var timings = List[Int](capacity=_SAMPLES)
    for _ in range(_SAMPLES):
        var started = perf_counter_ns()
        var checksum = _normal_kernel_batch(jacobian, simd)
        timings.append(perf_counter_ns() - started)
        if checksum != expected:
            raise Error("normal-matrix checksum changed")
    _sort(timings)
    print(
        "case=jt_j_", name,
        " rows=4096 columns=8 calls_per_sample=256 samples=31",
        " p50_ns=", timings[15],
        " p95_ns=", timings[29],
        " checksum=", expected,
        sep="",
    )


def main() raises:
    var arguments = argv()
    if len(arguments) == 2:
        var mode = String(arguments[1])
        var kind = -1
        if mode == "well_conditioned":
            kind = 0
        elif mode == "ill_conditioned":
            kind = 1
        elif mode == "bounded_finite_difference":
            kind = 2
        elif mode == "exact_fit":
            kind = 3
        if kind < 0:
            raise Error("unknown least-squares profiling workload")
        var checksum = 0.0
        for _ in range(5_000):
            checksum += _batch(kind)
        keep(checksum)
        print("profile_case=", mode, " checksum=", checksum, sep="")
        return

    print(
        "BENCH_HEADER nerai-least-squares mojo=1.0.0 optimized=true ",
        "warmup=3 statistic=nearest-rank-p50-p95",
        sep="",
    )
    var verification_jacobian = _DenseMatrix(4096, 8)
    var verification_residuals = List[Float64](length=4096, fill=0.0)
    _fill_kernel_fixture(verification_jacobian, verification_residuals)
    _verify_kernel_equivalence(verification_jacobian, verification_residuals)
    print(
        "KERNEL_EQUIVALENCE scalar_simd=pass "
        "combined_abs_relative=2e-12"
    )
    _measure("well_conditioned", 0)
    _measure("ill_conditioned", 1)
    _measure("bounded_finite_difference", 2)
    _measure("exact_fit", 3)
    _measure_kernel("scalar", False)
    _measure_kernel("simd", True)
    _measure_normal_kernel("scalar", False)
    _measure_normal_kernel("simd", True)
