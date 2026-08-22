from nerai._kernel import (
    _DenseMatrix,
    _QrWorkspace,
    _add_diagonal_damping,
    _dot,
    _jt_j,
    _jt_j_scalar,
    _jt_residual,
    _jt_residual_scalar,
    _norm2,
    _norm_inf,
    _solve_least_squares_qr,
    _solve_damped_least_squares_qr,
    _solve_spd,
)
from std.math import abs, sqrt
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.utils.numerics import inf


def assert_close(
    actual: Float64, expected: Float64, tolerance: Float64 = 1.0e-12
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def assert_relative_close(
    actual: Float64, expected: Float64, tolerance: Float64 = 2.0e-15
) raises:
    assert_true(abs(actual / expected - 1.0) <= tolerance)


def matrix_from_values(
    rows: Int, cols: Int, values: List[Float64]
) raises -> _DenseMatrix:
    if len(values) != rows * cols:
        raise Error("test matrix fixture has the wrong value count")
    var matrix = _DenseMatrix(rows, cols)
    for row in range(rows):
        for col in range(cols):
            matrix.set(row, col, values[row * cols + col])
    return matrix^


def test_matrix_access_is_checked_and_layout_is_private() raises:
    var matrix = _DenseMatrix(2, 3)
    matrix.set(1, 2, 7.5)
    assert_equal(matrix.rows, 2)
    assert_equal(matrix.cols, 3)
    assert_close(matrix.get(1, 2), 7.5)

    with assert_raises(contains="matrix dimensions must be non-negative"):
        _ = _DenseMatrix(-1, 2)
    with assert_raises(contains="matrix index is out of range"):
        _ = matrix.get(-1, 0)
    with assert_raises(contains="matrix index is out of range"):
        matrix.set(0, 3, 0.0)


def test_dot_and_vector_norm_fixtures() raises:
    # 1*4 + (-2)*5 + 3*(-6) = -24.
    assert_close(_dot([1.0, -2.0, 3.0], [4.0, 5.0, -6.0]), -24.0)
    assert_close(_norm2([3.0, 4.0]), 5.0)
    assert_close(_norm_inf([-3.0, 4.0, -12.0]), 12.0)
    assert_close(_norm2(List[Float64]()), 0.0)
    assert_close(_norm_inf(List[Float64]()), 0.0)

    # Naive squaring overflows, while the true norm is representable.
    assert_relative_close(_norm2([3.0e200, 4.0e200]), 5.0e200)


def test_transpose_products_match_hand_computation() raises:
    # J = [[1, 2], [3, 4], [5, 6]], r = [7, 8, 9].
    # J^T r = [76, 100], J^T J = [[35, 44], [44, 56]].
    var jacobian = matrix_from_values(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var transpose_residual = _jt_residual(jacobian, [7.0, 8.0, 9.0])
    assert_close(transpose_residual[0], 76.0)
    assert_close(transpose_residual[1], 100.0)

    var normal = _jt_j(jacobian)
    assert_close(normal.get(0, 0), 35.0)
    assert_close(normal.get(0, 1), 44.0)
    assert_close(normal.get(1, 0), 44.0)
    assert_close(normal.get(1, 1), 56.0)


def test_simd_transpose_product_matches_scalar_across_tails() raises:
    for rows in [0, 1, 2, 3, 7, 64, 257]:
        var jacobian = _DenseMatrix(rows, 5)
        var residuals = List[Float64](length=rows, fill=0.0)
        for row in range(rows):
            residuals[row] = Float64((row * 29) % 31 - 15) / 17.0
            for col in range(5):
                jacobian.set(
                    row,
                    col,
                    Float64((row * 13 + col * 19) % 37 - 18) / 11.0,
                )
        var scalar = _jt_residual_scalar(jacobian, residuals)
        var simd = _jt_residual(jacobian, residuals)
        for col in range(5):
            assert_close(simd[col], scalar[col], 2.0e-12)


def test_simd_normal_matrix_matches_scalar_across_tails() raises:
    for rows in [0, 1, 2, 3, 7, 64, 257]:
        var jacobian = _DenseMatrix(rows, 5)
        for row in range(rows):
            for col in range(5):
                jacobian.set(
                    row,
                    col,
                    Float64((row * 13 + col * 19) % 37 - 18) / 11.0,
                )
        var scalar = _jt_j_scalar(jacobian)
        var simd = _jt_j(jacobian)
        for row in range(5):
            for col in range(5):
                assert_close(simd.get(row, col), scalar.get(row, col), 2.0e-12)


def test_diagonal_damping_fixture() raises:
    # [[4, 1], [1, 9]] + 0.5 * diag([2, 3]) = [[5, 1], [1, 10.5]].
    var matrix = matrix_from_values(2, 2, [4.0, 1.0, 1.0, 9.0])
    _add_diagonal_damping(matrix, 0.5, [2.0, 3.0])
    assert_close(matrix.get(0, 0), 5.0)
    assert_close(matrix.get(0, 1), 1.0)
    assert_close(matrix.get(1, 0), 1.0)
    assert_close(matrix.get(1, 1), 10.5)


def test_one_by_one_spd_solve() raises:
    # 4*x = 8, so x = 2.
    var matrix = matrix_from_values(1, 1, [4.0])
    var solution = _solve_spd(matrix, [8.0])
    assert_close(solution[0], 2.0)


def test_two_by_two_spd_solve() raises:
    # 4*x + y = 1; x + 3*y = 2, so [x, y] = [1/11, 7/11].
    var matrix = matrix_from_values(2, 2, [4.0, 1.0, 1.0, 3.0])
    var solution = _solve_spd(matrix, [1.0, 2.0])
    assert_close(solution[0], 1.0 / 11.0)
    assert_close(solution[1], 7.0 / 11.0)


def test_three_by_three_spd_solve() raises:
    # 4*x + y + z = 9; x + 3*y = 7; x + 2*z = 7.
    # Direct substitution gives [x, y, z] = [1, 2, 3].
    var matrix = matrix_from_values(
        3,
        3,
        [4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0],
    )
    var solution = _solve_spd(matrix, [9.0, 7.0, 7.0])
    assert_close(solution[0], 1.0)
    assert_close(solution[1], 2.0)
    assert_close(solution[2], 3.0)


def test_qr_solve_preserves_nearly_collinear_information() raises:
    var epsilon = 1.0e-8
    var matrix = matrix_from_values(
        4,
        2,
        [
            1.0,
            1.0,
            1.0,
            1.0 + epsilon,
            1.0,
            1.0 + 2.0 * epsilon,
            1.0,
            1.0 + 3.0 * epsilon,
        ],
    )
    # b = 2*column_0 - column_1. Forming J^T J squares this fixture's
    # condition number; Householder QR retains the independent direction.
    var solution = _solve_least_squares_qr(
        matrix,
        [1.0, 1.0 - epsilon, 1.0 - 2.0 * epsilon, 1.0 - 3.0 * epsilon],
    )
    assert_close(solution[0], 2.0, 2.0e-7)
    assert_close(solution[1], -1.0, 2.0e-7)


def test_qr_householder_is_stable_under_extreme_finite_rescaling() raises:
    for scale in [1.0e-310, 1.0e-200, 1.0e200]:
        var matrix = matrix_from_values(
            3,
            2,
            [scale, 0.0, 0.0, scale, scale, scale],
        )
        var solution = _solve_least_squares_qr(
            matrix,
            [2.0 * scale, -3.0 * scale, -scale],
        )
        assert_close(solution[0], 2.0, 2.0e-12)
        assert_close(solution[1], -3.0, 2.0e-12)


def test_qr_extreme_rescaling_still_rejects_rank_deficiency() raises:
    for scale in [1.0e-310, 1.0e200]:
        var matrix = matrix_from_values(
            3,
            2,
            [scale, 2.0 * scale, 2.0 * scale, 4.0 * scale, 3.0 * scale, 6.0 * scale],
        )
        with assert_raises(contains="rank-deficient"):
            _ = _solve_least_squares_qr(matrix, [scale, 2.0 * scale, 3.0 * scale])


def test_damped_qr_avoids_unnecessary_squared_norm_overflow() raises:
    var scale = 1.0e200
    var jacobian = matrix_from_values(2, 1, [scale, scale])
    var workspace = _QrWorkspace(3, 1)
    var solution = _solve_damped_least_squares_qr(
        workspace,
        jacobian,
        [-2.0 * scale, -2.0 * scale],
        1.0e-200,
    )
    assert_close(solution[0], 2.0, 2.0e-12)

    # The unscaled norm of two 1.3e308 entries overflows, while multiplying
    # each scaled component by sqrt(damping) first yields a finite coefficient.
    var extreme_scale = 1.3e308
    var extreme = matrix_from_values(2, 1, [extreme_scale, extreme_scale])
    var extreme_workspace = _QrWorkspace(3, 1)
    extreme_workspace._load_damped(extreme, [0.0, 0.0], 1.0e-308)
    var expected = (sqrt(1.0e-308) * extreme_scale) * sqrt(2.0)
    assert_relative_close(extreme_workspace._matrix.get(2, 0), expected, 2.0e-15)


def test_kernel_shape_mismatches_raise() raises:
    with assert_raises(contains="dot-product vector lengths must match"):
        _ = _dot([1.0], [1.0, 2.0])

    var rectangular = _DenseMatrix(2, 3)
    with assert_raises(contains="Jacobian row count must match"):
        _ = _jt_residual(rectangular, [1.0])
    with assert_raises(contains="diagonal damping requires a square matrix"):
        _add_diagonal_damping(rectangular, 1.0, [1.0, 1.0])
    with assert_raises(contains="positive-definite solve requires a square matrix"):
        _ = _solve_spd(rectangular, [1.0, 2.0])

    var square = _DenseMatrix(2, 2)
    with assert_raises(contains="damping diagonal length must match"):
        _add_diagonal_damping(square, 1.0, [1.0])
    with assert_raises(contains="right-hand-side length must match"):
        _ = _solve_spd(square, [1.0])


def test_singular_indefinite_and_nonfinite_pivots_raise() raises:
    var singular = matrix_from_values(2, 2, [1.0, 1.0, 1.0, 1.0])
    with assert_raises(contains="singular or not positive definite"):
        _ = _solve_spd(singular, [1.0, 1.0])

    var indefinite = matrix_from_values(2, 2, [1.0, 0.0, 0.0, -1.0])
    with assert_raises(contains="singular or not positive definite"):
        _ = _solve_spd(indefinite, [1.0, 1.0])

    var nonfinite = matrix_from_values(1, 1, [inf[DType.float64]()])
    with assert_raises(contains="singular or not positive definite"):
        _ = _solve_spd(nonfinite, [1.0])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
