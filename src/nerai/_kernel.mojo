"""Private dense ``Float64`` operations used by the least-squares solver.

Matrices use one flat column-major ``List[Float64]``. Jacobian columns are the
hot vectors for finite differences, transpose products, and QR, so this SoA
layout keeps every one of those scans contiguous. Public-to-this-module
``get()`` and ``set()`` calls check indices and raise on invalid input. Kernel
loops first check argument shapes, then index the trusted flat buffers directly.
"""

from std.math import sqrt
from std.sys import simd_width_of
from std.utils.numerics import isfinite


struct _DenseMatrix(Copyable):
    """A private dense column-major matrix value."""

    var rows: Int
    var cols: Int
    var _values: List[Float64]

    def __init__(out self, rows: Int, cols: Int, *, fill: Float64 = 0.0) raises:
        if rows < 0 or cols < 0:
            raise Error("matrix dimensions must be non-negative")
        self.rows = rows
        self.cols = cols
        self._values = List[Float64](length=rows * cols, fill=fill)

    def get(self, row: Int, col: Int) raises -> Float64:
        """Return one element, raising when either index is out of range."""
        self._check_index(row, col)
        return self._values[col * self.rows + row]

    def set(mut self, row: Int, col: Int, value: Float64) raises:
        """Set one element, raising when either index is out of range."""
        self._check_index(row, col)
        self._values[col * self.rows + row] = value

    def _offset(self, row: Int, col: Int) -> Int:
        return col * self.rows + row

    def _check_index(self, row: Int, col: Int) raises:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise Error("matrix index is out of range")


def _dot(left: List[Float64], right: List[Float64]) raises -> Float64:
    """Return the dot product of two equal-length vectors."""
    if len(left) != len(right):
        raise Error("dot-product vector lengths must match")
    var result = 0.0
    for index in range(len(left)):
        result += left[index] * right[index]
    return result


def _norm2(values: List[Float64]) -> Float64:
    """Return a scaled Euclidean norm without overflowing naive squares."""
    var scale = 0.0
    var scaled_sum = 1.0
    for index in range(len(values)):
        var magnitude = abs(values[index])
        if magnitude == 0.0:
            continue
        if scale < magnitude:
            var ratio = scale / magnitude
            scaled_sum = 1.0 + scaled_sum * ratio * ratio
            scale = magnitude
        else:
            var ratio = magnitude / scale
            scaled_sum += ratio * ratio
    if scale == 0.0:
        return 0.0
    return scale * sqrt(scaled_sum)


def _norm_inf(values: List[Float64]) -> Float64:
    """Return the maximum absolute vector element, or zero when empty."""
    var result = 0.0
    for index in range(len(values)):
        result = max(result, abs(values[index]))
    return result


def _jt_residual(
    jacobian: _DenseMatrix, residuals: List[Float64]
) raises -> List[Float64]:
    """Return ``J^T r`` for a row-compatible residual vector."""
    if jacobian.rows != len(residuals):
        raise Error("Jacobian row count must match residual vector length")

    var result = List[Float64](length=jacobian.cols, fill=0.0)
    if jacobian.rows == 0:
        return result^
    comptime width = simd_width_of[DType.float64]()
    var vector_end = jacobian.rows - jacobian.rows % width
    # Safety: both pointers refer to live contiguous Float64 lists. Each load
    # begins below vector_end, which is rounded down to a complete native SIMD
    # width, and each Jacobian column has exactly `rows` elements.
    var jacobian_ptr = jacobian._values.unsafe_ptr()
    var residual_ptr = residuals.unsafe_ptr()
    for col in range(jacobian.cols):
        var vector_value = SIMD[DType.float64, width](0.0)
        var column_start = col * jacobian.rows
        for row in range(0, vector_end, width):
            vector_value += jacobian_ptr.unsafe_load[width=width](
                column_start + row
            ) * residual_ptr.unsafe_load[width=width](row)
        var value = vector_value.reduce_add()
        for row in range(vector_end, jacobian.rows):
            value += jacobian._values[column_start + row] * residuals[row]
        result[col] = value
    return result^


def _jt_residual_scalar(
    jacobian: _DenseMatrix, residuals: List[Float64]
) raises -> List[Float64]:
    """Scalar reference for SIMD differential tests and tiny diagnostics."""
    if jacobian.rows != len(residuals):
        raise Error("Jacobian row count must match residual vector length")
    var result = List[Float64](length=jacobian.cols, fill=0.0)
    for col in range(jacobian.cols):
        var value = 0.0
        var column_start = col * jacobian.rows
        for row in range(jacobian.rows):
            value += jacobian._values[column_start + row] * residuals[row]
        result[col] = value
    return result^


def _jt_j(jacobian: _DenseMatrix) raises -> _DenseMatrix:
    """Return the symmetric normal matrix ``J^T J``."""
    var result = _DenseMatrix(jacobian.cols, jacobian.cols)
    for left_col in range(jacobian.cols):
        for right_col in range(left_col, jacobian.cols):
            var value = _column_dot(jacobian, left_col, right_col)
            result._values[result._offset(left_col, right_col)] = value
            result._values[result._offset(right_col, left_col)] = value
    return result^


def _jt_j_scalar(jacobian: _DenseMatrix) raises -> _DenseMatrix:
    """Scalar normal-matrix reference for SIMD differential tests."""
    var result = _DenseMatrix(jacobian.cols, jacobian.cols)
    for left_col in range(jacobian.cols):
        for right_col in range(left_col, jacobian.cols):
            var value = _column_dot_scalar(jacobian, left_col, right_col)
            result._values[result._offset(left_col, right_col)] = value
            result._values[result._offset(right_col, left_col)] = value
    return result^


def _add_diagonal_damping(
    mut matrix: _DenseMatrix, damping: Float64, diagonal: List[Float64]
) raises:
    """Add ``damping * diagonal`` to a square matrix's diagonal."""
    if matrix.rows != matrix.cols:
        raise Error("diagonal damping requires a square matrix")
    if len(diagonal) != matrix.rows:
        raise Error("damping diagonal length must match the matrix dimension")
    for index in range(matrix.rows):
        var offset = matrix._offset(index, index)
        matrix._values[offset] += damping * diagonal[index]


def _solve_spd(
    matrix: _DenseMatrix,
    right_hand_side: List[Float64],
    *,
    minimum_relative_pivot: Float64 = 0.0,
) raises -> List[Float64]:
    """Solve a symmetric positive-definite system by Cholesky factorization.

    A non-positive or non-finite Cholesky pivot is a numerical failure and
    raises explicitly. The private solver supplies symmetric matrices, so the
    lower triangle is authoritative during factorization.
    """
    if matrix.rows != matrix.cols:
        raise Error("positive-definite solve requires a square matrix")
    if len(right_hand_side) != matrix.rows:
        raise Error("right-hand-side length must match the matrix dimension")
    if not isfinite(minimum_relative_pivot) or minimum_relative_pivot < 0.0:
        raise Error("minimum relative pivot must be finite and non-negative")

    var dimension = matrix.rows
    var lower = _DenseMatrix(dimension, dimension)
    var largest_diagonal = 0.0
    for index in range(dimension):
        largest_diagonal = max(
            largest_diagonal, abs(matrix._values[matrix._offset(index, index)])
        )
    for row in range(dimension):
        for col in range(row + 1):
            var value = matrix._values[matrix._offset(row, col)]
            for inner in range(col):
                value -= (
                    lower._values[lower._offset(row, inner)]
                    * lower._values[lower._offset(col, inner)]
                )

            if row == col:
                if (
                    not isfinite(value)
                    or value <= 0.0
                    or value <= minimum_relative_pivot * largest_diagonal
                ):
                    raise Error("matrix is singular or not positive definite")
                lower._values[lower._offset(row, col)] = sqrt(value)
            else:
                lower._values[lower._offset(row, col)] = (
                    value / lower._values[lower._offset(col, col)]
                )

    var intermediate = List[Float64](length=dimension, fill=0.0)
    for row in range(dimension):
        var value = right_hand_side[row]
        for col in range(row):
            value -= lower._values[lower._offset(row, col)] * intermediate[col]
        intermediate[row] = value / lower._values[lower._offset(row, row)]

    var solution = List[Float64](length=dimension, fill=0.0)
    for reverse_row in range(dimension):
        var row = dimension - reverse_row - 1
        var value = intermediate[row]
        for col in range(row + 1, dimension):
            value -= lower._values[lower._offset(col, row)] * solution[col]
        solution[row] = value / lower._values[lower._offset(row, row)]
    return solution^


def _column_dot(matrix: _DenseMatrix, left_col: Int, right_col: Int) -> Float64:
    """Return a native-SIMD dot product between two trusted matrix columns."""
    if matrix.rows == 0:
        return 0.0
    comptime width = simd_width_of[DType.float64]()
    var vector_end = matrix.rows - matrix.rows % width
    var values = matrix._values.unsafe_ptr()
    var left_start = left_col * matrix.rows
    var right_start = right_col * matrix.rows
    var vector_value = SIMD[DType.float64, width](0.0)
    # Safety: both starts identify complete live columns and vector_end rounds
    # every load down to a complete width inside those columns.
    for row in range(0, vector_end, width):
        vector_value += values.unsafe_load[width=width](left_start + row) * (
            values.unsafe_load[width=width](right_start + row)
        )
    var result = vector_value.reduce_add()
    for row in range(vector_end, matrix.rows):
        result += matrix._values[left_start + row] * matrix._values[right_start + row]
    return result


def _column_dot_scalar(matrix: _DenseMatrix, left_col: Int, right_col: Int) -> Float64:
    """Return a scalar dot product between two trusted matrix columns."""
    var left_start = left_col * matrix.rows
    var right_start = right_col * matrix.rows
    var result = 0.0
    for row in range(matrix.rows):
        result += matrix._values[left_start + row] * matrix._values[right_start + row]
    return result


struct _QrWorkspace:
    """Reusable in-place Householder QR storage for dense least squares."""

    var _matrix: _DenseMatrix
    var _rhs: List[Float64]
    var _reflector: List[Float64]
    var _permutation: List[Int]
    var _pivot_solution: List[Float64]

    def __init__(out self, rows: Int, cols: Int) raises:
        if rows < cols or cols <= 0:
            raise Error("QR workspace requires rows >= columns > 0")
        self._matrix = _DenseMatrix(rows, cols)
        self._rhs = List[Float64](length=rows, fill=0.0)
        self._reflector = List[Float64](length=rows, fill=0.0)
        self._permutation = List[Int](length=cols, fill=0)
        self._pivot_solution = List[Float64](length=cols, fill=0.0)

    def _load(mut self, matrix: _DenseMatrix, right_hand_side: List[Float64]) raises:
        if (
            matrix.rows != self._matrix.rows
            or matrix.cols != self._matrix.cols
            or len(right_hand_side) != matrix.rows
        ):
            raise Error("QR workspace dimensions do not match the system")
        for index in range(len(matrix._values)):
            self._matrix._values[index] = matrix._values[index]
        for row in range(matrix.rows):
            self._rhs[row] = right_hand_side[row]

    def _load_damped(
        mut self,
        jacobian: _DenseMatrix,
        residuals: List[Float64],
        damping: Float64,
    ) raises:
        if (
            self._matrix.rows != jacobian.rows + jacobian.cols
            or self._matrix.cols != jacobian.cols
            or len(residuals) != jacobian.rows
        ):
            raise Error("QR workspace dimensions do not match the damped system")
        if not isfinite(damping) or damping <= 0.0:
            raise Error("LM damping must be finite and positive")
        for index in range(len(self._matrix._values)):
            self._matrix._values[index] = 0.0
        for row in range(len(self._rhs)):
            self._rhs[row] = 0.0
        for col in range(jacobian.cols):
            for row in range(jacobian.rows):
                self._matrix._values[self._matrix._offset(row, col)] = jacobian._values[
                    jacobian._offset(row, col)
                ]
            # Form sqrt(damping) * max(||J_col||, sqrt(1e-15)) without
            # squaring a finite column norm or multiplying before the square
            # root. Both naive intermediates can overflow unnecessarily.
            var damping_scale = sqrt(damping)
            var augmented_value = max(
                _scaled_column_tail_norm(jacobian, col, 0, damping_scale),
                damping_scale * sqrt(1.0e-15),
            )
            if not isfinite(augmented_value):
                raise Error("damped least-squares augmentation is not finite")
            self._matrix._values[
                self._matrix._offset(jacobian.rows + col, col)
            ] = augmented_value
        for row in range(jacobian.rows):
            self._rhs[row] = -residuals[row]

    def _factor_and_solve(mut self) raises -> List[Float64]:
        var rows = self._matrix.rows
        var cols = self._matrix.cols
        var largest_initial_norm = 0.0
        for col in range(cols):
            self._permutation[col] = col
            largest_initial_norm = max(
                largest_initial_norm, _column_tail_norm(self._matrix, col, 0)
            )
        if not isfinite(largest_initial_norm) or largest_initial_norm == 0.0:
            raise Error("least-squares matrix is rank-deficient")
        for pivot in range(cols):
            var selected = pivot
            var selected_norm = _column_tail_norm(self._matrix, pivot, pivot)
            for col in range(pivot + 1, cols):
                var candidate_norm = _column_tail_norm(self._matrix, col, pivot)
                if candidate_norm > selected_norm:
                    selected = col
                    selected_norm = candidate_norm
            if (
                not isfinite(selected_norm)
                or selected_norm / largest_initial_norm <= 1.0e-14
            ):
                raise Error("least-squares matrix is rank-deficient")
            if selected != pivot:
                _swap_columns(self._matrix, pivot, selected)
                var original = self._permutation[pivot]
                self._permutation[pivot] = self._permutation[selected]
                self._permutation[selected] = original

            for row in range(rows):
                self._reflector[row] = 0.0
            var diagonal = self._matrix._values[self._matrix._offset(pivot, pivot)]
            var alpha_sign = -1.0 if diagonal >= 0.0 else 1.0
            var alpha = alpha_sign * selected_norm
            for row in range(pivot, rows):
                self._reflector[row] = (
                    self._matrix._values[self._matrix._offset(row, pivot)]
                    / selected_norm
                )
            self._reflector[pivot] -= alpha_sign
            var denominator = 0.0
            for row in range(pivot, rows):
                denominator += self._reflector[row] * self._reflector[row]
            if not isfinite(denominator) or denominator == 0.0:
                raise Error("least-squares matrix is rank-deficient")

            for col in range(pivot, cols):
                var column_scale = 0.0
                for row in range(pivot, rows):
                    column_scale = max(
                        column_scale,
                        abs(self._matrix._values[self._matrix._offset(row, col)]),
                    )
                if not isfinite(column_scale):
                    raise Error("least-squares matrix contains a non-finite value")
                if column_scale == 0.0:
                    continue
                var projection = 0.0
                for row in range(pivot, rows):
                    projection += (
                        self._reflector[row]
                        * self._matrix._values[self._matrix._offset(row, col)]
                        / column_scale
                    )
                projection *= 2.0 / denominator
                for row in range(pivot, rows):
                    var offset = self._matrix._offset(row, col)
                    self._matrix._values[offset] = column_scale * (
                        self._matrix._values[offset] / column_scale
                        - projection * self._reflector[row]
                    )

            var rhs_scale = 0.0
            for row in range(pivot, rows):
                rhs_scale = max(rhs_scale, abs(self._rhs[row]))
            if not isfinite(rhs_scale):
                raise Error("least-squares right-hand side contains a non-finite value")
            if rhs_scale != 0.0:
                var rhs_projection = 0.0
                for row in range(pivot, rows):
                    rhs_projection += self._reflector[row] * self._rhs[row] / rhs_scale
                rhs_projection *= 2.0 / denominator
                for row in range(pivot, rows):
                    self._rhs[row] = rhs_scale * (
                        self._rhs[row] / rhs_scale
                        - rhs_projection * self._reflector[row]
                    )
            self._matrix._values[self._matrix._offset(pivot, pivot)] = alpha
            for row in range(pivot + 1, rows):
                self._matrix._values[self._matrix._offset(row, pivot)] = 0.0

        for reverse_row in range(cols):
            var row = cols - reverse_row - 1
            var value = self._rhs[row]
            for col in range(row + 1, cols):
                value -= self._matrix._values[self._matrix._offset(row, col)] * (
                    self._pivot_solution[col]
                )
            var diagonal = self._matrix._values[self._matrix._offset(row, row)]
            if (
                not isfinite(diagonal)
                or abs(diagonal) / largest_initial_norm <= 1.0e-14
            ):
                raise Error("least-squares matrix is rank-deficient")
            self._pivot_solution[row] = value / diagonal

        var solution = List[Float64](length=cols, fill=0.0)
        for col in range(cols):
            var value = self._pivot_solution[col]
            if not isfinite(value):
                raise Error("least-squares solution is not finite")
            solution[self._permutation[col]] = value
        return solution^


def _solve_least_squares_qr(
    matrix: _DenseMatrix, right_hand_side: List[Float64]
) raises -> List[Float64]:
    """Solve a full-column-rank dense least-squares system by pivoted QR."""
    if matrix.rows < matrix.cols or matrix.cols <= 0:
        raise Error("least-squares solve requires rows >= columns > 0")
    if len(right_hand_side) != matrix.rows:
        raise Error("least-squares right-hand side must match matrix rows")
    var workspace = _QrWorkspace(matrix.rows, matrix.cols)
    workspace._load(matrix, right_hand_side)
    return workspace._factor_and_solve()


def _solve_damped_least_squares_qr(
    mut workspace: _QrWorkspace,
    jacobian: _DenseMatrix,
    residuals: List[Float64],
    damping: Float64,
) raises -> List[Float64]:
    """Solve ``min ||J s + r||² + damping*||D s||²`` by QR."""
    workspace._load_damped(jacobian, residuals, damping)
    return workspace._factor_and_solve()


def _column_tail_norm(matrix: _DenseMatrix, col: Int, start_row: Int) -> Float64:
    var scale = 0.0
    var scaled_sum = 1.0
    for row in range(start_row, matrix.rows):
        var magnitude = abs(matrix._values[matrix._offset(row, col)])
        if magnitude == 0.0:
            continue
        if scale < magnitude:
            var ratio = scale / magnitude
            scaled_sum = 1.0 + scaled_sum * ratio * ratio
            scale = magnitude
        else:
            var ratio = magnitude / scale
            scaled_sum += ratio * ratio
    if scale == 0.0:
        return 0.0
    return scale * sqrt(scaled_sum)


def _scaled_column_tail_norm(
    matrix: _DenseMatrix,
    col: Int,
    start_row: Int,
    multiplier: Float64,
) -> Float64:
    """Return ``multiplier * ||column tail||`` without an unscaled norm."""
    var scale = 0.0
    var scaled_sum = 1.0
    for row in range(start_row, matrix.rows):
        var magnitude = abs(matrix._values[matrix._offset(row, col)])
        if magnitude == 0.0:
            continue
        if scale < magnitude:
            var ratio = scale / magnitude
            scaled_sum = 1.0 + scaled_sum * ratio * ratio
            scale = magnitude
        else:
            var ratio = magnitude / scale
            scaled_sum += ratio * ratio
    if scale == 0.0 or multiplier == 0.0:
        return 0.0
    var scaled = multiplier * scale
    if scaled == 0.0:
        # A rounded-to-zero first product can become representable after the
        # remaining sqrt(scaled_sum) factor.
        return (multiplier * sqrt(scaled_sum)) * scale
    return scaled * sqrt(scaled_sum)


def _swap_columns(mut matrix: _DenseMatrix, left: Int, right: Int):
    for row in range(matrix.rows):
        var left_offset = matrix._offset(row, left)
        var right_offset = matrix._offset(row, right)
        var value = matrix._values[left_offset]
        matrix._values[left_offset] = matrix._values[right_offset]
        matrix._values[right_offset] = value
