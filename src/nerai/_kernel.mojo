"""Private dense ``Float64`` operations used by the least-squares solver.

Matrices use one flat row-major ``List[Float64]``. Public-to-this-module
``get()`` and ``set()`` calls check indices and raise on invalid input. Kernel
loops first check argument shapes, then index the trusted flat buffers directly.
"""

from std.math import sqrt
from std.utils.numerics import isfinite


struct _DenseMatrix(Copyable):
    """A private dense row-major matrix value."""

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
        return self._values[row * self.cols + col]

    def set(mut self, row: Int, col: Int, value: Float64) raises:
        """Set one element, raising when either index is out of range."""
        self._check_index(row, col)
        self._values[row * self.cols + col] = value

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
    for col in range(jacobian.cols):
        var value = 0.0
        for row in range(jacobian.rows):
            value += jacobian._values[row * jacobian.cols + col] * residuals[row]
        result[col] = value
    return result^


def _jt_j(jacobian: _DenseMatrix) raises -> _DenseMatrix:
    """Return the symmetric normal matrix ``J^T J``."""
    var result = _DenseMatrix(jacobian.cols, jacobian.cols)
    for left_col in range(jacobian.cols):
        for right_col in range(left_col, jacobian.cols):
            var value = 0.0
            for row in range(jacobian.rows):
                value += (
                    jacobian._values[row * jacobian.cols + left_col]
                    * jacobian._values[row * jacobian.cols + right_col]
                )
            result._values[left_col * result.cols + right_col] = value
            result._values[right_col * result.cols + left_col] = value
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
        var offset = index * matrix.cols + index
        matrix._values[offset] += damping * diagonal[index]


def _solve_spd(
    matrix: _DenseMatrix, right_hand_side: List[Float64]
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

    var dimension = matrix.rows
    var lower = _DenseMatrix(dimension, dimension)
    for row in range(dimension):
        for col in range(row + 1):
            var value = matrix._values[row * matrix.cols + col]
            for inner in range(col):
                value -= (
                    lower._values[row * dimension + inner]
                    * lower._values[col * dimension + inner]
                )

            if row == col:
                if not isfinite(value) or value <= 0.0:
                    raise Error("matrix is singular or not positive definite")
                lower._values[row * dimension + col] = sqrt(value)
            else:
                lower._values[row * dimension + col] = (
                    value / lower._values[col * dimension + col]
                )

    var intermediate = List[Float64](length=dimension, fill=0.0)
    for row in range(dimension):
        var value = right_hand_side[row]
        for col in range(row):
            value -= lower._values[row * dimension + col] * intermediate[col]
        intermediate[row] = value / lower._values[row * dimension + row]

    var solution = List[Float64](length=dimension, fill=0.0)
    for reverse_row in range(dimension):
        var row = dimension - reverse_row - 1
        var value = intermediate[row]
        for col in range(row + 1, dimension):
            value -= lower._values[col * dimension + row] * solution[col]
        solution[row] = value / lower._values[row * dimension + row]
    return solution^
