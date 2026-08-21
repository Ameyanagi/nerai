from nerai import LossKind
from nerai._kernel import _DenseMatrix
from nerai._objective import _build_objective_model, _objective_cost
from std.math import abs, sqrt
from std.testing import TestSuite, assert_true


def assert_close(
    actual: Float64, expected: Float64, tolerance: Float64 = 1.0e-12
) raises:
    assert_true(abs(actual - expected) <= tolerance)


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


def test_linear_matches_direct_weighted_least_squares() raises:
    # u = w*r = [1, -6, 0], so F = (1^2 + (-6)^2) / 2 = 18.5.
    # J_model = diag(w) J_raw and J_model^T u = [12.5, -5].
    var raw_jacobian = matrix_from_values(3, 2, [1.0, 2.0, -1.0, 0.5, 5.0, -2.0])
    var objective = _build_objective_model(
        [2.0, -3.0, 4.0],
        raw_jacobian,
        [0.5, 2.0, 0.0],
        LossKind.LINEAR,
        7.0,
    )

    assert_close(objective.cost, 18.5)
    assert_close(objective.residuals[0], 1.0)
    assert_close(objective.residuals[1], -6.0)
    assert_close(objective.residuals[2], 0.0)
    assert_close(objective.jacobian.get(0, 0), 0.5)
    assert_close(objective.jacobian.get(0, 1), 1.0)
    assert_close(objective.jacobian.get(1, 0), -2.0)
    assert_close(objective.jacobian.get(1, 1), 1.0)
    assert_close(objective.jacobian.get(2, 0), 0.0)
    assert_close(objective.jacobian.get(2, 1), 0.0)
    assert_close(objective.gradient[0], 12.5)
    assert_close(objective.gradient[1], -5.0)
    assert_close(objective.optimality, 12.5)


def test_zero_weight_row_contributes_nothing() raises:
    var raw_jacobian = matrix_from_values(2, 1, [3.0, 1.0e100])
    var objective = _build_objective_model(
        [2.0, -1.0e100],
        raw_jacobian,
        [1.0, 0.0],
        LossKind.SOFT_L1,
        1.0,
    )

    # Only u=2 remains: F=sqrt(1+2^2)-1 and rho'=1/sqrt(5).
    assert_close(objective.cost, sqrt(5.0) - 1.0)
    assert_close(objective.residuals[1], 0.0)
    assert_close(objective.jacobian.get(1, 0), 0.0)


def test_huber_and_soft_l1_match_independent_fixtures() raises:
    var identity = matrix_from_values(2, 2, [1.0, 0.0, 0.0, 1.0])

    # Huber: u=[2, 2], C=1. Each outlier contributes
    # C*(abs(u)-C/2)=1.5 and rho'=C/abs(u)=1/2.
    var huber = _build_objective_model(
        [1.0, 4.0], identity, [2.0, 0.5], LossKind.HUBER, 1.0
    )
    assert_close(huber.cost, 3.0)
    assert_close(huber.residuals[0], sqrt(2.0))
    assert_close(huber.residuals[1], sqrt(2.0))
    assert_close(huber.gradient[0], 2.0)
    assert_close(huber.gradient[1], 0.5)

    # Soft-L1: u=[3, -2], C=2. Directly from
    # C^2*(sqrt(1+(u/C)^2)-1), the total is
    # 2*sqrt(13) + 4*sqrt(2) - 8.
    var soft_l1 = _build_objective_model(
        [1.5, -4.0], identity, [2.0, 0.5], LossKind.SOFT_L1, 2.0
    )
    assert_close(soft_l1.cost, 2.0 * sqrt(13.0) + 4.0 * sqrt(2.0) - 8.0)


def affine_residuals(parameters: List[Float64]) -> List[Float64]:
    # r(x) = A*x-b for A=[[1,2],[-3,0.5],[0.25,-1]], b=[1,-2,0.5].
    return [
        parameters[0] + 2.0 * parameters[1] - 1.0,
        -3.0 * parameters[0] + 0.5 * parameters[1] + 2.0,
        0.25 * parameters[0] - parameters[1] - 0.5,
    ]


def check_scalar_gradient(loss: LossKind) raises:
    var parameters: List[Float64] = [0.4, -0.7]
    var weights: List[Float64] = [1.0, 0.5, 2.0]
    var raw_jacobian = matrix_from_values(3, 2, [1.0, 2.0, -3.0, 0.5, 0.25, -1.0])
    var objective = _build_objective_model(
        affine_residuals(parameters), raw_jacobian, weights, loss, 0.75
    )

    # Independent central differences of scalar F use h=1e-6. The point is
    # away from Huber's threshold, so the O(h^2) check is smooth and fixed.
    var step = 1.0e-6
    for col in range(2):
        var plus = parameters.copy()
        var minus = parameters.copy()
        plus[col] += step
        minus[col] -= step
        var finite_difference = (
            _objective_cost(affine_residuals(plus), weights, loss, 0.75)
            - _objective_cost(affine_residuals(minus), weights, loss, 0.75)
        ) / (2.0 * step)
        assert_close(objective.gradient[col], finite_difference, 2.0e-9)


def test_model_gradient_matches_scalar_finite_differences() raises:
    check_scalar_gradient(LossKind.HUBER)
    check_scalar_gradient(LossKind.SOFT_L1)


def test_residual_sign_changes_preserve_robust_cost() raises:
    var positive: List[Float64] = [0.25, 2.0, 7.0]
    var negative: List[Float64] = [-0.25, -2.0, -7.0]
    var weights: List[Float64] = [1.0, 0.5, 2.0]
    assert_close(
        _objective_cost(positive, weights, LossKind.HUBER, 1.5),
        _objective_cost(negative, weights, LossKind.HUBER, 1.5),
    )
    assert_close(
        _objective_cost(positive, weights, LossKind.SOFT_L1, 1.5),
        _objective_cost(negative, weights, LossKind.SOFT_L1, 1.5),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
