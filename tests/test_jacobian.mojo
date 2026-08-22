from nerai import Bounds, ResidualModel
from nerai._jacobian import (
    _central_difference_jacobian,
    _forward_difference_jacobian,
    _forward_difference_jacobian_without_base,
)
from std.math import abs, cos, exp, sin
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.utils.numerics import inf


struct ConstantModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [3.0, -2.0]


struct AffineModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 3

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [
            2.0 * parameters[0] - 3.0 * parameters[1] + 1.0,
            -parameters[0] + 0.5 * parameters[1] - 2.0,
            4.0 * parameters[1] + 7.0,
        ]


struct QuadraticModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [parameters[0] * parameters[0], parameters[1] * parameters[1]]


struct CubicModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 1

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [parameters[0] * parameters[0] * parameters[0] - 8.0]


struct CoupledNonlinearModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var x = parameters[0]
        var y = parameters[1]
        return [sin(x * y), exp(x) + x * y * y]


struct ChangingLengthModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        if self.calls == 1:
            return [parameters[0], parameters[1]]
        return [parameters[0], parameters[1], 0.0]


struct NonfiniteModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [parameters[0], inf[DType.float64]()]


struct RaisingModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        raise Error("Jacobian model failure")


struct BoundedDomainModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 1

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        if parameters[0] <= 0.0 or parameters[0] >= 1.0:
            raise Error("bounded finite difference escaped the open domain")
        return [parameters[0] * parameters[0]]


def assert_close(
    actual: Float64, expected: Float64, tolerance: Float64 = 1.0e-12
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def assert_jacobian_gate(actual: Float64, expected: Float64) raises:
    # NERAI-004 numerical gate: 1e-6 relative or 1e-8 absolute error.
    var error = abs(actual - expected)
    assert_true(error <= 1.0e-8 or error <= 1.0e-6 * abs(expected))


def test_constant_residual_has_exact_zero_columns_and_n_calls() raises:
    var model = ConstantModel()
    var evaluations = 0
    var jacobian = _forward_difference_jacobian(
        model, [2.0, -4.0], [3.0, -2.0], evaluations
    )

    assert_equal(jacobian.rows, 2)
    assert_equal(jacobian.cols, 2)
    assert_close(jacobian.get(0, 0), 0.0)
    assert_close(jacobian.get(0, 1), 0.0)
    assert_close(jacobian.get(1, 0), 0.0)
    assert_close(jacobian.get(1, 1), 0.0)
    assert_equal(model.calls, 2)
    assert_equal(evaluations, 2)


def test_affine_jacobian_matches_exact_hand_derivatives() raises:
    # At x = [2, -1], r = [8, -4.5, 3] and
    # dr/dx = [[2, -3], [-1, 0.5], [0, 4]]. A binary-exact 2^-10
    # relative step isolates floating-point rounding; tolerance is 1e-12.
    var model = AffineModel()
    var evaluations = 0
    var jacobian = _forward_difference_jacobian(
        model,
        [2.0, -1.0],
        [8.0, -4.5, 3.0],
        evaluations,
        relative_step=1.0 / 1024.0,
    )

    assert_close(jacobian.get(0, 0), 2.0, 1.0e-12)
    assert_close(jacobian.get(0, 1), -3.0, 1.0e-12)
    assert_close(jacobian.get(1, 0), -1.0, 1.0e-12)
    assert_close(jacobian.get(1, 1), 0.5, 1.0e-12)
    assert_close(jacobian.get(2, 0), 0.0, 1.0e-12)
    assert_close(jacobian.get(2, 1), 4.0, 1.0e-12)
    assert_equal(model.calls, 2)
    assert_equal(evaluations, 2)


def test_quadratic_jacobian_meets_documented_absolute_tolerance() raises:
    # For r = [x^2, y^2] at [1.5, -2], J = [[3, 0], [0, -4]].
    # Forward truncation is O(h); each entry uses a 1e-7 absolute tolerance.
    var model = QuadraticModel()
    var evaluations = 0
    var jacobian = _forward_difference_jacobian(
        model, [1.5, -2.0], [2.25, 4.0], evaluations
    )

    assert_close(jacobian.get(0, 0), 3.0, 1.0e-7)
    assert_close(jacobian.get(0, 1), 0.0, 1.0e-7)
    assert_close(jacobian.get(1, 0), 0.0, 1.0e-7)
    assert_close(jacobian.get(1, 1), -4.0, 1.0e-7)


def test_coupled_nonlinear_jacobian_meets_numerical_gate() raises:
    # r = [sin(x*y), exp(x) + x*y^2] at x=0.75, y=-0.5.
    # J = [[cos(xy)*y, cos(xy)*x], [exp(x)+y^2, 2xy]].
    # Each entry uses the plan gate: 1e-6 relative or 1e-8 absolute.
    var x = 0.75
    var y = -0.5
    var model = CoupledNonlinearModel()
    var evaluations = 0
    var jacobian = _forward_difference_jacobian(
        model,
        [x, y],
        [sin(x * y), exp(x) + x * y * y],
        evaluations,
    )

    assert_jacobian_gate(jacobian.get(0, 0), cos(x * y) * y)
    assert_jacobian_gate(jacobian.get(0, 1), cos(x * y) * x)
    assert_jacobian_gate(jacobian.get(1, 0), exp(x) + y * y)
    assert_jacobian_gate(jacobian.get(1, 1), 2.0 * x * y)


def test_central_difference_is_more_accurate_for_a_cubic() raises:
    var forward_model = CubicModel()
    var central_model = CubicModel()
    var forward_evaluations = 0
    var central_evaluations = 0
    var forward = _forward_difference_jacobian(
        forward_model,
        [2.0],
        [0.0],
        forward_evaluations,
        relative_step=1.0e-3,
    )
    var central = _central_difference_jacobian(
        central_model,
        [2.0],
        [0.0],
        central_evaluations,
        relative_step=1.0e-3,
    )

    assert_true(abs(central.get(0, 0) - 12.0) < abs(forward.get(0, 0) - 12.0))
    assert_equal(forward_evaluations, 1)
    assert_equal(central_evaluations, 2)


def test_bounded_differences_stay_inside_and_switch_direction() raises:
    var parameter = 1.0 - 1.0e-12
    var base = parameter * parameter
    var bounds = Bounds([0.0], [1.0])
    var forward_model = BoundedDomainModel()
    var central_model = BoundedDomainModel()
    var forward_evaluations = 0
    var central_evaluations = 0
    var forward = _forward_difference_jacobian(
        forward_model,
        [parameter],
        [base],
        forward_evaluations,
        relative_step=1.0e-3,
        bounds=bounds.copy(),
    )
    var central = _central_difference_jacobian(
        central_model,
        [parameter],
        [base],
        central_evaluations,
        relative_step=1.0e-3,
        bounds=bounds.copy(),
    )

    assert_true(abs(forward.get(0, 0) - 2.0 * parameter) <= 1.1e-3)
    assert_true(abs(central.get(0, 0) - 2.0 * parameter) <= 1.0e-10)
    assert_equal(forward_model.calls, 1)
    assert_equal(central_model.calls, 2)
    assert_equal(forward_evaluations, 1)
    assert_equal(central_evaluations, 2)


def test_forward_difference_accepts_a_signed_inward_step_one_ulp_below_upper() raises:
    # The representable predecessor of 1.0 is strictly feasible, but half of
    # its one-ULP gap rounds onto the upper bound. Forward stepping must switch
    # to a negative inward perturbation instead of rejecting its sign.
    var parameter = bitcast[DType.float64](bitcast[DType.uint64](1.0) - UInt64(1))
    var model = BoundedDomainModel()
    var evaluations = 0
    var jacobian = _forward_difference_jacobian(
        model,
        [parameter],
        [parameter * parameter],
        evaluations,
        bounds=Bounds([0.0], [1.0]),
    )

    assert_true(abs(jacobian.get(0, 0) - 2.0 * parameter) <= 2.0e-8)
    assert_equal(model.calls, 1)
    assert_equal(evaluations, 1)


def test_without_base_accounts_for_n_plus_one_calls() raises:
    var model = ConstantModel()
    var evaluations = 0
    var jacobian = _forward_difference_jacobian_without_base(
        model, [2.0, -4.0], evaluations
    )

    assert_equal(jacobian.rows, 2)
    assert_equal(jacobian.cols, 2)
    assert_equal(model.calls, 3)
    assert_equal(evaluations, 3)


def test_changed_residual_length_mid_jacobian_raises() raises:
    var model = ChangingLengthModel()
    var evaluations = 0
    with assert_raises(contains="unexpected residual count"):
        _ = _forward_difference_jacobian(model, [1.0, 2.0], [1.0, 2.0], evaluations)
    assert_equal(model.calls, 2)
    assert_equal(evaluations, 2)


def test_nonfinite_callback_residual_raises() raises:
    var model = NonfiniteModel()
    var evaluations = 0
    with assert_raises(contains="model residuals must be finite"):
        _ = _forward_difference_jacobian(model, [1.0, 2.0], [1.0, 2.0], evaluations)
    assert_equal(evaluations, 1)


def test_model_error_propagates_unchanged() raises:
    var model = RaisingModel()
    var evaluations = 0
    with assert_raises(contains="Jacobian model failure"):
        _ = _forward_difference_jacobian(model, [1.0, 2.0], [1.0, 2.0], evaluations)
    assert_equal(model.calls, 1)
    assert_equal(evaluations, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
