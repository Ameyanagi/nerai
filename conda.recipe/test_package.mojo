from nerai import (
    Bounds,
    CurveFit,
    CurveModel,
    LeastSquaresResult,
    LeastSquaresOptions,
    LeastSquaresProblem,
    LossKind,
    ResidualModel,
    TerminationReason,
    evaluate_loss,
    least_squares,
    robust_cost,
)
from std.collections import List
from std.math import abs
from std.testing import assert_false, assert_true


struct InstalledCurve(CurveModel):
    def __init__(out self):
        pass

    def values(
        mut self, parameters: List[Float64], t: Span[Float64, ...]
    ) raises -> List[Float64]:
        var values = List[Float64](length=len(t), fill=0.0)
        for index in range(len(t)):
            values[index] = parameters[0] * t[index] + parameters[1]
        return values^


struct InstalledModel(ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [parameters[0] - 1.0, parameters[1] + 2.0]


def main() raises:
    var evaluation = evaluate_loss(LossKind.HUBER, 4.0)
    assert_true(evaluation.value == 3.0)
    assert_true(robust_cost(LossKind.HUBER, 4.0, scale=2.0) == 6.0)
    assert_false(TerminationReason.MAX_EVALUATIONS.is_success())

    var result = LeastSquaresResult(
        [1.0],
        cost=0.0,
        optimality=0.0,
        iterations=0,
        residual_evaluations=1,
        jacobian_evaluations=1,
        termination=TerminationReason.GRADIENT_TOLERANCE,
    )
    result.validate()
    assert_true(result.converged())

    var problem = LeastSquaresProblem(
        InstalledModel(),
        [1.0, -2.0],
        weights=[1.0, 0.5],
        options=LeastSquaresOptions(loss=LossKind.SOFT_L1),
    )
    var residuals = problem.evaluate_initial_residuals()
    assert_true(residuals[0] == 0.0)
    assert_true(residuals[1] == 0.0)
    assert_true(problem.model.calls == 1)

    var fitted = least_squares(problem)
    assert_true(fitted.converged())
    assert_true(fitted.parameters[0] == 1.0)
    assert_true(fitted.parameters[1] == -2.0)

    var t: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var y: List[Float64] = [1.0, 3.0, 5.0, 7.0]
    var curve = CurveFit(
        InstalledCurve(),
        t,
        y,
        [1.5, 0.5],
        parameter_names=["slope", "intercept"],
        bounds=Bounds.nonnegative(2),
    )
    var curve_result = curve.solve()
    assert_true(curve_result.result.converged())
    assert_true(abs(curve_result.parameter("slope") - 2.0) <= 1.0e-8)
    assert_true(abs(curve_result.parameter("intercept") - 1.0) <= 1.0e-8)
    assert_true(len(curve_result.residuals()) == 4)
