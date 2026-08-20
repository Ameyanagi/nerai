from nerai import (
    LeastSquaresResult,
    LossKind,
    TerminationReason,
    evaluate_loss,
    robust_cost,
)
from std.testing import assert_false, assert_true


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
