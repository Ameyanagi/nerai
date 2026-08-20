from nerai import LeastSquaresResult, TerminationReason
from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)


def test_termination_categories_are_explicit() raises:
    assert_true(TerminationReason.GRADIENT_TOLERANCE.is_success())
    assert_true(TerminationReason.STEP_TOLERANCE.is_success())
    assert_true(TerminationReason.COST_TOLERANCE.is_success())
    assert_false(TerminationReason.MAX_ITERATIONS.is_success())
    assert_false(TerminationReason.MAX_EVALUATIONS.is_success())
    assert_false(TerminationReason.NUMERICAL_FAILURE.is_success())

    assert_true(TerminationReason.MAX_ITERATIONS.is_limit())
    assert_true(TerminationReason.MAX_EVALUATIONS.is_limit())
    assert_true(TerminationReason.NUMERICAL_FAILURE.is_failure())


def test_result_preserves_solver_report() raises:
    var result = LeastSquaresResult(
        [1.25, -0.5],
        cost=0.125,
        optimality=1e-9,
        iterations=4,
        residual_evaluations=11,
        jacobian_evaluations=4,
        termination=TerminationReason.GRADIENT_TOLERANCE,
    )

    assert_equal(len(result.parameters), 2)
    assert_true(result.parameters[0] == 1.25)
    assert_true(result.parameters[1] == -0.5)
    assert_true(result.cost == 0.125)
    assert_true(result.optimality == 1e-9)
    assert_equal(result.iterations, 4)
    assert_equal(result.residual_evaluations, 11)
    assert_equal(result.jacobian_evaluations, 4)
    assert_true(result.termination == TerminationReason.GRADIENT_TOLERANCE)
    assert_true(result.converged())


def test_result_rejects_invalid_reports() raises:
    with assert_raises(contains="at least one parameter"):
        _ = LeastSquaresResult(
            List[Float64](),
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_ITERATIONS,
        )
    with assert_raises(contains="cost must be finite and non-negative"):
        _ = LeastSquaresResult(
            [1.0],
            cost=-1.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.NUMERICAL_FAILURE,
        )
    with assert_raises(contains="Jacobian evaluation count must be non-negative"):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=-1,
            termination=TerminationReason.MAX_EVALUATIONS,
        )
    with assert_raises(contains="at least one residual evaluation"):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=0,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_EVALUATIONS,
        )
    with assert_raises(contains="iteration count cannot exceed completed trials"):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=1,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_ITERATIONS,
        )
    with assert_raises(contains="Jacobian evaluations cannot exceed residual"):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=2,
            termination=TerminationReason.MAX_EVALUATIONS,
        )


def test_every_reachable_termination_mutation_has_defined_semantics() raises:
    var reason = TerminationReason.GRADIENT_TOLERANCE

    reason._stopped = False
    reason._detail = None
    assert_true(reason == TerminationReason.GRADIENT_TOLERANCE)
    assert_true(reason.is_success())

    reason._detail = False
    assert_true(reason == TerminationReason.STEP_TOLERANCE)
    assert_true(reason.is_success())

    reason._detail = True
    assert_true(reason == TerminationReason.COST_TOLERANCE)
    assert_true(reason.is_success())

    reason._stopped = True
    reason._detail = None
    assert_true(reason == TerminationReason.MAX_ITERATIONS)
    assert_true(reason.is_limit())

    reason._detail = False
    assert_true(reason == TerminationReason.MAX_EVALUATIONS)
    assert_true(reason.is_limit())

    reason._detail = True
    assert_true(reason == TerminationReason.NUMERICAL_FAILURE)
    assert_true(reason.is_failure())


def test_mutated_numeric_report_can_be_revalidated() raises:
    var result = LeastSquaresResult(
        [1.0],
        cost=0.0,
        optimality=0.0,
        iterations=0,
        residual_evaluations=1,
        jacobian_evaluations=0,
        termination=TerminationReason.MAX_ITERATIONS,
    )
    result.validate()

    result.cost = -1.0
    with assert_raises(contains="result cost must be finite and non-negative"):
        result.validate()

    # Termination remains total even while the numeric snapshot is invalid.
    result.termination._stopped = True
    result.termination._detail = True
    assert_true(result.termination.is_failure())
    assert_false(result.converged())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
