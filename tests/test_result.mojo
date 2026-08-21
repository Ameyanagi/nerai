from nerai import LeastSquaresResult, TerminationReason
from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)
from std.utils.numerics import nan


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
    assert_equal(result.active_bounds, [0, 0])


def test_result_rejects_invalid_reports() raises:
    with assert_raises(
        contains="result parameters must contain at least one parameter; got 0"
    ):
        _ = LeastSquaresResult(
            List[Float64](),
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_ITERATIONS,
        )
    with assert_raises(contains="result parameters[0] must be finite; got nan"):
        _ = LeastSquaresResult(
            [nan[DType.float64]()],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.NUMERICAL_FAILURE,
        )
    with assert_raises(
        contains="result cost must be finite and non-negative; got -1.0"
    ):
        _ = LeastSquaresResult(
            [1.0],
            cost=-1.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.NUMERICAL_FAILURE,
        )
    with assert_raises(
        contains="result optimality must be finite and non-negative; got -1.0"
    ):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=-1.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.NUMERICAL_FAILURE,
        )
    with assert_raises(contains="iterations must be non-negative; got -1"):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=-1,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_ITERATIONS,
        )
    with assert_raises(contains="jacobian_evaluations must be non-negative; got -1"):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=-1,
            termination=TerminationReason.MAX_EVALUATIONS,
        )
    with assert_raises(contains="residual_evaluations must be at least 1; got 0"):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=0,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_EVALUATIONS,
        )
    with assert_raises(
        contains=(
            "iterations 1 cannot exceed completed trials 0 (residual_evaluations - 1)"
        )
    ):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=1,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_ITERATIONS,
        )
    with assert_raises(
        contains="jacobian_evaluations 2 cannot exceed residual_evaluations 1"
    ):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=2,
            termination=TerminationReason.MAX_EVALUATIONS,
        )
    with assert_raises(
        contains="active_bounds count 2 must equal result parameters count 1"
    ):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_EVALUATIONS,
            active_bounds=Optional[List[Int]]([0, 1]),
        )
    with assert_raises(contains="active_bounds[0]"):
        _ = LeastSquaresResult(
            [1.0],
            cost=0.0,
            optimality=0.0,
            iterations=0,
            residual_evaluations=1,
            jacobian_evaluations=0,
            termination=TerminationReason.MAX_EVALUATIONS,
            active_bounds=Optional[List[Int]]([2]),
        )


def test_termination_reasons_are_distinct_nominal_values() raises:
    assert_true(
        TerminationReason.GRADIENT_TOLERANCE != TerminationReason.STEP_TOLERANCE
    )
    assert_true(TerminationReason.COST_TOLERANCE != TerminationReason.MAX_ITERATIONS)
    assert_true(
        TerminationReason.MAX_EVALUATIONS != TerminationReason.NUMERICAL_FAILURE
    )


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
    with assert_raises(
        contains="result cost must be finite and non-negative; got -1.0"
    ):
        result.validate()

    result.termination = TerminationReason.NUMERICAL_FAILURE
    assert_true(result.termination.is_failure())
    assert_false(result.converged())


def test_result_equality_compares_complete_public_reports() raises:
    var first = LeastSquaresResult(
        [1.0, -2.0],
        cost=0.125,
        optimality=0.25,
        iterations=2,
        residual_evaluations=5,
        jacobian_evaluations=2,
        termination=TerminationReason.GRADIENT_TOLERANCE,
    )
    var same = LeastSquaresResult(
        [1.0, -2.0],
        cost=0.125,
        optimality=0.25,
        iterations=2,
        residual_evaluations=5,
        jacobian_evaluations=2,
        termination=TerminationReason.GRADIENT_TOLERANCE,
    )
    assert_true(first == same)

    same.parameters[1] = -3.0
    assert_false(first == same)
    same.parameters[1] = -2.0
    same.active_bounds[1] = -1
    assert_false(first == same)
    same.active_bounds[1] = 0
    same.termination = TerminationReason.STEP_TOLERANCE
    assert_true(first != same)


def test_active_bounds_write_after_parameters_only_when_nonzero() raises:
    var result = LeastSquaresResult(
        [1.0, 2.0],
        cost=0.0,
        optimality=0.0,
        iterations=0,
        residual_evaluations=1,
        jacobian_evaluations=0,
        termination=TerminationReason.GRADIENT_TOLERANCE,
        active_bounds=Optional[List[Int]]([-1, 1]),
    )
    assert_true(
        String(result).endswith(
            "active bounds[0]      lower\nactive bounds[1]      upper\n"
        )
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
