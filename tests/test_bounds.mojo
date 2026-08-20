from nerai import (
    Bounds,
    LeastSquaresProblem,
    ResidualModel,
    least_squares,
)
from std.collections import List
from std.math import abs, exp
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)
from std.utils.numerics import inf, nan


struct RecordingBoundsModel(Copyable, ResidualModel):
    var recorded: List[List[Float64]]

    def __init__(out self):
        self.recorded = List[List[Float64]]()

    def residual_count(self) -> Int:
        return 12

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.recorded.append(parameters.copy())
        var t = [
            0.0,
            0.36363636363636365,
            0.7272727272727273,
            1.0909090909090908,
            1.4545454545454546,
            1.8181818181818183,
            2.1818181818181817,
            2.5454545454545454,
            2.909090909090909,
            3.272727272727273,
            3.6363636363636367,
            4.0,
        ]
        var y = [
            0.5006838553450637,
            0.5417404053516538,
            0.5535853306626277,
            0.5334302221006102,
            0.5522224279596892,
            0.5621795888666042,
            0.5986672544241665,
            0.6006968930372695,
            0.6313013486887672,
            0.5939625949296087,
            0.676785520948536,
            0.6580713567968877,
        ]
        var residuals = List[Float64](length=12, fill=0.0)
        for index in range(12):
            residuals[index] = parameters[0] * exp(-parameters[1] * t[index]) - y[index]
        return residuals^


def positive_bounds() raises -> Bounds:
    return Bounds(
        [0.0, 0.0],
        [inf[DType.float64](), inf[DType.float64]()],
    )


def test_bounds_validate_endpoints_and_dimensions() raises:
    with assert_raises(contains="at least one parameter"):
        _ = Bounds(List[Float64](), List[Float64]())
    with assert_raises(contains="1 lower entries and 2 upper entries"):
        _ = Bounds([0.0], [1.0, 2.0])
    with assert_raises(contains="bounds[1]"):
        _ = Bounds([0.0, nan[DType.float64]()], [1.0, 2.0])
    with assert_raises(contains="bounds[0]"):
        _ = Bounds([0.0], [nan[DType.float64]()])
    with assert_raises(contains="bounds[1] are empty"):
        _ = Bounds([-1.0, 3.0], [1.0, 3.0])
    with assert_raises(contains="lower 4.0 must be strictly below upper 3.0"):
        _ = Bounds([4.0], [3.0])
    with assert_raises(contains="bounds[0] are empty"):
        _ = Bounds([inf[DType.float64]()], [inf[DType.float64]()])
    with assert_raises(contains="bounds[0] are empty"):
        _ = Bounds([-inf[DType.float64]()], [-inf[DType.float64]()])

    var unbounded = Bounds([-inf[DType.float64]()], [inf[DType.float64]()])
    unbounded.validate()
    assert_true(unbounded.contains([0.0]))


def test_contains_uses_closed_intervals_and_checks_length() raises:
    var bounds = Bounds([0.0, -2.0], [1.0, 3.0])
    assert_true(bounds.contains([0.0, -2.0]))
    assert_true(bounds.contains([1.0, 3.0]))
    assert_true(bounds.contains([0.25, 2.0]))
    assert_false(bounds.contains([-1.0, 0.0]))
    assert_false(bounds.contains([0.5]))
    assert_false(bounds.contains([nan[DType.float64](), 0.0]))


def test_bounds_equality_accessors_and_writable_block() raises:
    var first = Bounds([0.0, -inf[DType.float64]()], [inf[DType.float64](), 3.0])
    var same = Bounds([0.0, -inf[DType.float64]()], [inf[DType.float64](), 3.0])
    var different = Bounds([0.0, -2.0], [inf[DType.float64](), 3.0])

    assert_true(first == same)
    assert_true(first != different)
    assert_equal(first.parameter_count(), 2)
    assert_equal(first.lower(0), 0.0)
    assert_equal(first.upper(1), 3.0)
    assert_equal(
        String(first),
        String(
            "bounds[0]             [0.0, inf]\n",
            "bounds[1]             [-inf, 3.0]\n",
        ),
    )


def test_problem_requires_matching_strictly_feasible_initial_values() raises:
    with assert_raises(contains="bounds have 1 parameters for 2 initial"):
        _ = LeastSquaresProblem(
            RecordingBoundsModel(),
            [0.5, 0.1],
            bounds=Bounds([0.0], [1.0]),
        )
    with assert_raises(contains="initial parameter[1] 0.0"):
        _ = LeastSquaresProblem(
            RecordingBoundsModel(),
            [0.5, 0.0],
            bounds=positive_bounds(),
        )


def test_bounded_fit_is_strictly_feasible_and_reports_active_rate() raises:
    var unbounded_problem = LeastSquaresProblem(RecordingBoundsModel(), [0.5, 0.1])
    var unbounded = least_squares(unbounded_problem)
    assert_true(unbounded.parameters[1] < 0.0)
    assert_true(abs(unbounded.parameters[0] - 0.51122) <= 1.0e-3)
    assert_true(abs(unbounded.parameters[1] + 0.06457) <= 1.0e-3)

    var bounded_problem = LeastSquaresProblem(
        RecordingBoundsModel(),
        [0.5, 0.1],
        bounds=positive_bounds(),
    )
    var bounded = least_squares(bounded_problem)

    # Trial and accepted iterates are strictly feasible. This forward scheme's
    # finite differences perturb only upward, so every recorded callback is
    # also strictly feasible for these lower-only bounds.
    for index in range(len(bounded_problem.model.recorded)):
        assert_true(bounded_problem.model.recorded[index][0] > 0.0)
        assert_true(bounded_problem.model.recorded[index][1] > 0.0)
    assert_true(bounded.parameters[1] >= 0.0)
    assert_true(bounded.parameters[1] <= 1.0e-6)
    assert_true(abs(bounded.parameters[0] - 0.5836105665926237) <= 1.0e-3)
    assert_equal(bounded.active_bounds, [0, -1])
    assert_true(String(bounded).endswith("active bounds[1]      lower\n"))


def test_bounded_solves_are_exactly_deterministic() raises:
    var first_problem = LeastSquaresProblem(
        RecordingBoundsModel(),
        [0.5, 0.1],
        bounds=positive_bounds(),
    )
    var second_problem = LeastSquaresProblem(
        RecordingBoundsModel(),
        [0.5, 0.1],
        bounds=positive_bounds(),
    )
    var first = least_squares(first_problem)
    var second = least_squares(second_problem)

    assert_equal(first.iterations, second.iterations)
    assert_true(first.parameters[0] == second.parameters[0])
    assert_true(first.parameters[1] == second.parameters[1])
    assert_equal(first.active_bounds, second.active_bounds)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
