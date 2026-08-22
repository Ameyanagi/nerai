from nerai import (
    Bounds,
    FitReport,
    FitStatistics,
    JacobianScheme,
    LeastSquaresOptions,
    LeastSquaresProblem,
    LeastSquaresResult,
    ResidualModel,
    TerminationReason,
    fit_statistics,
    least_squares,
)
from std.collections import List
from std.math import abs, exp, sqrt
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.utils.numerics import inf


struct ScipyDecayModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 25

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        var t = [
            0.0,
            0.20833333333333334,
            0.4166666666666667,
            0.625,
            0.8333333333333334,
            1.0416666666666667,
            1.25,
            1.4583333333333335,
            1.6666666666666667,
            1.875,
            2.0833333333333335,
            2.291666666666667,
            2.5,
            2.7083333333333335,
            2.916666666666667,
            3.125,
            3.3333333333333335,
            3.541666666666667,
            3.75,
            3.9583333333333335,
            4.166666666666667,
            4.375,
            4.583333333333334,
            4.791666666666667,
            5.0,
        ]
        var y = [
            2.5152358539877215,
            2.1087551483663853,
            1.9050663105664043,
            1.6611495518892907,
            1.2975361049924259,
            1.1406678953850795,
            1.0485470693546353,
            0.8849242942863328,
            0.7776680019112797,
            0.6302136754442806,
            0.6255290435363732,
            0.5415321783848346,
            0.43773639350417365,
            0.4318448652148168,
            0.34790590925119713,
            0.23752760315669744,
            0.26086745886513757,
            0.16158847170632618,
            0.22502190765099225,
            0.1540282248418524,
            0.126041297379791,
            0.08288007873970038,
            0.16218689287408977,
            0.07961970944963043,
            0.05407706744764089,
        ]
        var residuals = List[Float64](length=25, fill=0.0)
        for index in range(25):
            residuals[index] = parameters[0] * exp(-parameters[1] * t[index]) - y[index]
        return residuals^


struct EqualDimensionsModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [parameters[0] - 1.0, parameters[1] + 1.0]


struct RankDeficientModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 4

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var combined = parameters[0] + parameters[1]
        return [
            combined - 1.0,
            2.0 * combined - 2.0,
            3.0 * combined - 3.0,
            4.0 * combined - 4.0,
        ]


struct ExactFitStatisticsModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 4

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [
            parameters[0] - 2.0,
            2.0 * parameters[0] - 4.0,
            3.0 * parameters[0] - 6.0,
            4.0 * parameters[0] - 8.0,
        ]


struct UpperEdgeStatisticsModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 4

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [
            parameters[0] - 0.1,
            2.0 * parameters[0] - 0.3,
            3.0 * parameters[0] - 0.2,
            4.0 * parameters[0] - 0.5,
        ]


def make_result(parameters: List[Float64]) raises -> LeastSquaresResult:
    return LeastSquaresResult(
        parameters,
        cost=0.125,
        optimality=0.25,
        iterations=0,
        residual_evaluations=1,
        jacobian_evaluations=0,
        termination=TerminationReason.GRADIENT_TOLERANCE,
    )


def assert_relative_close(
    actual: Float64, expected: Float64, tolerance: Float64
) raises:
    assert_true(abs(actual / expected - 1.0) <= tolerance)


def diagonal_covariance(size: Int) -> List[Float64]:
    var covariance = List[Float64](length=size * size, fill=0.0)
    for index in range(size):
        covariance[index * size + index] = 1.0
    return covariance^


def test_statistics_match_scipy_curve_fit_fixture() raises:
    var problem = LeastSquaresProblem(ScipyDecayModel(), [1.0, 1.0])
    var result = least_squares(problem)

    assert_true(result.converged())
    assert_true(abs(result.parameters[0] - 2.487360807946705) <= 1.0e-6)
    assert_true(abs(result.parameters[1] - 0.6994084195065889) <= 1.0e-6)
    var solve_calls = problem.model.calls
    var statistics = fit_statistics(problem, result)

    assert_equal(problem.model.calls, solve_calls + 3)
    assert_equal(statistics.degrees_of_freedom, 23)
    assert_relative_close(statistics.reduced_chi_squared, 0.0017814213127852873, 1.0e-6)
    var reference_covariance = [
        0.0007961582227048817,
        0.00022689406410722306,
        0.00022689406410722306,
        0.00014900099085114673,
    ]
    for row in range(2):
        for column in range(2):
            assert_relative_close(
                statistics.covariance(row, column),
                reference_covariance[row * 2 + column],
                1.0e-2,
            )
    assert_relative_close(statistics.standard_error(0), 0.02821627584754731, 5e-3)
    assert_relative_close(statistics.standard_error(1), 0.012206596202510621, 5e-3)
    assert_true(
        statistics.correlation(0, 1)
        == statistics.covariance(0, 1)
        / (statistics.standard_error(0) * statistics.standard_error(1))
    )
    assert_true(abs(statistics.correlation(0, 0) - 1.0) <= 1.0e-12)


def test_absolute_sigma_skips_reduced_chi_squared_rescaling() raises:
    var weights = List[Float64](length=25, fill=0.0)
    for index in range(25):
        weights[index] = 1.0 / (0.03 + 0.002 * Float64(index))
    var problem = LeastSquaresProblem(ScipyDecayModel(), [1.0, 1.0], weights=weights)
    var result = least_squares(problem)
    var relative = fit_statistics(problem, result)
    var absolute = fit_statistics(problem, result, absolute_sigma=True)

    assert_equal(absolute.degrees_of_freedom, relative.degrees_of_freedom)
    assert_true(absolute.reduced_chi_squared == relative.reduced_chi_squared)
    var scale = sqrt(relative.reduced_chi_squared)
    for index in range(2):
        assert_relative_close(
            absolute.standard_error(index),
            relative.standard_error(index) / scale,
            1.0e-12,
        )


def test_central_scheme_matches_fixture_with_more_residual_calls() raises:
    var forward_problem = LeastSquaresProblem(ScipyDecayModel(), [1.0, 1.0])
    var central_problem = LeastSquaresProblem(
        ScipyDecayModel(),
        [1.0, 1.0],
        options=LeastSquaresOptions(jacobian_scheme=JacobianScheme.CENTRAL),
    )
    var forward = least_squares(forward_problem)
    var central = least_squares(central_problem)

    assert_true(abs(forward.parameters[0] - central.parameters[0]) <= 1.0e-8)
    assert_true(abs(forward.parameters[1] - central.parameters[1]) <= 1.0e-8)
    assert_true(central.residual_evaluations > forward.residual_evaluations)


def test_nonpositive_degrees_of_freedom_teaches_the_remedy() raises:
    var problem = LeastSquaresProblem(EqualDimensionsModel(), [0.0, 0.0])
    var result = make_result([1.0, -1.0])
    with assert_raises(contains="degrees of freedom"):
        _ = fit_statistics(problem, result)


def test_rank_deficient_jacobian_has_a_specific_error() raises:
    var problem = LeastSquaresProblem(RankDeficientModel(), [0.5, 0.5])
    var result = make_result([0.5, 0.5])
    with assert_raises(contains="rank-deficient"):
        _ = fit_statistics(problem, result)


def test_exact_fit_has_zero_covariance_and_standard_error() raises:
    var problem = LeastSquaresProblem(ExactFitStatisticsModel(), [1.0])
    var result = make_result([2.0])
    var statistics = fit_statistics(problem, result)

    assert_equal(statistics.degrees_of_freedom, 3)
    assert_true(statistics.reduced_chi_squared == 0.0)
    assert_true(statistics.covariance(0, 0) == 0.0)
    assert_true(statistics.standard_error(0) == 0.0)
    with assert_raises(contains="correlation is undefined when standard error is zero"):
        _ = statistics.correlation(0, 0)


def test_correlation_rejects_an_underflowed_standard_error_product() raises:
    var statistics = FitStatistics(
        [0.0, 0.0, 0.0, 0.0],
        [1.0e-300, 1.0e-300],
        degrees_of_freedom=1,
        reduced_chi_squared=0.0,
    )
    with assert_raises(contains="standard-error product underflows"):
        _ = statistics.correlation(0, 1)


def test_statistics_jacobian_accepts_one_ulp_below_an_upper_bound() raises:
    var predecessor = bitcast[DType.float64](bitcast[DType.uint64](1.0) - UInt64(1))
    var problem = LeastSquaresProblem(
        UpperEdgeStatisticsModel(),
        [predecessor],
        bounds=Bounds([0.0], [1.0]),
    )
    var statistics = fit_statistics(problem, make_result([predecessor]))

    assert_equal(statistics.degrees_of_freedom, 3)
    assert_true(statistics.standard_error(0) > 0.0)


def test_statistics_reject_invalid_storage() raises:
    with assert_raises(contains="fit statistics require at least one parameter; got 0"):
        _ = FitStatistics(
            List[Float64](),
            List[Float64](),
            degrees_of_freedom=1,
            reduced_chi_squared=1.0,
        )
    with assert_raises(contains="covariance has 3 entries"):
        _ = FitStatistics(
            [1.0, 0.0, 1.0],
            [1.0, 1.0],
            degrees_of_freedom=1,
            reduced_chi_squared=1.0,
        )
    with assert_raises(contains="covariance entry 0 must be finite"):
        _ = FitStatistics(
            [inf[DType.float64]()],
            [1.0],
            degrees_of_freedom=1,
            reduced_chi_squared=1.0,
        )
    with assert_raises(contains="diagonal entry 0 must be non-negative"):
        _ = FitStatistics(
            [-1.0],
            [1.0],
            degrees_of_freedom=1,
            reduced_chi_squared=1.0,
        )
    with assert_raises(contains="standard error 0 must be finite and non-negative"):
        _ = FitStatistics(
            [1.0],
            [-1.0],
            degrees_of_freedom=1,
            reduced_chi_squared=1.0,
        )
    with assert_raises(contains="degrees of freedom must be at least 1"):
        _ = FitStatistics(
            [1.0],
            [1.0],
            degrees_of_freedom=0,
            reduced_chi_squared=1.0,
        )
    with assert_raises(contains="reduced chi-squared must be finite"):
        _ = FitStatistics(
            [1.0],
            [1.0],
            degrees_of_freedom=1,
            reduced_chi_squared=-1.0,
        )


def test_writable_values_have_exact_stable_blocks() raises:
    assert_equal(String(TerminationReason.GRADIENT_TOLERANCE), "gradient tolerance")
    assert_equal(String(TerminationReason.STEP_TOLERANCE), "step tolerance")
    assert_equal(String(TerminationReason.COST_TOLERANCE), "cost tolerance")
    assert_equal(String(TerminationReason.MAX_ITERATIONS), "max iterations")
    assert_equal(String(TerminationReason.MAX_EVALUATIONS), "max evaluations")
    assert_equal(String(TerminationReason.NUMERICAL_FAILURE), "numerical failure")

    var result = LeastSquaresResult(
        [1.5, -0.5],
        cost=0.125,
        optimality=0.25,
        iterations=2,
        residual_evaluations=5,
        jacobian_evaluations=2,
        termination=TerminationReason.GRADIENT_TOLERANCE,
    )
    var expected_result = String(
        "termination           gradient tolerance\n",
        "converged             yes\n",
        "cost                  0.125\n",
        "optimality            0.25\n",
        "iterations            2\n",
        "residual evaluations  5\n",
        "jacobian evaluations  2\n",
        "parameters[0]         1.5\n",
        "parameters[1]         -0.5\n",
    )
    assert_equal(String(result), expected_result)

    var statistics = FitStatistics(
        [4.0, 1.0, 1.0, 9.0],
        [2.0, 3.0],
        degrees_of_freedom=3,
        reduced_chi_squared=0.125,
    )
    var expected_statistics = String(
        "degrees of freedom    3\n",
        "reduced chi-squared   0.125\n",
        "standard errors[0]    2.0\n",
        "standard errors[1]    3.0\n",
    )
    assert_equal(String(statistics), expected_statistics)

    var report_result = make_result([2.4873608, 0.6994084])
    var report_statistics = FitStatistics(
        [1.0, 0.0, 0.0, 1.0],
        [0.0282163, 0.0122066],
        degrees_of_freedom=23,
        reduced_chi_squared=0.125,
    )
    var report = FitReport(report_result, report_statistics)
    report_result.parameters[0] = 0.0
    report_statistics.degrees_of_freedom = 1
    var expected_report = String(
        "termination           gradient tolerance\n",
        "converged             yes\n",
        "cost                  0.125\n",
        "optimality            0.25\n",
        "iterations            0\n",
        "residual evaluations  1\n",
        "jacobian evaluations  0\n",
        "parameters[0]         2.487 +/- 0.028\n",
        "parameters[1]         0.699 +/- 0.012\n",
        "degrees of freedom    23\n",
        "reduced chi-squared   0.125\n",
    )
    assert_equal(String(report), expected_report)


def test_report_rounding_examples_and_fallback_are_exact() raises:
    var result = make_result([2.4873608, 0.6994084, 1234.567, 5.0])
    var statistics = FitStatistics(
        diagonal_covariance(4),
        [0.0282163, 0.0122066, 42.0, 1.0e300],
        degrees_of_freedom=7,
        reduced_chi_squared=0.5,
    )
    var report = FitReport(result, statistics)
    var expected = String(
        "termination           gradient tolerance\n",
        "converged             yes\n",
        "cost                  0.125\n",
        "optimality            0.25\n",
        "iterations            0\n",
        "residual evaluations  1\n",
        "jacobian evaluations  0\n",
        "parameters[0]         2.487 +/- 0.028\n",
        "parameters[1]         0.699 +/- 0.012\n",
        "parameters[2]         1235 +/- 42\n",
        "parameters[3]         ",
        5.0,
        " +/- ",
        1.0e300,
        "\n",
        "degrees of freedom    7\n",
        "reduced chi-squared   0.5\n",
    )
    assert_equal(String(report), expected)


def test_report_uses_names_and_marks_active_bounds_without_uncertainty() raises:
    var result = LeastSquaresResult(
        [2.4873608, 1.0],
        cost=0.125,
        optimality=0.25,
        iterations=0,
        residual_evaluations=1,
        jacobian_evaluations=0,
        termination=TerminationReason.GRADIENT_TOLERANCE,
        active_bounds=Optional[List[Int]]([0, 1]),
    )
    var statistics = FitStatistics(
        [1.0, 0.0, 0.0, 1.0],
        [0.0282163, 0.5],
        degrees_of_freedom=3,
        reduced_chi_squared=0.125,
    )
    var report = FitReport(
        result,
        statistics,
        parameter_names=["a_parameter_name_longer_than_22_bytes", "rate"],
    )
    var expected = String(
        "termination           gradient tolerance\n",
        "converged             yes\n",
        "cost                  0.125\n",
        "optimality            0.25\n",
        "iterations            0\n",
        "residual evaluations  1\n",
        "jacobian evaluations  0\n",
        "a_parameter_name_longer_than_22_bytes  2.487 +/- 0.028\n",
        "rate                  1.0 (at upper bound)\n",
        "active bounds[1]      upper\n",
        "degrees of freedom    3\n",
        "reduced chi-squared   0.125\n",
    )
    assert_equal(String(report), expected)


def test_result_and_statistics_parameter_counts_must_match() raises:
    var result = make_result([1.0])
    var statistics = FitStatistics(
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0],
        degrees_of_freedom=1,
        reduced_chi_squared=1.0,
    )
    with assert_raises(contains="result has 1 parameters but statistics has 2"):
        _ = FitReport(result, statistics)

    var problem = LeastSquaresProblem(EqualDimensionsModel(), [0.0, 0.0])
    with assert_raises(contains="result has 1 parameters but problem has 2"):
        _ = fit_statistics(problem, result)


def test_statistics_equality_compares_every_field() raises:
    var first = FitStatistics(
        [1.0], [1.0], degrees_of_freedom=2, reduced_chi_squared=0.5
    )
    var same = FitStatistics(
        [1.0], [1.0], degrees_of_freedom=2, reduced_chi_squared=0.5
    )
    var different = FitStatistics(
        [2.0], [1.0], degrees_of_freedom=2, reduced_chi_squared=0.5
    )
    assert_true(first == same)
    assert_true(first != different)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
