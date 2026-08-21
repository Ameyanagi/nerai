from nerai import Bounds, CurveFit, CurveModel, FitReport, LeastSquaresOptions
from std.collections import List
from std.math import abs, exp
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.utils.numerics import inf, nan


struct ExponentialCurve(Copyable, CurveModel):
    def __init__(out self):
        pass

    def values(
        mut self, parameters: List[Float64], t: Span[Float64, ...]
    ) raises -> List[Float64]:
        var values = List[Float64](length=len(t), fill=0.0)
        for index in range(len(t)):
            values[index] = parameters[0] * exp(-parameters[1] * t[index])
        return values^


struct GaussianCurve(Copyable, CurveModel):
    def __init__(out self):
        pass

    def values(
        mut self, parameters: List[Float64], x: Span[Float64, ...]
    ) raises -> List[Float64]:
        var values = List[Float64](length=len(x), fill=0.0)
        for index in range(len(x)):
            var normalized = (x[index] - parameters[1]) / parameters[2]
            values[index] = parameters[0] * exp(-0.5 * normalized * normalized)
        return values^


struct WrongLengthCurve(Copyable, CurveModel):
    def __init__(out self):
        pass

    def values(
        mut self, parameters: List[Float64], t: Span[Float64, ...]
    ) raises -> List[Float64]:
        return [parameters[0]]


def decay_t() -> List[Float64]:
    return [
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


def decay_y() -> List[Float64]:
    return [
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


def decay_sigma() -> List[Float64]:
    return [
        0.03,
        0.03208333333333333,
        0.034166666666666665,
        0.03625,
        0.03833333333333333,
        0.04041666666666666,
        0.042499999999999996,
        0.044583333333333336,
        0.04666666666666666,
        0.04875,
        0.05083333333333333,
        0.05291666666666667,
        0.055,
        0.05708333333333333,
        0.059166666666666666,
        0.06125,
        0.06333333333333332,
        0.06541666666666666,
        0.0675,
        0.06958333333333333,
        0.07166666666666666,
        0.07375,
        0.07583333333333334,
        0.07791666666666666,
        0.08,
    ]


def gaussian_x() -> List[Float64]:
    return [
        -5.0,
        -4.75,
        -4.5,
        -4.25,
        -4.0,
        -3.75,
        -3.5,
        -3.25,
        -3.0,
        -2.75,
        -2.5,
        -2.25,
        -2.0,
        -1.75,
        -1.5,
        -1.25,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
    ]


def gaussian_y() -> List[Float64]:
    return [
        0.0001807352760405818,
        0.02410895173226558,
        -0.021421458399293176,
        -0.07005947188413682,
        -0.03372218389293266,
        -0.07366468868858532,
        0.016409248626213492,
        0.1299442519771425,
        0.003267852136104546,
        0.026980070024550495,
        0.1709981648850381,
        0.24568166840852007,
        0.35091841994889256,
        0.44282742810460207,
        0.7457164805348266,
        1.0914969278709645,
        1.265962921532036,
        1.707209792737026,
        1.967847014389081,
        2.3645696680131962,
        2.6032272641727485,
        2.916789885733709,
        2.898604281484504,
        2.9572983249254197,
        2.763106154126025,
        2.452778211625598,
        1.9186040567075076,
        1.7007236219525124,
        1.3696200096827569,
        1.044937391194566,
        0.6256457650914571,
        0.479044609598544,
        0.2642037537845295,
        0.15242372860166728,
        0.21668269074110857,
        0.012015287983594011,
        0.04004263722456927,
        0.09347822172343373,
        -0.03509027420104574,
        -0.0032691202556011075,
        0.011488610380763598,
    ]


def bounded_t() -> List[Float64]:
    return [
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


def bounded_y() -> List[Float64]:
    return [
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


def assert_relative_close(
    actual: Float64, expected: Float64, tolerance: Float64
) raises:
    assert_true(abs(actual / expected - 1.0) <= tolerance)


def test_exponential_front_door_matches_scipy() raises:
    var t = decay_t()
    var y = decay_y()
    var fit = CurveFit(ExponentialCurve(), t, y, [1.0, 1.0])
    var result = fit.solve()

    assert_true(result.result.converged())
    assert_true(abs(result.result.parameters[0] - 2.487360807946705) <= 1.0e-6)
    assert_true(abs(result.result.parameters[1] - 0.6994084195065889) <= 1.0e-6)
    assert_relative_close(
        result.statistics.standard_error(0), 0.02821627584754731, 5.0e-3
    )
    assert_relative_close(
        result.statistics.standard_error(1), 0.012206596202510621, 5.0e-3
    )
    assert_equal(result.statistics.degrees_of_freedom, 23)


def test_sigma_weighting_matches_absolute_sigma_false_semantics() raises:
    var t = decay_t()
    var y = decay_y()
    var uniform_sigma = List[Float64](length=len(y), fill=2.0)
    var scale_invariant_options = LeastSquaresOptions(
        ftol=None,
        xtol=1.0e-12,
        gtol=None,
    )
    var unweighted_fit = CurveFit(
        ExponentialCurve(),
        t,
        y,
        [1.0, 1.0],
        options=scale_invariant_options.copy(),
    )
    var uniform_fit = CurveFit(
        ExponentialCurve(),
        t,
        y,
        [1.0, 1.0],
        sigma=uniform_sigma,
        options=scale_invariant_options.copy(),
    )
    var unweighted = unweighted_fit.solve()
    var uniform = uniform_fit.solve()

    for index in range(2):
        assert_relative_close(
            uniform.result.parameters[index],
            unweighted.result.parameters[index],
            1.0e-8,
        )
        assert_relative_close(
            uniform.statistics.standard_error(index),
            unweighted.statistics.standard_error(index),
            1.0e-8,
        )

    var sigma = decay_sigma()
    var weighted_fit = CurveFit(ExponentialCurve(), t, y, [1.0, 1.0], sigma=sigma)
    var weighted = weighted_fit.solve()
    var expected_parameters = [2.4956284512914686, 0.7053790560553745]
    var expected_errors = [0.020914312905673485, 0.012407539523481984]
    for index in range(2):
        assert_relative_close(
            weighted.result.parameters[index], expected_parameters[index], 5.0e-3
        )
        assert_relative_close(
            weighted.statistics.standard_error(index),
            expected_errors[index],
            5.0e-3,
        )


def test_gaussian_front_door_matches_scipy_and_writes_uncertainties() raises:
    var x = gaussian_x()
    var y = gaussian_y()
    var fit = CurveFit(GaussianCurve(), x, y, [1.0, 0.0, 2.0])
    var result = fit.solve()
    var expected = [
        2.9357367116390236,
        0.5049038858565743,
        1.1851977478173878,
    ]
    for index in range(3):
        assert_true(abs(result.result.parameters[index] - expected[index]) <= 1.0e-6)
    var report = FitReport(result.result, result.statistics)
    assert_equal(String(result), String(report))
    assert_true(" +/- " in String(result))


def test_construction_rejects_invalid_data_and_sigma() raises:
    var two = [0.0, 1.0]
    var three = [1.0, 2.0, 3.0]
    with assert_raises(
        contains="initial_parameters must contain at least one parameter; got 0"
    ):
        _ = CurveFit(ExponentialCurve(), three, three.copy(), List[Float64]())

    with assert_raises(contains="t has 2 entries but y has 3 entries"):
        _ = CurveFit(ExponentialCurve(), two, three, [1.0])

    var nonfinite_y = [1.0, nan[DType.float64](), 2.0]
    with assert_raises(contains="y[1] must be finite"):
        _ = CurveFit(ExponentialCurve(), three, nonfinite_y, [1.0])

    var good_y = [1.0, 2.0, 3.0]
    var short_sigma = [1.0, 1.0]
    with assert_raises(contains="sigma has 2 entries but y has 3 entries"):
        _ = CurveFit(ExponentialCurve(), three, good_y, [1.0], sigma=short_sigma)

    var zero_sigma = [1.0, 0.0, 1.0]
    with assert_raises(contains="sigma[1]"):
        _ = CurveFit(ExponentialCurve(), three, good_y, [1.0], sigma=zero_sigma)
    var negative_sigma = [1.0, -1.0, 1.0]
    with assert_raises(contains="sigma[1]"):
        _ = CurveFit(ExponentialCurve(), three, good_y, [1.0], sigma=negative_sigma)
    var infinite_sigma = [1.0, inf[DType.float64](), 1.0]
    with assert_raises(contains="sigma[1]"):
        _ = CurveFit(ExponentialCurve(), three, good_y, [1.0], sigma=infinite_sigma)

    with assert_raises(
        contains="2 observations for 2 parameters leaves zero degrees of freedom"
    ):
        _ = CurveFit(ExponentialCurve(), two, two.copy(), [1.0, 1.0])


def test_wrong_model_length_reports_both_counts() raises:
    var t = [0.0, 1.0, 2.0]
    var y = [1.0, 1.0, 1.0]
    var fit = CurveFit(WrongLengthCurve(), t, y, [1.0])
    with assert_raises(contains="model returned 1 values for 3 t points"):
        _ = fit.solve()


def test_bounded_front_door_reports_active_rate() raises:
    var t = bounded_t()
    var y = bounded_y()
    var bounds = Bounds(
        [0.0, 0.0],
        [inf[DType.float64](), inf[DType.float64]()],
    )
    var fit = CurveFit(ExponentialCurve(), t, y, [0.5, 0.1], bounds=bounds.copy())
    var result = fit.solve()

    assert_true(result.result.parameters[1] >= 0.0)
    assert_true(result.result.parameters[1] <= 1.0e-6)
    assert_equal(result.result.active_bounds, [0, -1])


def test_repeated_solves_are_exactly_deterministic() raises:
    var t = decay_t()
    var y = decay_y()
    var fit = CurveFit(ExponentialCurve(), t, y, [1.0, 1.0])
    var first = fit.solve()
    var second = fit.solve()

    assert_equal(first.result.iterations, second.result.iterations)
    assert_true(first.result.parameters[0] == second.result.parameters[0])
    assert_true(first.result.parameters[1] == second.result.parameters[1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
