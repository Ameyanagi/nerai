from nerai import LossKind, evaluate_loss, robust_cost
from std.math import abs
from std.testing import TestSuite, assert_raises, assert_true
from std.utils.numerics import nan


def assert_close(actual: Float64, expected: Float64, tolerance: Float64 = 1e-12) raises:
    assert_true(abs(actual - expected) <= tolerance)


def assert_relative_close(
    actual: Float64, expected: Float64, tolerance: Float64 = 2e-15
) raises:
    assert_true(actual == actual and actual - actual == 0.0)
    assert_true(abs(actual / expected - 1.0) <= tolerance)


def test_linear_reference_values() raises:
    var evaluation = evaluate_loss(LossKind.LINEAR, 4.0)
    assert_close(evaluation.value, 4.0)
    assert_close(evaluation.first_derivative, 1.0)
    assert_close(evaluation.second_derivative, 0.0)


def test_huber_reference_values() raises:
    var inside = evaluate_loss(LossKind.HUBER, 0.25)
    assert_close(inside.value, 0.25)
    assert_close(inside.first_derivative, 1.0)
    assert_close(inside.second_derivative, 0.0)

    var outside = evaluate_loss(LossKind.HUBER, 4.0)
    assert_close(outside.value, 3.0)
    assert_close(outside.first_derivative, 0.5)
    assert_close(outside.second_derivative, -0.0625)


def test_soft_l1_reference_values() raises:
    var evaluation = evaluate_loss(LossKind.SOFT_L1, 3.0)
    assert_close(evaluation.value, 2.0)
    assert_close(evaluation.first_derivative, 0.5)
    assert_close(evaluation.second_derivative, -0.0625)


def test_loss_derivatives_remain_representable_at_extreme_z() raises:
    var tiny = evaluate_loss(LossKind.SOFT_L1, 1e-20)
    assert_true(tiny.value == 1e-20)

    var huber = evaluate_loss(LossKind.HUBER, 1e210)
    var soft_l1 = evaluate_loss(LossKind.SOFT_L1, 1e210)
    assert_relative_close(huber.second_derivative, -5e-316, 2e-8)
    assert_relative_close(soft_l1.second_derivative, -5e-316, 2e-8)

    var largest = evaluate_loss(LossKind.SOFT_L1, Float64.MAX_FINITE)
    assert_true(largest.value == largest.value)
    assert_true(largest.value - largest.value == 0.0)
    assert_true(largest.first_derivative > 0.0)


def test_huber_is_continuous_at_threshold() raises:
    var at_threshold = evaluate_loss(LossKind.HUBER, 1.0)
    var below = evaluate_loss(LossKind.HUBER, 1.0 - 1e-9)
    var above = evaluate_loss(LossKind.HUBER, 1.0 + 1e-9)

    assert_close(at_threshold.value, 1.0)
    assert_close(at_threshold.first_derivative, 1.0)
    assert_close(below.value, at_threshold.value, 2e-9)
    assert_close(above.value, at_threshold.value, 2e-9)
    assert_close(below.first_derivative, at_threshold.first_derivative, 2e-9)
    assert_close(above.first_derivative, at_threshold.first_derivative, 2e-9)


def test_cost_is_symmetric_and_scale_aware() raises:
    var positive = robust_cost(LossKind.HUBER, 4.0, scale=2.0)
    var negative = robust_cost(LossKind.HUBER, -4.0, scale=2.0)

    assert_close(positive, 6.0)
    assert_close(negative, positive)
    assert_close(robust_cost(LossKind.LINEAR, 4.0, scale=2.0), 8.0)
    assert_close(robust_cost(LossKind.LINEAR, 2.0, scale=Float64.MAX_FINITE), 2.0)


def test_cost_is_stable_when_intermediate_forms_would_overflow() raises:
    var large_inlier = 1.5e154
    assert_relative_close(
        robust_cost(LossKind.LINEAR, large_inlier),
        1.125e308,
    )
    assert_relative_close(
        robust_cost(LossKind.HUBER, large_inlier, scale=Float64.MAX_FINITE),
        1.125e308,
    )

    assert_relative_close(
        robust_cost(LossKind.HUBER, 1e308, scale=1e-308),
        1.0,
    )
    assert_relative_close(
        robust_cost(LossKind.SOFT_L1, 1e308, scale=1e-308),
        1.0,
    )
    var smallest_scale = Float64("5e-324")
    assert_relative_close(
        robust_cost(LossKind.HUBER, 1e308, scale=smallest_scale),
        4.9406564584124655e-16,
    )
    assert_relative_close(
        robust_cost(LossKind.SOFT_L1, 1e308, scale=smallest_scale),
        4.9406564584124655e-16,
    )
    assert_close(
        robust_cost(LossKind.SOFT_L1, 1.0, scale=Float64.MAX_FINITE),
        0.5,
    )
    assert_relative_close(
        robust_cost(LossKind.SOFT_L1, 1e154, scale=1e154),
        4.1421356237309504e307,
    )


def test_cost_rejects_only_unrepresentable_results() raises:
    with assert_raises(contains="robust cost overflowed"):
        _ = robust_cost(LossKind.LINEAR, Float64.MAX_FINITE)
    with assert_raises(contains="robust cost overflowed"):
        _ = robust_cost(
            LossKind.HUBER,
            Float64.MAX_FINITE,
            scale=Float64.MAX_FINITE,
        )
    with assert_raises(contains="robust cost overflowed"):
        _ = robust_cost(
            LossKind.SOFT_L1,
            Float64.MAX_FINITE,
            scale=Float64.MAX_FINITE,
        )


def test_loss_inputs_are_validated() raises:
    with assert_raises(contains="squared residual must be finite and non-negative"):
        _ = evaluate_loss(LossKind.LINEAR, -1.0)
    with assert_raises(contains="squared residual must be finite and non-negative"):
        _ = evaluate_loss(LossKind.LINEAR, Float64.MAX)
    with assert_raises(contains="squared residual must be finite and non-negative"):
        _ = evaluate_loss(LossKind.LINEAR, nan[DType.float64]())
    with assert_raises(contains="residual must be finite"):
        _ = robust_cost(LossKind.LINEAR, Float64.MAX)
    with assert_raises(contains="loss scale must be finite and positive"):
        _ = robust_cost(LossKind.LINEAR, 1.0, scale=0.0)


def test_loss_kinds_are_distinct_nominal_values() raises:
    assert_true(LossKind.LINEAR != LossKind.HUBER)
    assert_true(LossKind.LINEAR != LossKind.SOFT_L1)
    assert_true(LossKind.HUBER != LossKind.SOFT_L1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
