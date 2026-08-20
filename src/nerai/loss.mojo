"""Built-in robust loss functions for nonlinear least squares."""

from std.math import sqrt
from std.utils.numerics import isfinite


struct LossKind(Copyable, Equatable, ImplicitlyCopyable):
    """Nominal selection of a built-in loss function.

    Every built-in evaluates ``rho(z)`` and its first two derivatives with
    respect to ``z``, where ``z`` is a normalized squared residual.
    """

    comptime LINEAR = LossKind(0)
    comptime HUBER = LossKind(1)
    comptime SOFT_L1 = LossKind(2)

    var _value: Int

    def __init__(out self, value: Int):
        self._value = value

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value


struct LossEvaluation(Copyable, ImplicitlyCopyable):
    """A loss value and its first two derivatives with respect to ``z``."""

    var value: Float64
    var first_derivative: Float64
    var second_derivative: Float64

    def __init__(
        out self,
        value: Float64,
        first_derivative: Float64,
        second_derivative: Float64,
    ):
        self.value = value
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative


def evaluate_loss(loss: LossKind, squared_residual: Float64) raises -> LossEvaluation:
    """Evaluate ``rho(z)``, ``rho'(z)``, and ``rho''(z)``.

    Args:
        loss: Built-in loss function to evaluate.
        squared_residual: Finite, non-negative ``z = (residual / scale)^2``.

    Raises:
        Error: If ``squared_residual`` is negative, NaN, or infinite.
    """
    if not isfinite(squared_residual) or squared_residual < 0.0:
        raise Error("squared residual must be finite and non-negative")

    if loss == LossKind.LINEAR:
        return LossEvaluation(squared_residual, 1.0, 0.0)

    if loss == LossKind.HUBER:
        if squared_residual <= 1.0:
            return LossEvaluation(squared_residual, 1.0, 0.0)
        var root = sqrt(squared_residual)
        return LossEvaluation(
            2.0 * root - 1.0,
            1.0 / root,
            (-0.5 / squared_residual) / root,
        )

    var shifted_root = sqrt(1.0 + squared_residual)
    return LossEvaluation(
        2.0 * (squared_residual / (shifted_root + 1.0)),
        1.0 / shifted_root,
        (-0.5 / (1.0 + squared_residual)) / shifted_root,
    )


def robust_cost(
    loss: LossKind, residual: Float64, *, scale: Float64 = 1.0
) raises -> Float64:
    """Return one residual's contribution to the robust least-squares cost.

    The returned value is ``0.5 * scale^2 * rho((residual / scale)^2)``.
    ``scale`` therefore marks the residual magnitude at which non-linear
    losses begin reducing outlier influence. Linear loss is independent of
    scale and always returns ``0.5 * residual^2``.

    Raises:
        Error: If ``residual`` is not finite, ``scale`` is not finite and
            positive, or the mathematical cost exceeds finite ``Float64``.
    """
    if not isfinite(residual):
        raise Error("residual must be finite")
    if not isfinite(scale) or scale <= 0.0:
        raise Error("loss scale must be finite and positive")

    var magnitude = abs(residual)

    # Multiplying by one half before squaring preserves every representable
    # linear cost, including values whose unscaled square would overflow.
    if loss == LossKind.LINEAR:
        var linear_cost = (0.5 * magnitude) * magnitude
        if not isfinite(linear_cost):
            raise Error("robust cost overflowed")
        return linear_cost

    var cost: Float64
    if loss == LossKind.HUBER:
        if magnitude <= scale:
            cost = (0.5 * magnitude) * magnitude
        else:
            # C * (|r| - C/2) is the Huber outlier branch without either
            # normalizing the residual or squaring the scale.
            cost = scale * (magnitude - 0.5 * scale)
    elif magnitude <= scale:
        # For q = |r| / C <= 1, rationalizing sqrt(1 + q^2) - 1
        # yields r^2 / (sqrt(1 + q^2) + 1). Dividing before the final
        # multiplication avoids an overflowing intermediate square.
        var ratio = magnitude / scale
        var denominator = sqrt(1.0 + ratio * ratio) + 1.0
        cost = (magnitude / denominator) * magnitude
    else:
        # For q = C / |r| < 1, the equivalent form below keeps every
        # intermediate bounded before the final representable product.
        var ratio = scale / magnitude
        var factor = sqrt(1.0 + ratio * ratio) - ratio
        cost = (scale * factor) * magnitude

    if not isfinite(cost):
        raise Error("robust cost overflowed")
    return cost
