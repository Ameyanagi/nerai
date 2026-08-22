"""Private weighted and robust least-squares objective construction."""

from std.math import sqrt
from std.utils.numerics import isfinite

from ._kernel import _DenseMatrix, _jt_residual, _norm_inf
from .loss import LossKind, evaluate_loss, robust_cost


struct _ObjectiveModel(Copyable):
    """One scalar objective value and its local Gauss-Newton model."""

    var cost: Float64
    var residuals: List[Float64]
    var jacobian: _DenseMatrix
    var gradient: List[Float64]
    var optimality: Float64

    def __init__(
        out self,
        cost: Float64,
        var residuals: List[Float64],
        var jacobian: _DenseMatrix,
        var gradient: List[Float64],
    ):
        self.cost = cost
        self.residuals = residuals^
        self.jacobian = jacobian^
        self.optimality = _norm_inf(gradient)
        self.gradient = gradient^


def _objective_cost(
    raw_residuals: List[Float64],
    weights: List[Float64],
    loss: LossKind,
    scale: Float64,
) raises -> Float64:
    """Return the robust objective after applying each weight exactly once."""
    if len(raw_residuals) == 0:
        raise Error("objective requires at least one residual")
    if len(weights) != len(raw_residuals):
        raise Error("objective weights must match the residual count")
    if not isfinite(scale) or scale <= 0.0:
        raise Error("objective loss scale must be finite and positive")

    var cost = 0.0
    for row in range(len(raw_residuals)):
        var raw_residual = raw_residuals[row]
        var weight = weights[row]
        if not isfinite(raw_residual):
            raise Error("objective residuals must be finite")
        if not isfinite(weight) or weight < 0.0:
            raise Error("objective weights must be finite and non-negative")

        var weighted_residual = weight * raw_residual
        if not isfinite(weighted_residual):
            raise Error("weighted residual is not finite")
        cost += robust_cost(loss, weighted_residual, scale=scale)
        if not isfinite(cost):
            raise Error("objective cost overflowed")
    return cost


def _build_objective_model(
    raw_residuals: List[Float64],
    raw_jacobian: _DenseMatrix,
    weights: List[Float64],
    loss: LossKind,
    scale: Float64,
) raises -> _ObjectiveModel:
    """Build an objective while preserving a caller-owned raw Jacobian."""
    return _build_objective_model_owned(
        raw_residuals,
        raw_jacobian.copy(),
        weights,
        loss,
        scale,
    )


def _build_objective_model_owned(
    raw_residuals: List[Float64],
    var raw_jacobian: _DenseMatrix,
    weights: List[Float64],
    loss: LossKind,
    scale: Float64,
) raises -> _ObjectiveModel:
    """Build ``F``, robustified ``f`` and ``J``, and ``J^T f``.

    A row first becomes ``u_i = w_i r_i``. The model then uses
    ``sqrt(rho'_i) * u_i`` and ``sqrt(rho'_i) * w_i * J_raw[i, :]``.
    The stable derivative formulas below are algebraically identical to
    ``evaluate_loss(loss, (u_i / scale)^2).first_derivative`` without
    requiring the normalized square to be representable.
    """
    if raw_jacobian.rows != len(raw_residuals):
        raise Error("raw Jacobian rows must match the residual count")
    if raw_jacobian.cols == 0:
        raise Error("objective Jacobian requires at least one column")

    var cost = _objective_cost(raw_residuals, weights, loss, scale)
    var model_residuals = List[Float64](length=len(raw_residuals), fill=0.0)
    var model_jacobian = raw_jacobian^

    for row in range(model_jacobian.rows):
        var weight = weights[row]
        var weighted_residual = weight * raw_residuals[row]
        var first_derivative = _loss_first_derivative(loss, weighted_residual, scale)
        if not isfinite(first_derivative) or first_derivative < 0.0:
            raise Error("robust loss derivative is invalid")
        if first_derivative == 0.0:
            # Built-in derivatives are mathematically positive. Preserve a
            # positive scale if an extreme ratio underflows in Float64.
            first_derivative = Float64("5e-324")

        var robust_scale = sqrt(first_derivative)
        var model_residual = robust_scale * weighted_residual
        if not isfinite(model_residual):
            raise Error("robust model residual is not finite")
        model_residuals[row] = model_residual

        var jacobian_scale = robust_scale * weight
        if not isfinite(jacobian_scale):
            raise Error("robust Jacobian scale is not finite")
        for col in range(model_jacobian.cols):
            var offset = model_jacobian._offset(row, col)
            var raw_value = model_jacobian._values[offset]
            if not isfinite(raw_value):
                raise Error("raw Jacobian values must be finite")
            var model_value = jacobian_scale * raw_value
            if not isfinite(model_value):
                raise Error("robust model Jacobian is not finite")
            model_jacobian._values[offset] = model_value

    var gradient = _jt_residual(model_jacobian, model_residuals)
    for col in range(len(gradient)):
        if not isfinite(gradient[col]):
            raise Error("objective gradient is not finite")
    return _ObjectiveModel(cost, model_residuals^, model_jacobian^, gradient^)


def _loss_first_derivative(
    loss: LossKind, weighted_residual: Float64, scale: Float64
) raises -> Float64:
    var normalized_ratio = weighted_residual / scale
    var squared_residual = normalized_ratio * normalized_ratio
    if isfinite(squared_residual):
        return evaluate_loss(loss, squared_residual).first_derivative

    # Continue with loss-specific forms only when the normalized square
    # overflows even though the weighted residual and scale are finite.
    if loss == LossKind.LINEAR:
        return 1.0

    var magnitude = abs(weighted_residual)
    if loss == LossKind.HUBER:
        if magnitude <= scale:
            return 1.0
        return scale / magnitude

    if magnitude <= scale:
        var bounded_ratio = magnitude / scale
        return 1.0 / sqrt(1.0 + bounded_ratio * bounded_ratio)

    var inverse_ratio = scale / magnitude
    return inverse_ratio / sqrt(1.0 + inverse_ratio * inverse_ratio)
