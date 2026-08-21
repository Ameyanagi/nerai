"""Private finite-difference Jacobian construction and call accounting."""

from std.utils.numerics import isfinite

from ._kernel import _DenseMatrix
from .problem import ResidualModel


# sqrt(2^-52), the square root of binary64 machine epsilon.
comptime _DEFAULT_RELATIVE_STEP = 1.4901161193847656e-8


def _forward_difference_jacobian[
    M: ResidualModel
](
    mut model: M,
    parameters: List[Float64],
    base_residuals: List[Float64],
    mut evaluations: Int,
    *,
    relative_step: Float64 = _DEFAULT_RELATIVE_STEP,
) raises -> _DenseMatrix:
    """Differentiate raw residuals using a supplied base evaluation.

    ``evaluations`` is incremented immediately before each callback invocation.
    Thus a successful Jacobian adds exactly one call per parameter, and a call
    that raises is still accounted for. Callback errors propagate directly
    without translation.
    """
    _validate_inputs(model, parameters, base_residuals, relative_step)
    var residual_count = len(base_residuals)
    var parameter_count = len(parameters)
    var result = _DenseMatrix(residual_count, parameter_count)

    for col in range(parameter_count):
        if model.residual_count() != residual_count:
            raise Error("model residual count changed during Jacobian evaluation")

        var parameter = parameters[col]
        var step = relative_step * max(1.0, abs(parameter))
        var perturbed_value = parameter + step
        if (
            not isfinite(step)
            or step <= 0.0
            or not isfinite(perturbed_value)
            or perturbed_value == parameter
        ):
            raise Error("finite-difference perturbation is not representable")

        var perturbed_parameters = parameters.copy()
        perturbed_parameters[col] = perturbed_value
        evaluations += 1
        var perturbed_residuals = model.residuals(perturbed_parameters)
        _validate_callback_result(model, perturbed_residuals, residual_count)

        for row in range(residual_count):
            result._values[row * parameter_count + col] = (
                perturbed_residuals[row] - base_residuals[row]
            ) / step
    return result^


def _central_difference_jacobian[
    M: ResidualModel
](
    mut model: M,
    parameters: List[Float64],
    base_residuals: List[Float64],
    mut evaluations: Int,
    *,
    relative_step: Float64 = _DEFAULT_RELATIVE_STEP,
) raises -> _DenseMatrix:
    """Differentiate raw residuals symmetrically around a supplied base.

    The base residual vector is validated for symmetry with forward
    differences but is not used in the derivative. ``evaluations`` increments
    immediately before each callback, so a successful Jacobian adds exactly
    two calls per parameter and a raising call remains accounted for.
    """
    _validate_inputs(model, parameters, base_residuals, relative_step)
    var residual_count = len(base_residuals)
    var parameter_count = len(parameters)
    var result = _DenseMatrix(residual_count, parameter_count)

    for col in range(parameter_count):
        if model.residual_count() != residual_count:
            raise Error("model residual count changed during Jacobian evaluation")

        var parameter = parameters[col]
        var step = relative_step * max(1.0, abs(parameter))
        var plus_value = parameter + step
        var minus_value = parameter - step
        if (
            not isfinite(step)
            or step <= 0.0
            or not isfinite(plus_value)
            or not isfinite(minus_value)
            or plus_value == parameter
            or minus_value == parameter
        ):
            raise Error("finite-difference perturbation is not representable")

        var perturbed_parameters = parameters.copy()
        perturbed_parameters[col] = plus_value
        evaluations += 1
        var plus_residuals = model.residuals(perturbed_parameters)
        _validate_callback_result(model, plus_residuals, residual_count)

        perturbed_parameters[col] = minus_value
        evaluations += 1
        var minus_residuals = model.residuals(perturbed_parameters)
        _validate_callback_result(model, minus_residuals, residual_count)

        for row in range(residual_count):
            result._values[row * parameter_count + col] = (
                plus_residuals[row] - minus_residuals[row]
            ) / (2.0 * step)
    return result^


def _forward_difference_jacobian_without_base[
    M: ResidualModel
](
    mut model: M,
    parameters: List[Float64],
    mut evaluations: Int,
    *,
    relative_step: Float64 = _DEFAULT_RELATIVE_STEP,
) raises -> _DenseMatrix:
    """Evaluate a base residual vector, then form a forward Jacobian.

    A successful call adds ``n + 1`` callback invocations to ``evaluations``
    for ``n`` parameters.
    """
    if len(parameters) == 0:
        raise Error("finite-difference Jacobian requires at least one parameter")
    for index in range(len(parameters)):
        if not isfinite(parameters[index]):
            raise Error("Jacobian parameters must be finite")
    _validate_relative_step(relative_step)

    var residual_count = model.residual_count()
    if residual_count <= 0:
        raise Error("finite-difference Jacobian requires at least one residual")
    evaluations += 1
    var base_residuals = model.residuals(parameters)
    _validate_callback_result(model, base_residuals, residual_count)
    return _forward_difference_jacobian(
        model,
        parameters,
        base_residuals,
        evaluations,
        relative_step=relative_step,
    )


def _validate_inputs[
    M: ResidualModel
](
    model: M,
    parameters: List[Float64],
    base_residuals: List[Float64],
    relative_step: Float64,
) raises:
    if len(parameters) == 0:
        raise Error("finite-difference Jacobian requires at least one parameter")
    for index in range(len(parameters)):
        if not isfinite(parameters[index]):
            raise Error("Jacobian parameters must be finite")
    _validate_relative_step(relative_step)

    var residual_count = len(base_residuals)
    if residual_count == 0:
        raise Error("finite-difference Jacobian requires at least one residual")
    if model.residual_count() != residual_count:
        raise Error("base residual count must match the model declaration")
    for index in range(residual_count):
        if not isfinite(base_residuals[index]):
            raise Error("model residuals must be finite")


def _validate_relative_step(relative_step: Float64) raises:
    if not isfinite(relative_step) or relative_step <= 0.0:
        raise Error("finite-difference step must be finite and positive")


def _validate_callback_result[
    M: ResidualModel
](model: M, residuals: List[Float64], expected_count: Int) raises:
    if model.residual_count() != expected_count:
        raise Error("model residual count changed during Jacobian evaluation")
    if len(residuals) != expected_count:
        raise Error("model returned an unexpected residual count")
    for index in range(expected_count):
        if not isfinite(residuals[index]):
            raise Error("model residuals must be finite")
