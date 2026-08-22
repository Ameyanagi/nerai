"""Private finite-difference Jacobian construction and call accounting."""

from std.collections import Optional
from std.utils.numerics import isfinite

from .bounds import Bounds
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
    bounds: Optional[Bounds] = None,
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

    var perturbed_parameters = parameters.copy()
    for col in range(parameter_count):
        if model.residual_count() != residual_count:
            raise Error("model residual count changed during Jacobian evaluation")

        var parameter = parameters[col]
        var nominal_step = relative_step * max(1.0, abs(parameter))
        var step = _bounded_forward_step(parameter, nominal_step, bounds, col)
        var perturbed_value = parameter + step
        if (
            not isfinite(step)
            or step == 0.0
            or not isfinite(perturbed_value)
            or perturbed_value == parameter
        ):
            raise Error("finite-difference perturbation is not representable")

        perturbed_parameters[col] = perturbed_value
        evaluations += 1
        var perturbed_residuals = model.residuals(perturbed_parameters)
        _validate_callback_result(model, perturbed_residuals, residual_count)

        for row in range(residual_count):
            result._values[col * residual_count + row] = (
                perturbed_residuals[row] - base_residuals[row]
            ) / step
        perturbed_parameters[col] = parameter
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
    bounds: Optional[Bounds] = None,
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

    var perturbed_parameters = parameters.copy()
    for col in range(parameter_count):
        if model.residual_count() != residual_count:
            raise Error("model residual count changed during Jacobian evaluation")

        var parameter = parameters[col]
        var nominal_step = relative_step * max(1.0, abs(parameter))
        var symmetric = _step_is_inside(
            parameter, nominal_step, bounds, col
        ) and _step_is_inside(parameter, -nominal_step, bounds, col)
        var step = _bounded_central_step(parameter, nominal_step, bounds, col)
        var plus_value = parameter + step
        var minus_value = parameter - step
        if symmetric and (
            not isfinite(step)
            or step <= 0.0
            or not isfinite(plus_value)
            or not isfinite(minus_value)
            or plus_value == parameter
            or minus_value == parameter
        ):
            raise Error("finite-difference perturbation is not representable")

        if symmetric:
            perturbed_parameters[col] = plus_value
            evaluations += 1
            var plus_residuals = model.residuals(perturbed_parameters)
            _validate_callback_result(model, plus_residuals, residual_count)

            perturbed_parameters[col] = minus_value
            evaluations += 1
            var minus_residuals = model.residuals(perturbed_parameters)
            _validate_callback_result(model, minus_residuals, residual_count)

            for row in range(residual_count):
                result._values[col * residual_count + row] = (
                    plus_residuals[row] - minus_residuals[row]
                ) / (2.0 * step)
        else:
            # Preserve central differences' two-callback budget with a
            # second-order one-sided stencil when a bound blocks one side.
            var second_value = parameter + 2.0 * step
            if not _step_is_inside(parameter, 2.0 * step, bounds, col):
                raise Error(
                    "bounded finite-difference perturbation is not representable"
                )
            perturbed_parameters[col] = plus_value
            evaluations += 1
            var first_residuals = model.residuals(perturbed_parameters)
            _validate_callback_result(model, first_residuals, residual_count)

            perturbed_parameters[col] = second_value
            evaluations += 1
            var second_residuals = model.residuals(perturbed_parameters)
            _validate_callback_result(model, second_residuals, residual_count)
            for row in range(residual_count):
                result._values[col * residual_count + row] = (
                    -3.0 * base_residuals[row]
                    + 4.0 * first_residuals[row]
                    - second_residuals[row]
                ) / (2.0 * step)
        perturbed_parameters[col] = parameter
    return result^


def _forward_difference_jacobian_without_base[
    M: ResidualModel
](
    mut model: M,
    parameters: List[Float64],
    mut evaluations: Int,
    *,
    relative_step: Float64 = _DEFAULT_RELATIVE_STEP,
    bounds: Optional[Bounds] = None,
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
        bounds=bounds,
    )


def _bounded_forward_step(
    parameter: Float64,
    nominal_step: Float64,
    bounds: Optional[Bounds],
    index: Int,
) raises -> Float64:
    """Prefer a forward step, then use an inward backward step near a bound."""
    if not bounds:
        return nominal_step
    var configured = bounds.value().copy()
    var forward = nominal_step
    if isfinite(configured.upper(index)):
        forward = min(forward, 0.5 * (configured.upper(index) - parameter))
    if _step_is_inside(parameter, forward, bounds, index):
        return forward

    var backward = nominal_step
    if isfinite(configured.lower(index)):
        backward = min(backward, 0.5 * (parameter - configured.lower(index)))
    if _step_is_inside(parameter, -backward, bounds, index):
        return -backward
    raise Error("bounded finite-difference perturbation is not representable")


def _bounded_central_step(
    parameter: Float64,
    nominal_step: Float64,
    bounds: Optional[Bounds],
    index: Int,
) raises -> Float64:
    """Return a symmetric step or an inward step for a one-sided stencil."""
    if not bounds:
        return nominal_step
    if _step_is_inside(parameter, nominal_step, bounds, index) and (
        _step_is_inside(parameter, -nominal_step, bounds, index)
    ):
        return nominal_step

    if _step_is_inside(parameter, 2.0 * nominal_step, bounds, index):
        return nominal_step
    if _step_is_inside(parameter, -2.0 * nominal_step, bounds, index):
        return -nominal_step

    var configured = bounds.value().copy()
    var forward = nominal_step
    if isfinite(configured.upper(index)):
        forward = min(forward, 0.25 * (configured.upper(index) - parameter))
    var backward = nominal_step
    if isfinite(configured.lower(index)):
        backward = min(backward, 0.25 * (parameter - configured.lower(index)))
    if backward >= forward and _step_is_inside(
        parameter, -2.0 * backward, bounds, index
    ):
        return -backward
    if _step_is_inside(parameter, 2.0 * forward, bounds, index):
        return forward
    if _step_is_inside(parameter, -2.0 * backward, bounds, index):
        return -backward
    raise Error("bounded finite-difference perturbation is not representable")


def _step_is_inside(
    parameter: Float64,
    step: Float64,
    bounds: Optional[Bounds],
    index: Int,
) -> Bool:
    var perturbed = parameter + step
    if (
        not isfinite(step)
        or step == 0.0
        or not isfinite(perturbed)
        or perturbed == parameter
    ):
        return False
    if not bounds:
        return True
    var configured = bounds.value().copy()
    if isfinite(configured.lower(index)) and perturbed <= configured.lower(index):
        return False
    if isfinite(configured.upper(index)) and perturbed >= configured.upper(index):
        return False
    return True


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
