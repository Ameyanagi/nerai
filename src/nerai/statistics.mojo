"""Post-fit uncertainty statistics and stable readable reports."""

from std.collections import List, Optional
from std.io import Writable, Writer
from std.math import floor, log10, sqrt
from std.utils.numerics import isfinite

from ._jacobian import (
    _DEFAULT_RELATIVE_STEP,
    _central_difference_jacobian,
    _forward_difference_jacobian,
)
from ._kernel import _DenseMatrix, _jt_j, _solve_spd
from ._objective import _build_objective_model
from .options import JacobianScheme
from .problem import LeastSquaresProblem, ResidualModel
from .result import LeastSquaresResult


struct FitStatistics(Copyable, Equatable, Writable):
    """Validated covariance and uncertainty statistics for one fitted result.

    Construction establishes every stored invariant. Trusted accessors do not
    revalidate; direct mutation of underscore-prefixed storage is out of
    contract, and ``validate()`` provides an explicit checkpoint afterward.

    Convention: Statistics use the loss-scaled weighted residuals and Jacobian
    produced by the solver's objective model. Reduced chi-squared is
    ``sum(model_residual_i^2) / (m_effective - n)``, where ``m_effective``
    counts positive-weight residuals. By default, covariance is
    ``(J_model^T J_model)^-1 * reduced_chi_squared``, equivalently
    ``(J^T W J)^-1 * reduced_chi_squared`` for the effective row weighting.
    This matches SciPy ``curve_fit(..., absolute_sigma=False)``. Passing
    ``absolute_sigma=True`` to ``fit_statistics`` omits the reduced-chi-squared
    scale and returns ``(J^T W J)^-1`` for known measurement uncertainties.
    Computing statistics re-evaluates the model ``n + 1`` times with forward
    differences or ``2n + 1`` times with central differences at the result
    parameters; those callbacks are outside every solver evaluation budget.
    """

    var _covariance: List[Float64]
    var _standard_errors: List[Float64]
    var degrees_of_freedom: Int
    var reduced_chi_squared: Float64

    def __init__(
        out self,
        covariance: List[Float64],
        standard_errors: List[Float64],
        *,
        degrees_of_freedom: Int,
        reduced_chi_squared: Float64,
    ) raises:
        """Copy and validate a row-major covariance and standard errors."""
        self._covariance = covariance.copy()
        self._standard_errors = standard_errors.copy()
        self.degrees_of_freedom = degrees_of_freedom
        self.reduced_chi_squared = reduced_chi_squared
        self.validate()

    def validate(self) raises:
        """Revalidate every stored statistic after unusual direct mutation."""
        var parameter_count = len(self._standard_errors)
        if parameter_count < 1:
            raise Error(
                String(
                    "fit statistics require at least one parameter; got ",
                    parameter_count,
                )
            )
        if len(self._covariance) != parameter_count * parameter_count:
            raise Error(
                String(
                    "covariance has ",
                    len(self._covariance),
                    " entries but ",
                    parameter_count,
                    " parameters require ",
                    parameter_count * parameter_count,
                )
            )
        for index in range(len(self._covariance)):
            if not isfinite(self._covariance[index]):
                raise Error(
                    String(
                        "covariance entry ",
                        index,
                        " must be finite; got ",
                        self._covariance[index],
                    )
                )
        for index in range(parameter_count):
            if self._covariance[index * parameter_count + index] <= 0.0:
                raise Error(
                    String(
                        "covariance diagonal entry ",
                        index,
                        " must be positive; got ",
                        self._covariance[index * parameter_count + index],
                    )
                )
            if (
                not isfinite(self._standard_errors[index])
                or self._standard_errors[index] <= 0.0
            ):
                raise Error(
                    String(
                        "standard error ",
                        index,
                        " must be finite and positive; got ",
                        self._standard_errors[index],
                    )
                )
        if self.degrees_of_freedom < 1:
            raise Error(
                String(
                    "degrees of freedom must be at least 1; got ",
                    self.degrees_of_freedom,
                )
            )
        if not isfinite(self.reduced_chi_squared) or self.reduced_chi_squared < 0.0:
            raise Error(
                String(
                    "reduced chi-squared must be finite and non-negative; got ",
                    self.reduced_chi_squared,
                )
            )

    def covariance(self, row: Int, column: Int) -> Float64:
        """Return one trusted row-major covariance entry."""
        return self._covariance[row * self.parameter_count() + column]

    def standard_error(self, index: Int) -> Float64:
        """Return one trusted parameter standard error."""
        return self._standard_errors[index]

    def correlation(self, row: Int, column: Int) -> Float64:
        """Return covariance normalized by the two standard errors."""
        return self.covariance(row, column) / (
            self.standard_error(row) * self.standard_error(column)
        )

    def parameter_count(self) -> Int:
        """Return the number of parameters described by these statistics."""
        return len(self._standard_errors)

    def __eq__(self, other: Self) -> Bool:
        """Return whether every stored statistic is exactly equal."""
        if len(self._covariance) != len(other._covariance) or len(
            self._standard_errors
        ) != len(other._standard_errors):
            return False
        for index in range(len(self._covariance)):
            if self._covariance[index] != other._covariance[index]:
                return False
        for index in range(len(self._standard_errors)):
            if self._standard_errors[index] != other._standard_errors[index]:
                return False
        return (
            self.degrees_of_freedom == other.degrees_of_freedom
            and self.reduced_chi_squared == other.reduced_chi_squared
        )

    def __str__(self) -> String:
        """Return the stable multiline statistics block."""
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        """Write the stable statistics block with one trailing newline."""
        writer.write("degrees of freedom    ", self.degrees_of_freedom, "\n")
        writer.write("reduced chi-squared   ", self.reduced_chi_squared, "\n")
        for index in range(self.parameter_count()):
            var label = String("standard errors[", index, "]")
            _write_padded_label(writer, label)
            writer.write(self._standard_errors[index], "\n")


def fit_statistics[
    M: ResidualModel
](
    mut problem: LeastSquaresProblem[M],
    result: LeastSquaresResult,
    *,
    absolute_sigma: Bool = False,
) raises -> FitStatistics:
    """Re-evaluate a fitted problem and estimate parameter uncertainty.

    Convention: One residual vector and one configured numerical Jacobian are
    evaluated at ``result.parameters`` using the problem's finite-difference
    step. The resulting loss-scaled weighted objective model defines
    ``reduced_chi_squared = sum(model_residual_i^2) / (m_effective - n)`` and
    ``covariance = (J^T W J)^-1 * reduced_chi_squared`` by default. With
    ``absolute_sigma=True``, covariance is ``(J^T W J)^-1`` without that
    rescaling. Unit weights with linear loss and the default match SciPy
    ``curve_fit(..., absolute_sigma=False)``. These diagnostic callbacks are
    outside every solver evaluation budget.

    Raises:
        Error: If dimensions mismatch, degrees of freedom are not positive, the
            model evaluation fails, or the model Jacobian is rank-deficient.
    """
    problem.validate()
    var parameter_count = len(problem.initial_parameters)
    if len(result.parameters) != parameter_count:
        raise Error(
            String(
                "result has ",
                len(result.parameters),
                " parameters but problem has ",
                parameter_count,
            )
        )

    var raw_residuals = problem.evaluate_residuals(result.parameters)
    var diagnostic_evaluations = 1
    var relative_step = _DEFAULT_RELATIVE_STEP
    if problem.options.finite_difference_step:
        relative_step = problem.options.finite_difference_step.value()
    var raw_jacobian: _DenseMatrix
    if problem.options.jacobian_scheme == JacobianScheme.CENTRAL:
        raw_jacobian = _central_difference_jacobian(
            problem.model,
            result.parameters,
            raw_residuals,
            diagnostic_evaluations,
            relative_step=relative_step,
        )
    else:
        raw_jacobian = _forward_difference_jacobian(
            problem.model,
            result.parameters,
            raw_residuals,
            diagnostic_evaluations,
            relative_step=relative_step,
        )
    var objective = _build_objective_model(
        raw_residuals,
        raw_jacobian,
        problem.weights,
        problem.options.loss,
        problem.options.loss_scale,
    )

    var effective_residual_count = 0
    for row in range(len(problem.weights)):
        if problem.weights[row] > 0.0:
            effective_residual_count += 1
    var degrees_of_freedom = effective_residual_count - parameter_count
    if degrees_of_freedom <= 0:
        if degrees_of_freedom == 0:
            raise Error(
                String(
                    effective_residual_count,
                    " residuals for ",
                    parameter_count,
                    (
                        " parameters leaves zero degrees of freedom; add "
                        "observations or fix parameters"
                    ),
                )
            )
        raise Error(
            String(
                effective_residual_count,
                " residuals for ",
                parameter_count,
                " parameters leaves ",
                degrees_of_freedom,
                " degrees of freedom; add observations or fix parameters",
            )
        )

    var residual_sum_squares = 0.0
    for row in range(len(objective.residuals)):
        residual_sum_squares += objective.residuals[row] * objective.residuals[row]
    var reduced_chi_squared = residual_sum_squares / Float64(degrees_of_freedom)
    if not isfinite(reduced_chi_squared) or reduced_chi_squared < 0.0:
        raise Error(
            String(
                "reduced chi-squared must be finite and non-negative; got ",
                reduced_chi_squared,
            )
        )

    var normal = _jt_j(objective.jacobian)
    var covariance = List[Float64](length=parameter_count * parameter_count, fill=0.0)
    var covariance_scale = 1.0
    if not absolute_sigma:
        covariance_scale = reduced_chi_squared
    for column in range(parameter_count):
        var right_hand_side = List[Float64](length=parameter_count, fill=0.0)
        right_hand_side[column] = 1.0
        var inverse_column: List[Float64]
        try:
            inverse_column = _solve_spd(normal, right_hand_side)
        except:
            raise Error(
                "Jacobian is rank-deficient at the solution: (J^T W J) could "
                "not be inverted, so some parameters are not independently "
                "determined by the data"
            )
        for row in range(parameter_count):
            covariance[row * parameter_count + column] = (
                inverse_column[row] * covariance_scale
            )

    for row in range(parameter_count):
        for column in range(row, parameter_count):
            var forward = covariance[row * parameter_count + column]
            var reverse = covariance[column * parameter_count + row]
            var symmetric = 0.5 * (forward + reverse)
            covariance[row * parameter_count + column] = symmetric
            covariance[column * parameter_count + row] = symmetric

    var standard_errors = List[Float64](length=parameter_count, fill=0.0)
    for index in range(parameter_count):
        standard_errors[index] = sqrt(covariance[index * parameter_count + index])
    return FitStatistics(
        covariance,
        standard_errors,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_squared=reduced_chi_squared,
    )


struct FitReport(Copyable, Writable):
    """A solver result paired with standard errors and fit diagnostics."""

    var _result: LeastSquaresResult
    var _statistics: FitStatistics
    var _parameter_names: List[String]

    def __init__(
        out self,
        result: LeastSquaresResult,
        statistics: FitStatistics,
        *,
        parameter_names: List[String] = List[String](),
    ) raises:
        """Copy a result and matching parameter statistics into one report."""
        if len(result.parameters) != statistics.parameter_count():
            raise Error(
                String(
                    "result has ",
                    len(result.parameters),
                    " parameters but statistics has ",
                    statistics.parameter_count(),
                )
            )
        if len(parameter_names) > 0:
            if len(parameter_names) != len(result.parameters):
                raise Error(
                    String(
                        "parameter_names has ",
                        len(parameter_names),
                        " entries but result parameters has ",
                        len(result.parameters),
                        " entries; provide exactly one name per parameter",
                    )
                )
            for index in range(len(parameter_names)):
                if parameter_names[index].byte_length() == 0:
                    raise Error(
                        String(
                            "parameter_names[",
                            index,
                            "] must not be empty; got an empty string — provide ",
                            "a non-empty parameter label",
                        )
                    )
        self._result = result.copy()
        self._statistics = statistics.copy()
        self._parameter_names = parameter_names.copy()

    def __str__(self) -> String:
        """Return the stable stderr-aware multiline fit report."""
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        """Write the stable stderr-aware report with one trailing newline."""
        writer.write("termination           ", self._result.termination, "\n")
        writer.write(
            "converged             ",
            "yes" if self._result.converged() else "no",
            "\n",
        )
        writer.write("cost                  ", self._result.cost, "\n")
        writer.write("optimality            ", self._result.optimality, "\n")
        writer.write("iterations            ", self._result.iterations, "\n")
        writer.write("residual evaluations  ", self._result.residual_evaluations, "\n")
        writer.write("jacobian evaluations  ", self._result.jacobian_evaluations, "\n")
        for index in range(len(self._result.parameters)):
            var label = String("parameters[", index, "]")
            if len(self._parameter_names) > 0:
                label = String(self._parameter_names[index])
            _write_padded_label(writer, label)
            if self._result.active_bounds[index] < 0:
                writer.write(
                    String(self._result.parameters[index]),
                    " (at lower bound)\n",
                )
            elif self._result.active_bounds[index] > 0:
                writer.write(
                    String(self._result.parameters[index]),
                    " (at upper bound)\n",
                )
            else:
                writer.write(
                    _format_estimate(
                        self._result.parameters[index],
                        self._statistics.standard_error(index),
                    ),
                    "\n",
                )
        for index in range(len(self._result.active_bounds)):
            if self._result.active_bounds[index] == 0:
                continue
            var label = String("active bounds[", index, "]")
            _write_padded_label(writer, label)
            if self._result.active_bounds[index] < 0:
                writer.write("lower\n")
            else:
                writer.write("upper\n")
        writer.write(
            "degrees of freedom    ", self._statistics.degrees_of_freedom, "\n"
        )
        writer.write(
            "reduced chi-squared   ",
            self._statistics.reduced_chi_squared,
            "\n",
        )


def _write_padded_label[W: Writer](mut writer: W, label: String):
    writer.write(label)
    if label.byte_length() >= 22:
        writer.write("  ")
        return
    for _ in range(label.byte_length(), 22):
        writer.write(" ")


def _format_estimate(value: Float64, standard_error: Float64) -> String:
    if not isfinite(value) or not isfinite(standard_error) or standard_error <= 0.0:
        return String(value, " +/- ", standard_error)

    var exponent = Int(floor(log10(standard_error)))
    var decimals = max(0, 1 - exponent)
    if decimals > 17:
        return String(value, " +/- ", standard_error)

    var factor = 1
    for _ in range(decimals):
        factor *= 10
    var scaled_value = value * Float64(factor)
    var scaled_error = standard_error * Float64(factor)
    if (
        not isfinite(scaled_value)
        or not isfinite(scaled_error)
        or abs(scaled_value) >= 1.0e15
        or abs(scaled_error) >= 1.0e15
    ):
        return String(value, " +/- ", standard_error)

    var rounded_value = _round_to_int(scaled_value)
    var rounded_error = _round_to_int(scaled_error)
    return String(
        _fixed_point(rounded_value, decimals),
        " +/- ",
        _fixed_point(rounded_error, decimals),
    )


def _round_to_int(value: Float64) -> Int:
    if value < 0.0:
        return -Int(floor(-value + 0.5))
    return Int(floor(value + 0.5))


def _fixed_point(scaled_value: Int, decimals: Int) -> String:
    var factor = 1
    for _ in range(decimals):
        factor *= 10

    var magnitude = scaled_value
    var result = String()
    if magnitude < 0:
        result += "-"
        magnitude = -magnitude
    result += String(magnitude // factor)
    if decimals == 0:
        return result^

    var fractional = String(magnitude % factor)
    while fractional.byte_length() < decimals:
        fractional = String("0") + fractional
    result += "."
    result += fractional
    return result^
