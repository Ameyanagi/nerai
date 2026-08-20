"""Public Levenberg-Marquardt nonlinear least-squares solver."""

from std.utils.numerics import isfinite

from ._jacobian import _DEFAULT_RELATIVE_STEP, _forward_difference_jacobian
from ._kernel import (
    _DenseMatrix,
    _add_diagonal_damping,
    _dot,
    _jt_j,
    _norm2,
    _solve_spd,
)
from ._objective import _ObjectiveModel, _build_objective_model, _objective_cost
from .problem import LeastSquaresProblem, ResidualModel
from .result import LeastSquaresResult
from .termination import TerminationReason


struct _LmProposal(Copyable):
    """A finite descent step and its positive quadratic-model reduction."""

    var step: List[Float64]
    var predicted_reduction: Float64

    def __init__(out self, var step: List[Float64], predicted_reduction: Float64):
        self.step = step^
        self.predicted_reduction = predicted_reduction


def least_squares[
    M: ResidualModel
](mut problem: LeastSquaresProblem[M]) raises -> LeastSquaresResult:
    """Solve one validated dense nonlinear least-squares problem.

    The problem is borrowed mutably so a stateful model remains owned by the
    caller and its callback state is observable after the solve. Problem
    configuration is validated once on entry; callback results are checked
    directly inside the loop against the solve-entry residual dimension.
    """
    problem.validate()

    var parameters = problem.initial_parameters.copy()
    var expected_residual_count = problem.residual_count
    var parameter_count = len(parameters)
    var residual_evaluations = 0
    var jacobian_evaluations = 0
    var iterations = 0
    var initial_is_finite = True
    var raw_residuals = _evaluate_model(
        problem.model,
        parameters,
        expected_residual_count,
        residual_evaluations,
        initial_is_finite,
    )
    if not initial_is_finite:
        raise Error("model residuals must be finite at the initial point")

    var initial_cost = _objective_cost(
        raw_residuals,
        problem.weights,
        problem.options.loss,
        problem.options.loss_scale,
    )
    if (
        residual_evaluations + parameter_count
        > problem.options.max_residual_evaluations
    ):
        # No gradient can be formed without violating the atomic Jacobian
        # reservation. A finite sentinel is required by the public report.
        return _make_result(
            parameters,
            initial_cost,
            0.0,
            iterations,
            residual_evaluations,
            jacobian_evaluations,
            TerminationReason.MAX_EVALUATIONS,
        )

    var relative_step = _DEFAULT_RELATIVE_STEP
    if problem.options.finite_difference_step:
        relative_step = problem.options.finite_difference_step.value()
    var raw_jacobian = _forward_difference_jacobian(
        problem.model,
        parameters,
        raw_residuals,
        residual_evaluations,
        relative_step=relative_step,
    )
    jacobian_evaluations += 1
    var objective: _ObjectiveModel
    try:
        objective = _build_objective_model(
            raw_residuals,
            raw_jacobian,
            problem.weights,
            problem.options.loss,
            problem.options.loss_scale,
        )
    except:
        return _make_result(
            parameters,
            initial_cost,
            0.0,
            iterations,
            residual_evaluations,
            jacobian_evaluations,
            TerminationReason.NUMERICAL_FAILURE,
        )

    if problem.options.gtol:
        if objective.optimality <= problem.options.gtol.value():
            return _make_result(
                parameters,
                objective.cost,
                objective.optimality,
                iterations,
                residual_evaluations,
                jacobian_evaluations,
                TerminationReason.GRADIENT_TOLERANCE,
            )
    if residual_evaluations >= problem.options.max_residual_evaluations:
        return _make_result(
            parameters,
            objective.cost,
            objective.optimality,
            iterations,
            residual_evaluations,
            jacobian_evaluations,
            TerminationReason.MAX_EVALUATIONS,
        )

    var damping = problem.options.initial_damping
    while True:
        # Reserve one trial call and, if accepted, every finite-difference
        # column needed to make the new accepted state reportable.
        if (
            residual_evaluations + 1 + parameter_count
            > problem.options.max_residual_evaluations
        ):
            return _make_result(
                parameters,
                objective.cost,
                objective.optimality,
                iterations,
                residual_evaluations,
                jacobian_evaluations,
                TerminationReason.MAX_EVALUATIONS,
            )

        var proposal: _LmProposal
        try:
            proposal = _lm_step(objective.jacobian, objective.gradient, damping)
        except:
            if damping >= problem.options.max_damping:
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.NUMERICAL_FAILURE,
                )
            damping = _increase_damping(damping, problem.options.max_damping)
            continue

        var trial_parameters = parameters.copy()
        var trial_parameters_are_finite = True
        for col in range(parameter_count):
            trial_parameters[col] += proposal.step[col]
            if not isfinite(trial_parameters[col]):
                trial_parameters_are_finite = False
        if not trial_parameters_are_finite:
            if damping >= problem.options.max_damping:
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.NUMERICAL_FAILURE,
                )
            damping = _increase_damping(damping, problem.options.max_damping)
            continue

        var trial_is_finite = True
        var trial_raw_residuals = _evaluate_model(
            problem.model,
            trial_parameters,
            expected_residual_count,
            residual_evaluations,
            trial_is_finite,
        )

        var trial_cost = 0.0
        var trial_cost_is_finite = trial_is_finite
        if trial_cost_is_finite:
            try:
                trial_cost = _objective_cost(
                    trial_raw_residuals,
                    problem.weights,
                    problem.options.loss,
                    problem.options.loss_scale,
                )
            except:
                trial_cost_is_finite = False

        iterations += 1
        if not trial_cost_is_finite:
            if damping >= problem.options.max_damping:
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.NUMERICAL_FAILURE,
                )
            damping = _increase_damping(damping, problem.options.max_damping)
            if iterations >= problem.options.max_iterations:
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.MAX_ITERATIONS,
                )
            if residual_evaluations >= problem.options.max_residual_evaluations:
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.MAX_EVALUATIONS,
                )
            continue

        var actual_reduction = objective.cost - trial_cost
        var ratio = actual_reduction / proposal.predicted_reduction
        var accepted = actual_reduction > 0.0 and ratio > 0.0
        if not accepted:
            if damping >= problem.options.max_damping:
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.NUMERICAL_FAILURE,
                )
            damping = _increase_damping(damping, problem.options.max_damping)
            if iterations >= problem.options.max_iterations:
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.MAX_ITERATIONS,
                )
            if residual_evaluations >= problem.options.max_residual_evaluations:
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.MAX_EVALUATIONS,
                )
            continue

        var previous_cost = objective.cost
        parameters = trial_parameters^
        raw_residuals = trial_raw_residuals^
        damping = _update_damping(
            damping,
            ratio,
            problem.options.min_damping,
            problem.options.max_damping,
        )

        # The trial-plus-Jacobian reservation above makes this complete
        # accepted-state Jacobian atomic with respect to the callback budget.
        raw_jacobian = _forward_difference_jacobian(
            problem.model,
            parameters,
            raw_residuals,
            residual_evaluations,
            relative_step=relative_step,
        )
        jacobian_evaluations += 1
        try:
            objective = _build_objective_model(
                raw_residuals,
                raw_jacobian,
                problem.weights,
                problem.options.loss,
                problem.options.loss_scale,
            )
        except:
            return _make_result(
                parameters,
                trial_cost,
                0.0,
                iterations,
                residual_evaluations,
                jacobian_evaluations,
                TerminationReason.NUMERICAL_FAILURE,
            )

        # Accepted-state precedence: gradient, step, cost, iteration budget,
        # then residual-evaluation budget.
        if problem.options.gtol:
            if objective.optimality <= problem.options.gtol.value():
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.GRADIENT_TOLERANCE,
                )
        if problem.options.xtol:
            if _norm2(proposal.step) <= problem.options.xtol.value() * (
                problem.options.xtol.value() + _norm2(parameters)
            ):
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.STEP_TOLERANCE,
                )
        if problem.options.ftol:
            if abs(previous_cost - objective.cost) <= (
                problem.options.ftol.value() * max(1.0, previous_cost)
            ):
                return _make_result(
                    parameters,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.COST_TOLERANCE,
                )
        if iterations >= problem.options.max_iterations:
            return _make_result(
                parameters,
                objective.cost,
                objective.optimality,
                iterations,
                residual_evaluations,
                jacobian_evaluations,
                TerminationReason.MAX_ITERATIONS,
            )
        if residual_evaluations >= problem.options.max_residual_evaluations:
            return _make_result(
                parameters,
                objective.cost,
                objective.optimality,
                iterations,
                residual_evaluations,
                jacobian_evaluations,
                TerminationReason.MAX_EVALUATIONS,
            )


def _evaluate_model[
    M: ResidualModel
](
    mut model: M,
    parameters: List[Float64],
    expected_residual_count: Int,
    mut residual_evaluations: Int,
    mut result_is_finite: Bool,
) raises -> List[Float64]:
    """Call a trusted model and enforce the solve-entry output contract."""
    if model.residual_count() != expected_residual_count:
        raise Error("model residual count changed during the solve")
    residual_evaluations += 1
    var residuals = model.residuals(parameters)
    if model.residual_count() != expected_residual_count:
        raise Error("model residual count changed during the solve")
    if len(residuals) != expected_residual_count:
        raise Error("model returned an unexpected residual count")

    result_is_finite = True
    for row in range(expected_residual_count):
        if not isfinite(residuals[row]):
            result_is_finite = False
    return residuals^


def _lm_step(
    jacobian: _DenseMatrix, gradient: List[Float64], damping: Float64
) raises -> _LmProposal:
    """Solve the diagonally scaled LM system and predict model reduction.

    ``D[j] = max((J^T J)[j, j], 1e-15)``. Predicted reduction uses the
    standard quadratic model
    ``-step^T gradient - 0.5 * step^T (J^T J) step``.
    """
    if len(gradient) != jacobian.cols:
        raise Error("LM gradient length must match the Jacobian columns")
    if not isfinite(damping) or damping <= 0.0:
        raise Error("LM damping must be finite and positive")

    var normal = _jt_j(jacobian)
    var diagonal = List[Float64](length=normal.rows, fill=0.0)
    for index in range(normal.rows):
        var value = normal._values[index * normal.cols + index]
        if not isfinite(value):
            raise Error("LM normal matrix is not finite")
        diagonal[index] = max(value, 1.0e-15)

    var damped_normal = normal.copy()
    _add_diagonal_damping(damped_normal, damping, diagonal)
    var right_hand_side = List[Float64](length=len(gradient), fill=0.0)
    for index in range(len(gradient)):
        if not isfinite(gradient[index]):
            raise Error("LM gradient is not finite")
        right_hand_side[index] = -gradient[index]

    var step = _solve_spd(damped_normal, right_hand_side)
    for index in range(len(step)):
        if not isfinite(step[index]):
            raise Error("LM step is not finite")

    var quadratic = _quadratic_form(normal, step)
    var predicted_reduction = -_dot(step, gradient) - 0.5 * quadratic
    if not isfinite(predicted_reduction) or predicted_reduction <= 0.0:
        raise Error("LM proposal is not a finite descent step")
    return _LmProposal(step^, predicted_reduction)


def _quadratic_form(matrix: _DenseMatrix, values: List[Float64]) raises -> Float64:
    if matrix.rows != matrix.cols or matrix.rows != len(values):
        raise Error("quadratic-form dimensions must match")
    var result = 0.0
    for row in range(matrix.rows):
        var row_product = 0.0
        for col in range(matrix.cols):
            row_product += matrix._values[row * matrix.cols + col] * values[col]
        result += values[row] * row_product
    if not isfinite(result):
        raise Error("quadratic form is not finite")
    return result


def _increase_damping(damping: Float64, max_damping: Float64) -> Float64:
    if damping >= 0.5 * max_damping:
        return max_damping
    return min(max_damping, 2.0 * damping)


def _update_damping(
    damping: Float64,
    ratio: Float64,
    min_damping: Float64,
    max_damping: Float64,
) -> Float64:
    if ratio > 0.75:
        return max(min_damping, damping / 3.0)
    if ratio < 0.25:
        return _increase_damping(damping, max_damping)
    return min(max_damping, max(min_damping, damping))


def _make_result(
    parameters: List[Float64],
    cost: Float64,
    optimality: Float64,
    iterations: Int,
    residual_evaluations: Int,
    jacobian_evaluations: Int,
    termination: TerminationReason,
) raises -> LeastSquaresResult:
    return LeastSquaresResult(
        parameters,
        cost=cost,
        optimality=optimality,
        iterations=iterations,
        residual_evaluations=residual_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        termination=termination,
    )
