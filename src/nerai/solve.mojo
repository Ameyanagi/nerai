"""Public Levenberg-Marquardt nonlinear least-squares solver."""

from std.collections import Optional
from std.utils.numerics import isfinite

from ._jacobian import (
    _DEFAULT_RELATIVE_STEP,
    _central_difference_jacobian,
    _forward_difference_jacobian,
)
from ._kernel import (
    _DenseMatrix,
    _QrWorkspace,
    _add_diagonal_damping,
    _dot,
    _jt_j,
    _norm2,
    _solve_spd,
    _solve_damped_least_squares_qr,
)
from ._objective import (
    _ObjectiveModel,
    _build_objective_model_owned,
    _objective_cost,
)
from .bounds import Bounds
from .options import JacobianScheme
from .problem import LeastSquaresProblem, ResidualModel
from .result import LeastSquaresResult
from .termination import TerminationReason


struct _LmProposal(Copyable):
    """A finite descent step and its positive quadratic-model reduction."""

    var step: List[Float64]
    var predicted_reduction: Float64
    var linear_reduction: Float64
    var quadratic: Float64

    def __init__(
        out self,
        var step: List[Float64],
        predicted_reduction: Float64,
        linear_reduction: Float64,
        quadratic: Float64,
    ):
        self.step = step^
        self.predicted_reduction = predicted_reduction
        self.linear_reduction = linear_reduction
        self.quadratic = quadratic


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
    var jacobian_callback_cost = _jacobian_callback_cost(
        problem.options.jacobian_scheme, parameter_count
    )
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
        for index in range(len(raw_residuals)):
            if not isfinite(raw_residuals[index]):
                raise Error(
                    String(
                        "model residual[",
                        index,
                        "] is not finite at the initial parameters; got ",
                        raw_residuals[index],
                        " — check the model and initial_parameters",
                    )
                )

    var initial_cost = _objective_cost(
        raw_residuals,
        problem.weights,
        problem.options.loss,
        problem.options.loss_scale,
    )
    if (
        residual_evaluations + jacobian_callback_cost
        > problem.options.max_residual_evaluations
    ):
        # No gradient can be formed without violating the atomic Jacobian
        # reservation. A finite sentinel is required by the public report.
        return _make_result(
            problem,
            parameters,
            raw_residuals,
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
    var raw_jacobian = _numerical_jacobian(
        problem.model,
        parameters,
        raw_residuals,
        residual_evaluations,
        problem.options.jacobian_scheme,
        relative_step=relative_step,
        bounds=problem.bounds,
    )
    jacobian_evaluations += 1
    var objective: _ObjectiveModel
    try:
        objective = _build_objective_model_owned(
            raw_residuals,
            raw_jacobian^,
            problem.weights,
            problem.options.loss,
            problem.options.loss_scale,
        )
    except:
        return _make_result(
            problem,
            parameters,
            raw_residuals,
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
                problem,
                parameters,
                raw_residuals,
                objective.cost,
                objective.optimality,
                iterations,
                residual_evaluations,
                jacobian_evaluations,
                TerminationReason.GRADIENT_TOLERANCE,
            )
    if residual_evaluations >= problem.options.max_residual_evaluations:
        return _make_result(
            problem,
            parameters,
            raw_residuals,
            objective.cost,
            objective.optimality,
            iterations,
            residual_evaluations,
            jacobian_evaluations,
            TerminationReason.MAX_EVALUATIONS,
        )

    var damping = problem.options.initial_damping
    var qr_workspace = _QrWorkspace(
        expected_residual_count + parameter_count, parameter_count
    )
    while True:
        # Reserve one trial call and, if accepted, every finite-difference
        # column needed to make the new accepted state reportable.
        if (
            residual_evaluations + 1 + jacobian_callback_cost
            > problem.options.max_residual_evaluations
        ):
            return _make_result(
                problem,
                parameters,
                raw_residuals,
                objective.cost,
                objective.optimality,
                iterations,
                residual_evaluations,
                jacobian_evaluations,
                TerminationReason.MAX_EVALUATIONS,
            )

        var proposal: _LmProposal
        try:
            proposal = _scaled_lm_step(
                qr_workspace,
                objective,
                problem.options.x_scale,
                parameters,
                problem.bounds,
                problem.options.xtol,
                damping,
            )
        except:
            if damping >= problem.options.max_damping:
                return _make_result(
                    problem,
                    parameters,
                    raw_residuals,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.NUMERICAL_FAILURE,
                )
            damping = _increase_damping(damping, problem.options.max_damping)
            continue

        var alpha = 1.0
        if problem.bounds:
            var bounds = problem.bounds.value().copy()
            alpha = _fraction_to_boundary(parameters, proposal.step, bounds)
            while not _step_is_strictly_feasible(
                parameters, proposal.step, alpha, bounds
            ):
                var smaller_alpha = 0.5 * alpha
                if smaller_alpha == 0.0:
                    alpha = 0.0
                    break
                alpha = smaller_alpha

            if alpha == 0.0:
                if damping >= problem.options.max_damping:
                    return _make_result(
                        problem,
                        parameters,
                        raw_residuals,
                        objective.cost,
                        objective.optimality,
                        iterations,
                        residual_evaluations,
                        jacobian_evaluations,
                        TerminationReason.NUMERICAL_FAILURE,
                    )
                damping = _increase_damping(damping, problem.options.max_damping)
                continue

            if alpha < 1.0:
                proposal.predicted_reduction = _scaled_predicted_reduction(
                    proposal, alpha
                )
                if (
                    not isfinite(proposal.predicted_reduction)
                    or proposal.predicted_reduction <= 0.0
                ):
                    if damping >= problem.options.max_damping:
                        return _make_result(
                            problem,
                            parameters,
                            raw_residuals,
                            objective.cost,
                            objective.optimality,
                            iterations,
                            residual_evaluations,
                            jacobian_evaluations,
                            TerminationReason.NUMERICAL_FAILURE,
                        )
                    damping = _increase_damping(damping, problem.options.max_damping)
                    continue
                for col in range(parameter_count):
                    proposal.step[col] *= alpha

        var trial_parameters = parameters.copy()
        var trial_parameters_are_finite = True
        for col in range(parameter_count):
            trial_parameters[col] += proposal.step[col]
            if not isfinite(trial_parameters[col]):
                trial_parameters_are_finite = False
        if not trial_parameters_are_finite:
            if damping >= problem.options.max_damping:
                return _make_result(
                    problem,
                    parameters,
                    raw_residuals,
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
                    problem,
                    parameters,
                    raw_residuals,
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
                    problem,
                    parameters,
                    raw_residuals,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.MAX_ITERATIONS,
                )
            if residual_evaluations >= problem.options.max_residual_evaluations:
                return _make_result(
                    problem,
                    parameters,
                    raw_residuals,
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
                    problem,
                    parameters,
                    raw_residuals,
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
                    problem,
                    parameters,
                    raw_residuals,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.MAX_ITERATIONS,
                )
            if residual_evaluations >= problem.options.max_residual_evaluations:
                return _make_result(
                    problem,
                    parameters,
                    raw_residuals,
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
        raw_jacobian = _numerical_jacobian(
            problem.model,
            parameters,
            raw_residuals,
            residual_evaluations,
            problem.options.jacobian_scheme,
            relative_step=relative_step,
            bounds=problem.bounds,
        )
        jacobian_evaluations += 1
        try:
            objective = _build_objective_model_owned(
                raw_residuals,
                raw_jacobian^,
                problem.weights,
                problem.options.loss,
                problem.options.loss_scale,
            )
        except:
            return _make_result(
                problem,
                parameters,
                raw_residuals,
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
                    problem,
                    parameters,
                    raw_residuals,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.GRADIENT_TOLERANCE,
                )
        var clipped_step_has_free_coordinate = False
        if alpha < 1.0 and problem.bounds:
            clipped_step_has_free_coordinate = _has_free_bound_coordinate(
                parameters,
                objective.gradient,
                problem.bounds.value(),
                problem.options.xtol,
            )
        if (
            alpha == 1.0 or not clipped_step_has_free_coordinate
        ) and problem.options.xtol:
            if _norm2(proposal.step) <= problem.options.xtol.value() * (
                problem.options.xtol.value() + _norm2(parameters)
            ):
                return _make_result(
                    problem,
                    parameters,
                    raw_residuals,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.STEP_TOLERANCE,
                )
        if (
            alpha == 1.0 or not clipped_step_has_free_coordinate
        ) and problem.options.ftol:
            if abs(previous_cost - objective.cost) <= (
                problem.options.ftol.value() * max(1.0, previous_cost)
            ):
                return _make_result(
                    problem,
                    parameters,
                    raw_residuals,
                    objective.cost,
                    objective.optimality,
                    iterations,
                    residual_evaluations,
                    jacobian_evaluations,
                    TerminationReason.COST_TOLERANCE,
                )
        if iterations >= problem.options.max_iterations:
            return _make_result(
                problem,
                parameters,
                raw_residuals,
                objective.cost,
                objective.optimality,
                iterations,
                residual_evaluations,
                jacobian_evaluations,
                TerminationReason.MAX_ITERATIONS,
            )
        if residual_evaluations >= problem.options.max_residual_evaluations:
            return _make_result(
                problem,
                parameters,
                raw_residuals,
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


def _jacobian_callback_cost(scheme: JacobianScheme, parameter_count: Int) -> Int:
    """Return the atomic residual-callback cost of one Jacobian."""
    if scheme == JacobianScheme.CENTRAL:
        return 2 * parameter_count
    return parameter_count


def _numerical_jacobian[
    M: ResidualModel
](
    mut model: M,
    parameters: List[Float64],
    base_residuals: List[Float64],
    mut evaluations: Int,
    scheme: JacobianScheme,
    *,
    relative_step: Float64,
    bounds: Optional[Bounds],
) raises -> _DenseMatrix:
    """Dispatch one numerical Jacobian without changing callback accounting."""
    if scheme == JacobianScheme.CENTRAL:
        return _central_difference_jacobian(
            model,
            parameters,
            base_residuals,
            evaluations,
            relative_step=relative_step,
            bounds=bounds,
        )
    return _forward_difference_jacobian(
        model,
        parameters,
        base_residuals,
        evaluations,
        relative_step=relative_step,
        bounds=bounds,
    )


def _scaled_lm_step(
    mut qr_workspace: _QrWorkspace,
    objective: _ObjectiveModel,
    x_scale: Optional[List[Float64]],
    parameters: List[Float64],
    bounds: Optional[Bounds],
    xtol: Optional[Float64],
    damping: Float64,
) raises -> _LmProposal:
    """Solve in ``z = x / d`` when an explicit parameter scale is present."""
    if not x_scale and not bounds:
        return _lm_step_with_workspace(
            qr_workspace,
            objective.jacobian,
            objective.residuals,
            objective.gradient,
            damping,
        )

    var model_jacobian = objective.jacobian.copy()
    var model_gradient = objective.gradient.copy()
    if x_scale:
        var scale = x_scale.value().copy()
        for col in range(model_jacobian.cols):
            for row in range(model_jacobian.rows):
                var offset = model_jacobian._offset(row, col)
                model_jacobian._values[offset] *= scale[col]
        for col in range(len(model_gradient)):
            model_gradient[col] *= scale[col]

    # Once an outward-moving variable reaches the reporting tolerance, hold
    # that active coordinate fixed so the remaining tangent variables can
    # finish converging instead of being throttled by ever-smaller fractions.
    if bounds:
        var configured_bounds = bounds.value().copy()
        var active_tolerance = 1.5e-8
        if xtol:
            active_tolerance = xtol.value()
        var outward_active = List[Bool](length=len(parameters), fill=False)
        var outward_active_count = 0
        for col in range(len(parameters)):
            outward_active[col] = _is_outward_active(
                parameters[col],
                objective.gradient[col],
                configured_bounds,
                col,
                active_tolerance,
            )
            if outward_active[col]:
                outward_active_count += 1
        if outward_active_count < len(parameters):
            for col in range(len(parameters)):
                if not outward_active[col]:
                    continue
                model_gradient[col] = 0.0
                for row in range(model_jacobian.rows):
                    model_jacobian._values[model_jacobian._offset(row, col)] = 0.0

    var proposal = _lm_step_with_workspace(
        qr_workspace,
        model_jacobian,
        objective.residuals,
        model_gradient,
        damping,
    )
    if x_scale:
        var scale = x_scale.value().copy()
        for col in range(len(proposal.step)):
            proposal.step[col] *= scale[col]
            if not isfinite(proposal.step[col]):
                raise Error("scaled LM step is not finite")
    return proposal^


def _fraction_to_boundary(
    parameters: List[Float64], step: List[Float64], bounds: Bounds
) -> Float64:
    """Return the largest prescribed strictly-feasible step fraction."""
    var alpha = 1.0
    for index in range(len(parameters)):
        if step[index] > 0.0 and isfinite(bounds.upper(index)):
            var gap = bounds.upper(index) - parameters[index]
            alpha = min(alpha, 0.995 * gap / abs(step[index]))
        elif step[index] < 0.0 and isfinite(bounds.lower(index)):
            var gap = parameters[index] - bounds.lower(index)
            alpha = min(alpha, 0.995 * gap / abs(step[index]))
    return alpha


def _is_outward_active(
    parameter: Float64,
    gradient: Float64,
    bounds: Bounds,
    index: Int,
    relative_tolerance: Float64,
) -> Bool:
    """Return whether descent points out through a tolerance-active bound."""
    var lower = bounds.lower(index)
    var upper = bounds.upper(index)
    var lower_is_active = isfinite(lower) and parameter - lower <= (
        relative_tolerance * max(1.0, abs(lower))
    )
    if lower_is_active:
        return gradient > 0.0
    var upper_is_active = isfinite(upper) and upper - parameter <= (
        relative_tolerance * max(1.0, abs(upper))
    )
    return upper_is_active and gradient < 0.0


def _has_free_bound_coordinate(
    parameters: List[Float64],
    gradient: List[Float64],
    bounds: Bounds,
    xtol: Optional[Float64],
) -> Bool:
    """Return whether at least one coordinate is not outward-active."""
    var relative_tolerance = 1.5e-8
    if xtol:
        relative_tolerance = xtol.value()
    for index in range(len(parameters)):
        if not _is_outward_active(
            parameters[index],
            gradient[index],
            bounds,
            index,
            relative_tolerance,
        ):
            return True
    return False


def _step_is_strictly_feasible(
    parameters: List[Float64],
    step: List[Float64],
    alpha: Float64,
    bounds: Bounds,
) -> Bool:
    """Check a trial step against every finite bound."""
    for index in range(len(parameters)):
        var candidate = parameters[index] + alpha * step[index]
        if not isfinite(candidate):
            return False
        if isfinite(bounds.lower(index)) and candidate <= bounds.lower(index):
            return False
        if isfinite(bounds.upper(index)) and candidate >= bounds.upper(index):
            return False
    return True


def _scaled_predicted_reduction(proposal: _LmProposal, alpha: Float64) -> Float64:
    """Evaluate the LM quadratic at a fraction of its original model step."""
    return alpha * proposal.linear_reduction - 0.5 * alpha * alpha * proposal.quadratic


def _lm_step(
    jacobian: _DenseMatrix,
    residuals: List[Float64],
    gradient: List[Float64],
    damping: Float64,
) raises -> _LmProposal:
    """Allocate one QR fallback workspace for a standalone private LM step."""
    var workspace = _QrWorkspace(jacobian.rows + jacobian.cols, jacobian.cols)
    return _lm_step_with_workspace(workspace, jacobian, residuals, gradient, damping)


def _lm_step_with_workspace(
    mut qr_workspace: _QrWorkspace,
    jacobian: _DenseMatrix,
    residuals: List[Float64],
    gradient: List[Float64],
    damping: Float64,
) raises -> _LmProposal:
    """Solve the diagonally scaled LM system and predict model reduction.

    ``D[j] = max((J^T J)[j, j], 1e-15)``. Predicted reduction uses the
    standard quadratic model
    ``-step^T gradient - 0.5 * step^T (J^T J) step``.
    """
    if len(gradient) != jacobian.cols:
        raise Error("LM gradient length must match the Jacobian columns")
    if len(residuals) != jacobian.rows:
        raise Error("LM residual length must match the Jacobian rows")
    if not isfinite(damping) or damping <= 0.0:
        raise Error("LM damping must be finite and positive")

    var normal = _jt_j(jacobian)
    var diagonal = List[Float64](length=normal.rows, fill=0.0)
    for index in range(normal.rows):
        var value = normal._values[normal._offset(index, index)]
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

    var step: List[Float64]
    try:
        # Damping normally makes the normal system safe and this is the fast
        # path. A small Cholesky pivot triggers the stable augmented QR path
        # before squared conditioning can corrupt the step.
        step = _solve_spd(
            damped_normal,
            right_hand_side,
            minimum_relative_pivot=1.0e-10,
        )
    except:
        step = _solve_damped_least_squares_qr(
            qr_workspace, jacobian, residuals, damping
        )
    for index in range(len(step)):
        if not isfinite(step[index]):
            raise Error("LM step is not finite")

    var quadratic = _quadratic_form(normal, step)
    var linear_reduction = -_dot(step, gradient)
    var predicted_reduction = linear_reduction - 0.5 * quadratic
    if not isfinite(predicted_reduction) or predicted_reduction <= 0.0:
        raise Error("LM proposal is not a finite descent step")
    return _LmProposal(
        step^,
        predicted_reduction,
        linear_reduction,
        quadratic,
    )


def _quadratic_form(matrix: _DenseMatrix, values: List[Float64]) raises -> Float64:
    if matrix.rows != matrix.cols or matrix.rows != len(values):
        raise Error("quadratic-form dimensions must match")
    var result = 0.0
    for row in range(matrix.rows):
        var row_product = 0.0
        for col in range(matrix.cols):
            row_product += matrix._values[matrix._offset(row, col)] * values[col]
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


def _active_bounds(
    parameters: List[Float64],
    bounds: Optional[Bounds],
    xtol: Optional[Float64],
) -> List[Int]:
    """Return the SciPy-style active mask for one accepted parameter vector."""
    var result = List[Int](length=len(parameters), fill=0)
    if not bounds:
        return result^

    var relative_tolerance = 1.5e-8
    if xtol:
        relative_tolerance = xtol.value()
    var configured_bounds = bounds.value().copy()
    for index in range(len(parameters)):
        var lower = configured_bounds.lower(index)
        var upper = configured_bounds.upper(index)
        if isfinite(lower) and parameters[index] - lower <= (
            relative_tolerance * max(1.0, abs(lower))
        ):
            result[index] = -1
        elif isfinite(upper) and upper - parameters[index] <= (
            relative_tolerance * max(1.0, abs(upper))
        ):
            result[index] = 1
    return result^


def _make_result[
    M: ResidualModel
](
    problem: LeastSquaresProblem[M],
    parameters: List[Float64],
    residuals: List[Float64],
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
        residuals=residuals,
        active_bounds=_active_bounds(
            parameters,
            problem.bounds,
            problem.options.xtol,
        ),
    )
