from nerai import (
    JacobianScheme,
    LeastSquaresOptions,
    LeastSquaresProblem,
    LossKind,
    ResidualModel,
    TerminationReason,
    least_squares,
)
from nerai._kernel import _DenseMatrix
from nerai.solve import _increase_damping, _lm_step, _update_damping
from std.math import abs, exp
from std.testing import TestSuite, assert_equal, assert_raises, assert_true


struct AffineFitModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 5

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        var slope = parameters[0]
        var intercept = parameters[1]
        # Exact observations from y=2*t-1 at t=[-2,-1,0,1,2].
        return [
            -2.0 * slope + intercept + 5.0,
            -slope + intercept + 3.0,
            intercept + 1.0,
            slope + intercept - 1.0,
            2.0 * slope + intercept - 3.0,
        ]


struct DecayModel(Copyable, ResidualModel):
    var calls: Int
    var outlier: Bool

    def __init__(out self, *, outlier: Bool = False):
        self.calls = 0
        self.outlier = outlier

    def residual_count(self) -> Int:
        return 6

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        var amplitude = parameters[0]
        var rate = parameters[1]
        # Clean observations are generated exactly in Float64 from
        # y(t)=2.5*exp(-0.7*t), t=[0,0.5,1,1.5,2,2.5]. The contaminated
        # fixture adds 4.0 only to the observation at t=1.5.
        var outlier_shift = 0.0
        if self.outlier:
            outlier_shift = 4.0
        return [
            amplitude - 2.5,
            amplitude * exp(-0.5 * rate) - 2.5 * exp(-0.35),
            amplitude * exp(-rate) - 2.5 * exp(-0.7),
            amplitude * exp(-1.5 * rate) - (2.5 * exp(-1.05) + outlier_shift),
            amplitude * exp(-2.0 * rate) - 2.5 * exp(-1.4),
            amplitude * exp(-2.5 * rate) - 2.5 * exp(-1.75),
        ]


struct ConstantResidualModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [1.0, -2.0]


struct SquareResidualModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 1

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [parameters[0] * parameters[0] - 1.0]


struct OneParameterLinearModel(Copyable, ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 1

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [parameters[0] - 2.0]


struct RaisingInitialModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 1

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        raise Error("initial model failure")


struct DriftingSolveModel(Copyable, ResidualModel):
    var calls: Int
    var declared_count: Int

    def __init__(out self):
        self.calls = 0
        self.declared_count = 2

    def residual_count(self) -> Int:
        return self.declared_count

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        if self.calls == 2:
            self.declared_count = 3
        return [parameters[0], parameters[1]]


struct ScaledDecayModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 11

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var residuals = List[Float64](length=11, fill=0.0)
        for index in range(11):
            var t = 200.0 * Float64(index)
            residuals[index] = parameters[0] * exp(-parameters[1] * t) - 2.5e6 * exp(
                -1.0e-3 * t
            )
        return residuals^


def tight_options(
    *, loss: LossKind = LossKind.LINEAR, loss_scale: Float64 = 1.0
) raises -> LeastSquaresOptions:
    return LeastSquaresOptions(
        loss=loss,
        loss_scale=loss_scale,
        ftol=1.0e-14,
        xtol=1.0e-14,
        gtol=1.0e-14,
        max_iterations=100,
        max_residual_evaluations=1000,
    )


def max_parameter_error(parameters: List[Float64]) -> Float64:
    return max(abs(parameters[0] - 2.5), abs(parameters[1] - 0.7))


def test_lm_step_matches_hand_fixture_and_damping_reduces_norm() raises:
    # J=[2], f=[4], g=8, J^T J=4, and D=4. At lambda=0.5,
    # (4+0.5*4)s=-8 gives s=-4/3. The quadratic prediction is 64/9.
    var jacobian = _DenseMatrix(1, 1)
    jacobian.set(0, 0, 2.0)
    var low = _lm_step(jacobian, [8.0], 0.5)
    var high = _lm_step(jacobian, [8.0], 2.0)

    assert_true(abs(low.step[0] + 4.0 / 3.0) <= 1.0e-12)
    assert_true(abs(low.predicted_reduction - 64.0 / 9.0) <= 1.0e-12)
    assert_true(low.predicted_reduction > 0.0)
    assert_true(abs(high.step[0]) < abs(low.step[0]))

    # For J=diag(1,2), g=[2,-8], and lambda=1, the damped system is
    # diag(2,8)*s=[-2,8], hence s=[-1,1] and pred=10-2.5=7.5.
    var two_parameter_jacobian = _DenseMatrix(2, 2)
    two_parameter_jacobian.set(0, 0, 1.0)
    two_parameter_jacobian.set(1, 1, 2.0)
    var two_parameter = _lm_step(two_parameter_jacobian, [2.0, -8.0], 1.0)
    assert_true(abs(two_parameter.step[0] + 1.0) <= 1.0e-12)
    assert_true(abs(two_parameter.step[1] - 1.0) <= 1.0e-12)
    assert_true(abs(two_parameter.predicted_reduction - 7.5) <= 1.0e-12)


def test_damping_policy_matches_fixed_thresholds_and_bounds() raises:
    assert_true(_update_damping(0.9, 0.8, 0.1, 10.0) == 0.3)
    assert_true(_update_damping(0.9, 0.5, 0.1, 10.0) == 0.9)
    assert_true(_update_damping(0.9, 0.1, 0.1, 10.0) == 1.8)
    assert_true(_update_damping(0.1, 0.8, 0.1, 10.0) == 0.1)
    assert_true(_increase_damping(8.0, 10.0) == 10.0)


def test_exact_affine_fit_meets_release_gate_and_counter_contract() raises:
    var problem = LeastSquaresProblem(
        AffineFitModel(), [0.0, 0.0], options=tight_options()
    )
    var result = least_squares(problem)

    # Closed-form normal equations for the symmetric t values give
    # slope=sum(t*y)/sum(t^2)=20/10=2 and intercept=mean(y)=-1.
    assert_true(abs(result.parameters[0] - 2.0) <= 2.0e-10)
    assert_true(abs(result.parameters[1] + 1.0) <= 1.0e-10)
    assert_true(result.cost < 1.0e-20)
    assert_true(result.converged())
    assert_equal(problem.model.calls, result.residual_evaluations)
    result.validate()


def test_clean_exponential_decay_recovers_generating_parameters() raises:
    var problem = LeastSquaresProblem(DecayModel(), [1.5, 0.3], options=tight_options())
    var result = least_squares(problem)

    assert_true(max_parameter_error(result.parameters) <= 2.0e-8)
    assert_true(result.cost <= 1.0e-20)
    assert_true(result.converged())
    assert_equal(problem.model.calls, result.residual_evaluations)


def test_explicit_x_scale_handles_widely_separated_parameter_scales() raises:
    var problem = LeastSquaresProblem(
        ScaledDecayModel(),
        [1.0e6, 5.0e-3],
        options=LeastSquaresOptions(x_scale=Optional[List[Float64]]([1.0e6, 1.0e-3])),
    )
    var result = least_squares(problem)

    assert_true(abs(result.parameters[0] / 2.5e6 - 1.0) <= 1.0e-6)
    assert_true(abs(result.parameters[1] / 1.0e-3 - 1.0) <= 1.0e-6)


def test_robust_losses_reduce_one_outlier_parameter_error() raises:
    # C=0.1 is small relative to the single +4 observation corruption but
    # leaves the exact clean residuals in the quadratic neighborhood.
    var linear_problem = LeastSquaresProblem(
        DecayModel(outlier=True),
        [1.5, 0.3],
        options=tight_options(loss=LossKind.LINEAR),
    )
    var huber_problem = LeastSquaresProblem(
        DecayModel(outlier=True),
        [1.5, 0.3],
        options=tight_options(loss=LossKind.HUBER, loss_scale=0.1),
    )
    var soft_l1_problem = LeastSquaresProblem(
        DecayModel(outlier=True),
        [1.5, 0.3],
        options=tight_options(loss=LossKind.SOFT_L1, loss_scale=0.1),
    )

    var linear = least_squares(linear_problem)
    var huber = least_squares(huber_problem)
    var soft_l1 = least_squares(soft_l1_problem)
    var linear_error = max_parameter_error(linear.parameters)
    var huber_error = max_parameter_error(huber.parameters)
    var soft_l1_error = max_parameter_error(soft_l1.parameters)

    assert_true(huber_error < 0.25 * linear_error)
    assert_true(soft_l1_error < 0.25 * linear_error)


def test_termination_reasons_and_rejected_state_preservation() raises:
    var convergence_problem = LeastSquaresProblem(
        AffineFitModel(), [0.0, 0.0], options=tight_options()
    )
    var convergence = least_squares(convergence_problem)
    assert_true(convergence.termination.is_success())

    var iteration_problem = LeastSquaresProblem(
        AffineFitModel(),
        [0.0, 0.0],
        options=LeastSquaresOptions(
            ftol=None,
            xtol=None,
            gtol=None,
            max_iterations=1,
            max_residual_evaluations=100,
        ),
    )
    var iteration_limit = least_squares(iteration_problem)
    assert_true(iteration_limit.termination == TerminationReason.MAX_ITERATIONS)
    assert_equal(iteration_limit.iterations, 1)

    # With n=2, the initial call fits but the complete two-column Jacobian
    # does not. The solver must return without starting a partial Jacobian.
    var evaluation_problem = LeastSquaresProblem(
        AffineFitModel(),
        [0.0, 0.0],
        options=LeastSquaresOptions(max_residual_evaluations=2),
    )
    var evaluation_limit = least_squares(evaluation_problem)
    assert_true(evaluation_limit.termination == TerminationReason.MAX_EVALUATIONS)
    assert_equal(evaluation_limit.residual_evaluations, 1)
    assert_equal(evaluation_limit.jacobian_evaluations, 0)
    assert_equal(evaluation_problem.model.calls, 1)

    # Central differences need two calls per column. The initial residual fits
    # this budget, but the complete one-column central Jacobian does not.
    var central_evaluation_problem = LeastSquaresProblem(
        OneParameterLinearModel(),
        [0.0],
        options=LeastSquaresOptions(
            jacobian_scheme=JacobianScheme.CENTRAL,
            max_residual_evaluations=2,
        ),
    )
    var central_evaluation_limit = least_squares(central_evaluation_problem)
    assert_true(
        central_evaluation_limit.termination == TerminationReason.MAX_EVALUATIONS
    )
    assert_equal(central_evaluation_limit.residual_evaluations, 1)
    assert_equal(central_evaluation_limit.jacobian_evaluations, 0)
    assert_equal(central_evaluation_problem.model.calls, 1)

    # At x=0.1 for r=x^2-1, the first Gauss-Newton proposal overshoots and
    # raises the cost. A one-iteration budget exposes rejected-state retention.
    var rejected_problem = LeastSquaresProblem(
        SquareResidualModel(),
        [0.1],
        options=LeastSquaresOptions(
            ftol=None,
            xtol=None,
            gtol=None,
            max_iterations=1,
            max_residual_evaluations=20,
        ),
    )
    var rejected = least_squares(rejected_problem)
    assert_true(rejected.termination == TerminationReason.MAX_ITERATIONS)
    assert_true(rejected.parameters[0] == 0.1)
    assert_true(abs(rejected.cost - 0.49005) <= 1.0e-12)
    assert_equal(rejected.iterations, 1)


def test_convergence_precedes_final_iteration_and_evaluation_limits() raises:
    # Counts are initial=1, initial J=1, trial=1, accepted-state J=1.
    # The first step reaches optimality about 0.002, so gradient convergence
    # must win even as both budgets become exhausted on that accepted state.
    var problem = LeastSquaresProblem(
        OneParameterLinearModel(),
        [0.0],
        options=LeastSquaresOptions(
            ftol=None,
            xtol=None,
            gtol=1.0e-2,
            max_iterations=1,
            max_residual_evaluations=4,
        ),
    )
    var result = least_squares(problem)

    assert_true(result.termination == TerminationReason.GRADIENT_TOLERANCE)
    assert_equal(result.iterations, 1)
    assert_equal(result.residual_evaluations, 4)
    assert_equal(problem.model.calls, 4)


def test_numerical_failure_preserves_last_valid_state() raises:
    # J is exactly zero and g is zero while F>0. With convergence tolerances
    # disabled, the LM system produces a zero step and non-positive predicted
    # reduction. Starting at maximum damping makes the failure explicit.
    var problem = LeastSquaresProblem(
        ConstantResidualModel(),
        [3.0, -4.0],
        options=LeastSquaresOptions(
            ftol=None,
            xtol=None,
            gtol=None,
            initial_damping=1.0,
            min_damping=1.0,
            max_damping=1.0,
        ),
    )
    var result = least_squares(problem)

    assert_true(result.termination == TerminationReason.NUMERICAL_FAILURE)
    assert_true(result.parameters[0] == 3.0)
    assert_true(result.parameters[1] == -4.0)
    assert_true(result.cost == 2.5)
    assert_equal(result.iterations, 0)
    assert_equal(problem.model.calls, result.residual_evaluations)


def test_invalid_initial_callback_raises() raises:
    var problem = LeastSquaresProblem(RaisingInitialModel(), [0.0])
    with assert_raises(contains="initial model failure"):
        _ = least_squares(problem)

    var drifting_problem = LeastSquaresProblem(DriftingSolveModel(), [0.0, 0.0])
    with assert_raises(contains="changed during Jacobian evaluation"):
        _ = least_squares(drifting_problem)


def test_repeated_solves_are_exactly_deterministic() raises:
    var first_problem = LeastSquaresProblem(
        AffineFitModel(), [0.0, 0.0], options=tight_options()
    )
    var second_problem = LeastSquaresProblem(
        AffineFitModel(), [0.0, 0.0], options=tight_options()
    )
    var first = least_squares(first_problem)
    var second = least_squares(second_problem)

    assert_true(first.termination == second.termination)
    assert_equal(first.iterations, second.iterations)
    assert_equal(first.residual_evaluations, second.residual_evaluations)
    assert_equal(first.jacobian_evaluations, second.jacobian_evaluations)
    assert_true(first.cost == second.cost)
    assert_true(first.optimality == second.optimality)
    assert_true(first.parameters[0] == second.parameters[0])
    assert_true(first.parameters[1] == second.parameters[1])
    assert_equal(first_problem.model.calls, first.residual_evaluations)
    assert_equal(second_problem.model.calls, second.residual_evaluations)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
