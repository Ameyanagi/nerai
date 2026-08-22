from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)
from std.utils.numerics import inf, nan

from nerai import (
    JacobianScheme,
    LeastSquaresOptions,
    LeastSquaresProblem,
    LossKind,
    ResidualModel,
)


struct AffineModel(Copyable, ResidualModel):
    """A stateful two-parameter, two-residual reference model."""

    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [
            parameters[0] + 2.0 * parameters[1] - 5.0,
            3.0 * parameters[0] - parameters[1] - 4.0,
        ]


struct OneResidualModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 1

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [parameters[0]]


struct WrongLengthModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [parameters[0]]


struct NonfiniteModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [parameters[0], inf[DType.float64]()]


struct RaisingModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 2

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        raise Error("model failure")


struct DriftingCountModel(Copyable, ResidualModel):
    var declared_count: Int

    def __init__(out self):
        self.declared_count = 2

    def residual_count(self) -> Int:
        return self.declared_count

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.declared_count = 3
        return [parameters[0], parameters[1]]


struct ReconfigurableModel(Copyable, ResidualModel):
    var declared_count: Int

    def __init__(out self):
        self.declared_count = 2

    def residual_count(self) -> Int:
        return self.declared_count

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var residuals = List[Float64]()
        for index in range(self.declared_count):
            residuals.append(parameters[0] + Float64(index))
        return residuals^


struct MoveOnlyModel(ResidualModel):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def residual_count(self) -> Int:
        return 1

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        self.calls += 1
        return [parameters[0] - 2.0]


def test_options_defaults_and_disabled_tolerances() raises:
    var defaults = LeastSquaresOptions()
    defaults.validate()

    assert_true(defaults.loss == LossKind.LINEAR)
    assert_equal(defaults.loss_scale, 1.0)
    assert_true(defaults.ftol)
    assert_equal(defaults.ftol.value(), 1.0e-8)
    assert_true(defaults.xtol)
    assert_equal(defaults.xtol.value(), 1.0e-8)
    assert_true(defaults.gtol)
    assert_equal(defaults.gtol.value(), 1.0e-8)
    assert_equal(defaults.max_iterations, 100)
    assert_equal(defaults.max_residual_evaluations, 1000)
    assert_equal(defaults.initial_damping, 1.0e-3)
    assert_equal(defaults.min_damping, 1.0e-15)
    assert_equal(defaults.max_damping, 1.0e15)
    assert_false(defaults.finite_difference_step)
    assert_false(defaults.x_scale)
    assert_true(defaults.jacobian_scheme == JacobianScheme.FORWARD)

    var disabled = LeastSquaresOptions(
        loss=LossKind.SOFT_L1,
        ftol=None,
        xtol=None,
        gtol=None,
        finite_difference_step=1.0e-6,
    )
    disabled.validate()
    assert_true(disabled.loss == LossKind.SOFT_L1)
    assert_false(disabled.ftol)
    assert_false(disabled.xtol)
    assert_false(disabled.gtol)
    assert_true(disabled.finite_difference_step)
    assert_equal(disabled.finite_difference_step.value(), 1.0e-6)


def test_options_reject_invalid_scale_tolerances_and_step() raises:
    with assert_raises(contains="loss_scale must be finite and positive; got 0.0"):
        _ = LeastSquaresOptions(loss_scale=0.0)
    with assert_raises(contains="loss_scale must be finite and positive; got nan"):
        _ = LeastSquaresOptions(loss_scale=nan[DType.float64]())
    with assert_raises(
        contains="ftol must be finite and positive when enabled; got -1.0"
    ):
        _ = LeastSquaresOptions(ftol=-1.0)
    with assert_raises(
        contains="xtol must be finite and positive when enabled; got inf"
    ):
        _ = LeastSquaresOptions(xtol=inf[DType.float64]())
    with assert_raises(
        contains="gtol must be finite and positive when enabled; got -1.0"
    ):
        _ = LeastSquaresOptions(gtol=-1.0)
    with assert_raises(
        contains=(
            "finite_difference_step must be finite and positive when enabled; got 0.0"
        )
    ):
        _ = LeastSquaresOptions(finite_difference_step=0.0)
    with assert_raises(
        contains=(
            "finite_difference_step must be finite and positive when enabled; got inf"
        )
    ):
        _ = LeastSquaresOptions(finite_difference_step=inf[DType.float64]())
    with assert_raises(contains="x_scale[0]"):
        _ = LeastSquaresOptions(x_scale=Optional[List[Float64]]([0.0]))
    with assert_raises(contains="x_scale[1]"):
        _ = LeastSquaresOptions(x_scale=Optional[List[Float64]]([1.0, -2.0]))
    with assert_raises(contains="x_scale[0]"):
        _ = LeastSquaresOptions(x_scale=Optional[List[Float64]]([inf[DType.float64]()]))
    with assert_raises(contains="x_scale[0]"):
        _ = LeastSquaresOptions(x_scale=Optional[List[Float64]]([nan[DType.float64]()]))


def test_options_reject_invalid_budgets_and_damping() raises:
    with assert_raises(contains="max_iterations must be positive; got 0"):
        _ = LeastSquaresOptions(max_iterations=0)
    with assert_raises(contains="max_residual_evaluations must be positive; got -1"):
        _ = LeastSquaresOptions(max_residual_evaluations=-1)
    with assert_raises(contains="initial_damping must be finite and positive; got 0.0"):
        _ = LeastSquaresOptions(initial_damping=0.0)
    with assert_raises(contains="min_damping must be finite and positive; got 0.0"):
        _ = LeastSquaresOptions(min_damping=0.0)
    with assert_raises(contains="max_damping must be finite and positive; got inf"):
        _ = LeastSquaresOptions(max_damping=inf[DType.float64]())
    with assert_raises(contains="min_damping 1.0 cannot exceed initial_damping 0.5"):
        _ = LeastSquaresOptions(
            min_damping=1.0,
            initial_damping=0.5,
            max_damping=2.0,
        )
    with assert_raises(contains="initial_damping 3.0 cannot exceed max_damping 2.0"):
        _ = LeastSquaresOptions(
            min_damping=0.5,
            initial_damping=3.0,
            max_damping=2.0,
        )


def test_mutated_options_are_revalidated() raises:
    var options = LeastSquaresOptions()
    options.max_iterations = 0
    with assert_raises(contains="max_iterations must be positive; got 0"):
        options.validate()

    options.max_iterations = 1
    options.ftol = nan[DType.float64]()
    with assert_raises(
        contains="ftol must be finite and positive when enabled; got nan"
    ):
        options.validate()

    options.ftol = None
    options.min_damping = 2.0
    with assert_raises(contains="min_damping 2.0 cannot exceed initial_damping 0.001"):
        options.validate()


def test_problem_evaluates_stateful_model_through_static_contract() raises:
    var problem = LeastSquaresProblem(
        AffineModel(),
        [1.0, 2.0],
        weights=[1.0, 0.5],
        options=LeastSquaresOptions(loss=LossKind.SOFT_L1),
    )

    var first = problem.evaluate_initial_residuals()
    assert_equal(len(first), 2)
    assert_equal(first[0], 0.0)
    assert_equal(first[1], -3.0)
    assert_equal(problem.model.calls, 1)

    var second = problem.evaluate_residuals([2.0, 1.0])
    assert_equal(second[0], -1.0)
    assert_equal(second[1], 1.0)
    assert_equal(problem.model.calls, 2)


def test_problem_owns_move_only_residual_model() raises:
    var problem = LeastSquaresProblem(MoveOnlyModel(), [2.0])
    var residuals = problem.evaluate_initial_residuals()
    assert_equal(residuals[0], 0.0)
    assert_equal(problem.model.calls, 1)


def test_problem_supplies_unit_weights() raises:
    var problem = LeastSquaresProblem(AffineModel(), [1.0, 2.0])
    assert_equal(len(problem.weights), 2)
    assert_equal(problem.weights[0], 1.0)
    assert_equal(problem.weights[1], 1.0)


def test_problem_rejects_invalid_dimensions_and_parameters() raises:
    with assert_raises(
        contains="initial_parameters must contain at least one parameter; got 0"
    ):
        _ = LeastSquaresProblem(AffineModel(), List[Float64]())
    with assert_raises(contains="initial_parameters[1] must be finite; got nan"):
        _ = LeastSquaresProblem(AffineModel(), [1.0, nan[DType.float64]()])
    with assert_raises(
        contains=(
            "model declares 1 residual for 2 parameters; a least-squares problem "
            "needs at least as many residuals as parameters"
        )
    ):
        _ = LeastSquaresProblem(OneResidualModel(), [1.0, 2.0])
    with assert_raises(contains="x_scale has 3 entries for 2 parameters"):
        _ = LeastSquaresProblem(
            AffineModel(),
            [1.0, 2.0],
            options=LeastSquaresOptions(
                x_scale=Optional[List[Float64]]([1.0, 2.0, 3.0])
            ),
        )

    var problem = LeastSquaresProblem(AffineModel(), [1.0, 2.0])
    with assert_raises(
        contains="parameters count 1 must equal initial_parameters count 2"
    ):
        _ = problem.evaluate_residuals([1.0])
    with assert_raises(contains="parameters[1] must be finite; got inf"):
        _ = problem.evaluate_residuals([1.0, inf[DType.float64]()])


def test_problem_rejects_invalid_weights() raises:
    with assert_raises(contains="weights count 1 must equal residual_count 2"):
        _ = LeastSquaresProblem(AffineModel(), [1.0, 2.0], weights=[1.0])
    with assert_raises(contains="weights[1] must be finite and non-negative; got -1.0"):
        _ = LeastSquaresProblem(AffineModel(), [1.0, 2.0], weights=[1.0, -1.0])
    with assert_raises(contains="weights[1] must be finite and non-negative; got inf"):
        _ = LeastSquaresProblem(
            AffineModel(), [1.0, 2.0], weights=[1.0, inf[DType.float64]()]
        )
    with assert_raises(
        contains="weights must include at least one positive entry; got all zeros"
    ):
        _ = LeastSquaresProblem(AffineModel(), [1.0, 2.0], weights=[0.0, 0.0])


def test_problem_revalidates_reachable_mutation_before_callback() raises:
    var problem = LeastSquaresProblem(AffineModel(), [1.0, 2.0])
    problem.weights[0] = -1.0
    with assert_raises(contains="weights[0] must be finite and non-negative; got -1.0"):
        _ = problem.evaluate_initial_residuals()
    assert_equal(problem.model.calls, 0)

    problem.weights[0] = 1.0
    problem.options.max_residual_evaluations = 0
    with assert_raises(contains="max_residual_evaluations must be positive; got 0"):
        _ = problem.evaluate_initial_residuals()
    assert_equal(problem.model.calls, 0)

    problem.options.max_residual_evaluations = 1
    problem.initial_parameters[0] = nan[DType.float64]()
    with assert_raises(contains="initial_parameters[0] must be finite; got nan"):
        _ = problem.evaluate_initial_residuals()
    assert_equal(problem.model.calls, 0)

    problem.initial_parameters[0] = 1.0
    problem.residual_count = 3
    with assert_raises(
        contains="model residual_count 2 must match problem residual_count 3"
    ):
        _ = problem.evaluate_initial_residuals()
    assert_equal(problem.model.calls, 0)


def test_problem_rejects_invalid_callback_results() raises:
    var wrong_length = LeastSquaresProblem(WrongLengthModel(), [1.0, 2.0])
    with assert_raises(
        contains="model returned a residual vector of length 1; expected length 2"
    ):
        _ = wrong_length.evaluate_initial_residuals()

    var nonfinite = LeastSquaresProblem(NonfiniteModel(), [1.0, 2.0])
    with assert_raises(contains="model residuals[1] must be finite; got inf"):
        _ = nonfinite.evaluate_initial_residuals()

    var raising = LeastSquaresProblem(RaisingModel(), [1.0, 2.0])
    with assert_raises(contains="model failure"):
        _ = raising.evaluate_initial_residuals()


def test_callback_cannot_change_its_entry_residual_dimension() raises:
    var problem = LeastSquaresProblem(DriftingCountModel(), [1.0, 2.0])
    with assert_raises(
        contains=(
            "model residual_count changed during residual evaluation; got 3 "
            "after entering with 2"
        )
    ):
        _ = problem.evaluate_initial_residuals()
    assert_equal(problem.model.declared_count, 3)


def test_coherent_direct_mutation_reconfigures_between_evaluations() raises:
    var problem = LeastSquaresProblem(ReconfigurableModel(), [2.0])
    var initial = problem.evaluate_initial_residuals()
    assert_equal(initial, [2.0, 3.0])

    problem.model.declared_count = 3
    problem.residual_count = 3
    problem.weights = [1.0, 0.5, 0.25]
    var reconfigured = problem.evaluate_initial_residuals()
    assert_equal(reconfigured, [2.0, 3.0, 4.0])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
