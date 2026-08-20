from nerai import (
    LeastSquaresOptions,
    LeastSquaresProblem,
    LossKind,
    ResidualModel,
)


struct AffineModel(Copyable, ResidualModel):
    """Two equations in two parameters with an observable call count."""

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


def main() raises:
    var problem = LeastSquaresProblem(
        AffineModel(),
        [1.0, 2.0],
        weights=[1.0, 0.5],
        options=LeastSquaresOptions(loss=LossKind.SOFT_L1),
    )
    var residuals = problem.evaluate_initial_residuals()

    print("residual 0:", residuals[0])
    print("residual 1:", residuals[1])
    print("model calls:", problem.model.calls)
