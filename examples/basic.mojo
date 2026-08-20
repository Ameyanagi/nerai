from nerai import (
    LeastSquaresProblem,
    ResidualModel,
    TerminationReason,
    least_squares,
)
from std.math import exp


struct ExponentialDecayModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 6

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var amplitude = parameters[0]
        var rate = parameters[1]
        # Exact observations follow y(t)=2.5*exp(-0.7*t) at
        # t=[0, 0.5, 1, 1.5, 2, 2.5].
        return [
            amplitude - 2.5,
            amplitude * exp(-0.5 * rate) - 2.5 * exp(-0.35),
            amplitude * exp(-rate) - 2.5 * exp(-0.7),
            amplitude * exp(-1.5 * rate) - 2.5 * exp(-1.05),
            amplitude * exp(-2.0 * rate) - 2.5 * exp(-1.4),
            amplitude * exp(-2.5 * rate) - 2.5 * exp(-1.75),
        ]


def termination_name(reason: TerminationReason) -> String:
    if reason == TerminationReason.GRADIENT_TOLERANCE:
        return "gradient tolerance"
    if reason == TerminationReason.STEP_TOLERANCE:
        return "step tolerance"
    if reason == TerminationReason.COST_TOLERANCE:
        return "cost tolerance"
    if reason == TerminationReason.MAX_ITERATIONS:
        return "maximum iterations"
    if reason == TerminationReason.MAX_EVALUATIONS:
        return "maximum residual evaluations"
    if reason == TerminationReason.NUMERICAL_FAILURE:
        return "numerical failure"
    return "unknown"


def main() raises:
    var problem = LeastSquaresProblem(ExponentialDecayModel(), [1.5, 0.3])
    var result = least_squares(problem)

    print("amplitude:", result.parameters[0])
    print("decay rate:", result.parameters[1])
    print("cost:", result.cost)
    print("iterations:", result.iterations)
    print("residual evaluations:", result.residual_evaluations)
    print("Jacobian evaluations:", result.jacobian_evaluations)
    print("termination:", termination_name(result.termination))
