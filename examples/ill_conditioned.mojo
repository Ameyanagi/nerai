"""Fit two strongly correlated columns through Nerai's adaptive QR path."""

from nerai import LeastSquaresProblem, ResidualModel, least_squares
from std.collections import List


struct CorrelatedLinearModel(Copyable, ResidualModel):
    def __init__(out self):
        pass

    def residual_count(self) -> Int:
        return 16

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var residuals = List[Float64](length=16, fill=0.0)
        for row in range(16):
            var x = (Float64(row) - 7.5) / 8.0
            var correlated = 1.0 + 1.0e-4 * x
            var observed = 2.0 - correlated
            residuals[row] = parameters[0] + parameters[1] * correlated - observed
        return residuals^


def main() raises:
    var problem = LeastSquaresProblem(CorrelatedLinearModel(), [0.0, 0.0])
    print(least_squares(problem), end="")
