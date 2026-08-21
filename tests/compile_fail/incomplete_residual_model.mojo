from nerai import LeastSquaresProblem


struct IncompleteModel(Copyable):
    def __init__(out self):
        pass

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        return [parameters[0]]


def main() raises:
    _ = LeastSquaresProblem(IncompleteModel(), [1.0])
