"""Inspectable nonlinear least-squares result values."""

from std.collections import List
from std.utils.numerics import isfinite

from .termination import TerminationReason


struct LeastSquaresResult(Copyable):
    """The last valid solver state and its explicit termination report.

    ``cost`` follows Nerai's objective convention and ``optimality`` is the
    infinity norm of the gradient. Evaluation counters count completed user
    callback calls, including calls used for numerical Jacobians. Every result
    includes the initial residual evaluation, so completed iterations cannot
    exceed ``residual_evaluations - 1``. Jacobian evaluations cannot exceed
    residual evaluations.
    """

    var parameters: List[Float64]
    var cost: Float64
    var optimality: Float64
    var iterations: Int
    var residual_evaluations: Int
    var jacobian_evaluations: Int
    var termination: TerminationReason

    def __init__(
        out self,
        parameters: List[Float64],
        *,
        cost: Float64,
        optimality: Float64,
        iterations: Int,
        residual_evaluations: Int,
        jacobian_evaluations: Int,
        termination: TerminationReason,
    ) raises:
        self.parameters = parameters.copy()
        self.cost = cost
        self.optimality = optimality
        self.iterations = iterations
        self.residual_evaluations = residual_evaluations
        self.jacobian_evaluations = jacobian_evaluations
        self.termination = termination
        self.validate()

    def converged(self) -> Bool:
        """Return whether termination represents successful convergence."""
        return self.termination.is_success()

    def validate(self) raises:
        """Revalidate public report fields after possible caller mutation."""
        if len(self.parameters) == 0:
            raise Error("least-squares result requires at least one parameter")
        for index in range(len(self.parameters)):
            if not isfinite(self.parameters[index]):
                raise Error("result parameters must be finite")
        if not isfinite(self.cost) or self.cost < 0.0:
            raise Error("result cost must be finite and non-negative")
        if not isfinite(self.optimality) or self.optimality < 0.0:
            raise Error("result optimality must be finite and non-negative")
        if self.iterations < 0:
            raise Error("iteration count must be non-negative")
        if self.residual_evaluations < 1:
            raise Error("result requires at least one residual evaluation")
        if self.jacobian_evaluations < 0:
            raise Error("Jacobian evaluation count must be non-negative")
        if self.iterations > self.residual_evaluations - 1:
            raise Error("iteration count cannot exceed completed trials")
        if self.jacobian_evaluations > self.residual_evaluations:
            raise Error("Jacobian evaluations cannot exceed residual evaluations")
