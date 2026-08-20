"""Explicit nonlinear least-squares termination values."""

from std.collections import Optional


struct TerminationReason(Copyable, Equatable, ImplicitlyCopyable):
    """The single reason a solver stopped.

    Tolerance reasons are successful convergence. Budget limits preserve a
    valid iterate but are not convergence. Numerical failure indicates that
    the algorithm could not safely continue from its last valid iterate.
    """

    comptime GRADIENT_TOLERANCE = TerminationReason(False, Optional[Bool](None))
    comptime STEP_TOLERANCE = TerminationReason(False, Optional(False))
    comptime COST_TOLERANCE = TerminationReason(False, Optional(True))
    comptime MAX_ITERATIONS = TerminationReason(True, Optional[Bool](None))
    comptime MAX_EVALUATIONS = TerminationReason(True, Optional(False))
    comptime NUMERICAL_FAILURE = TerminationReason(True, Optional(True))

    # Bool x Optional[Bool] has exactly six states. The first three are
    # successful tolerance exits; the final three are two limits and failure.
    var _stopped: Bool
    var _detail: Optional[Bool]

    def __init__(out self, stopped: Bool, detail: Optional[Bool]):
        self._stopped = stopped
        self._detail = detail.copy()

    def __eq__(self, other: Self) -> Bool:
        if self._stopped != other._stopped:
            return False
        if self._detail:
            if not other._detail:
                return False
            return self._detail.value() == other._detail.value()
        if other._detail:
            return False
        return True

    def is_success(self) -> Bool:
        """Return whether the solver satisfied a convergence tolerance."""
        return not self._stopped

    def is_limit(self) -> Bool:
        """Return whether an iteration or evaluation budget stopped work."""
        if not self._stopped:
            return False
        if not self._detail:
            return True
        return not self._detail.value()

    def is_failure(self) -> Bool:
        """Return whether a numerical condition prevented safe progress."""
        if not self._stopped or not self._detail:
            return False
        return self._detail.value()
