"""Explicit nonlinear least-squares termination values."""

from std.io import Writable, Writer


struct TerminationReason(Copyable, Equatable, ImplicitlyCopyable, Writable):
    """The single reason a solver stopped.

    Tolerance reasons are successful convergence. Budget limits preserve a
    valid iterate but are not convergence. Numerical failure indicates that
    the algorithm could not safely continue from its last valid iterate.
    """

    comptime GRADIENT_TOLERANCE = TerminationReason(0)
    comptime STEP_TOLERANCE = TerminationReason(1)
    comptime COST_TOLERANCE = TerminationReason(2)
    comptime MAX_ITERATIONS = TerminationReason(3)
    comptime MAX_EVALUATIONS = TerminationReason(4)
    comptime NUMERICAL_FAILURE = TerminationReason(5)

    var _value: Int

    def __init__(out self, value: Int):
        self._value = value

    def __eq__(self, other: Self) -> Bool:
        """Return whether two values represent the same termination reason."""
        return self._value == other._value

    def __str__(self) -> String:
        """Return the stable human-readable termination phrase."""
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        """Write the stable human-readable termination phrase."""
        if self == Self.GRADIENT_TOLERANCE:
            writer.write("gradient tolerance")
        elif self == Self.STEP_TOLERANCE:
            writer.write("step tolerance")
        elif self == Self.COST_TOLERANCE:
            writer.write("cost tolerance")
        elif self == Self.MAX_ITERATIONS:
            writer.write("max iterations")
        elif self == Self.MAX_EVALUATIONS:
            writer.write("max evaluations")
        else:
            writer.write("numerical failure")

    def is_success(self) -> Bool:
        """Return whether the solver satisfied a convergence tolerance."""
        return (
            self == Self.GRADIENT_TOLERANCE
            or self == Self.STEP_TOLERANCE
            or self == Self.COST_TOLERANCE
        )

    def is_limit(self) -> Bool:
        """Return whether an iteration or evaluation budget stopped work."""
        return self == Self.MAX_ITERATIONS or self == Self.MAX_EVALUATIONS

    def is_failure(self) -> Bool:
        """Return whether a numerical condition prevented safe progress."""
        return self == Self.NUMERICAL_FAILURE
