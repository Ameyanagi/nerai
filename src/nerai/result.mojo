"""Inspectable nonlinear least-squares result values."""

from std.collections import List, Optional
from std.io import Writable, Writer
from std.utils.numerics import isfinite

from .termination import TerminationReason


struct LeastSquaresResult(Copyable, Equatable, Writable):
    """The last valid solver state and its explicit termination report.

    ``cost`` follows Nerai's objective convention and ``optimality`` is the
    infinity norm of the gradient. Evaluation counters count completed user
    callback calls, including calls used for numerical Jacobians. Every result
    includes the initial residual evaluation, so completed iterations cannot
    exceed ``residual_evaluations - 1``. Jacobian evaluations cannot exceed
    residual evaluations. ``active_bounds`` follows SciPy's active-mask
    convention: ``-1`` is lower-active, ``0`` is free, and ``1`` is
    upper-active.
    """

    var parameters: List[Float64]
    var cost: Float64
    var optimality: Float64
    var iterations: Int
    var residual_evaluations: Int
    var jacobian_evaluations: Int
    var termination: TerminationReason
    var active_bounds: List[Int]

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
        active_bounds: Optional[List[Int]] = None,
    ) raises:
        self.parameters = parameters.copy()
        self.cost = cost
        self.optimality = optimality
        self.iterations = iterations
        self.residual_evaluations = residual_evaluations
        self.jacobian_evaluations = jacobian_evaluations
        self.termination = termination
        if active_bounds:
            self.active_bounds = active_bounds.value().copy()
        else:
            self.active_bounds = List[Int](length=len(parameters), fill=0)
        self.validate()

    def converged(self) -> Bool:
        """Return whether termination represents successful convergence."""
        return self.termination.is_success()

    def __eq__(self, other: Self) -> Bool:
        """Return whether every public report field is exactly equal."""
        if len(self.parameters) != len(other.parameters) or len(
            self.active_bounds
        ) != len(other.active_bounds):
            return False
        for index in range(len(self.parameters)):
            if (
                self.parameters[index] != other.parameters[index]
                or self.active_bounds[index] != other.active_bounds[index]
            ):
                return False
        return (
            self.cost == other.cost
            and self.optimality == other.optimality
            and self.iterations == other.iterations
            and self.residual_evaluations == other.residual_evaluations
            and self.jacobian_evaluations == other.jacobian_evaluations
            and self.termination == other.termination
        )

    def __str__(self) -> String:
        """Return the stable multiline solver report."""
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        """Write the stable multiline solver report with one trailing newline."""
        writer.write("termination           ", self.termination, "\n")
        writer.write(
            "converged             ", "yes" if self.converged() else "no", "\n"
        )
        writer.write("cost                  ", self.cost, "\n")
        writer.write("optimality            ", self.optimality, "\n")
        writer.write("iterations            ", self.iterations, "\n")
        writer.write("residual evaluations  ", self.residual_evaluations, "\n")
        writer.write("jacobian evaluations  ", self.jacobian_evaluations, "\n")
        for index in range(len(self.parameters)):
            var label = String("parameters[", index, "]")
            writer.write(label)
            for _ in range(label.byte_length(), 22):
                writer.write(" ")
            writer.write(self.parameters[index], "\n")
        for index in range(len(self.active_bounds)):
            if self.active_bounds[index] == 0:
                continue
            var label = String("active bounds[", index, "]")
            writer.write(label)
            for _ in range(label.byte_length(), 22):
                writer.write(" ")
            if self.active_bounds[index] < 0:
                writer.write("lower\n")
            else:
                writer.write("upper\n")

    def validate(self) raises:
        """Revalidate public report fields after possible caller mutation."""
        if len(self.parameters) == 0:
            raise Error(
                String(
                    "result parameters must contain at least one parameter; got ",
                    len(self.parameters),
                )
            )
        for index in range(len(self.parameters)):
            if not isfinite(self.parameters[index]):
                raise Error(
                    String(
                        "result parameters[",
                        index,
                        "] must be finite; got ",
                        self.parameters[index],
                    )
                )
        if len(self.active_bounds) != len(self.parameters):
            raise Error(
                String(
                    "active_bounds count ",
                    len(self.active_bounds),
                    " must equal result parameters count ",
                    len(self.parameters),
                )
            )
        for index in range(len(self.active_bounds)):
            if self.active_bounds[index] < -1 or self.active_bounds[index] > 1:
                raise Error(
                    String(
                        "active_bounds[",
                        index,
                        "] must be -1, 0, or 1; got ",
                        self.active_bounds[index],
                    )
                )
        if not isfinite(self.cost) or self.cost < 0.0:
            raise Error(
                String("result cost must be finite and non-negative; got ", self.cost)
            )
        if not isfinite(self.optimality) or self.optimality < 0.0:
            raise Error(
                String(
                    "result optimality must be finite and non-negative; got ",
                    self.optimality,
                )
            )
        if self.iterations < 0:
            raise Error(
                String("iterations must be non-negative; got ", self.iterations)
            )
        if self.residual_evaluations < 1:
            raise Error(
                String(
                    "residual_evaluations must be at least 1; got ",
                    self.residual_evaluations,
                )
            )
        if self.jacobian_evaluations < 0:
            raise Error(
                String(
                    "jacobian_evaluations must be non-negative; got ",
                    self.jacobian_evaluations,
                )
            )
        if self.iterations > self.residual_evaluations - 1:
            raise Error(
                String(
                    "iterations ",
                    self.iterations,
                    " cannot exceed completed trials ",
                    self.residual_evaluations - 1,
                    " (residual_evaluations - 1)",
                )
            )
        if self.jacobian_evaluations > self.residual_evaluations:
            raise Error(
                String(
                    "jacobian_evaluations ",
                    self.jacobian_evaluations,
                    " cannot exceed residual_evaluations ",
                    self.residual_evaluations,
                )
            )
