"""Validated configuration for nonlinear least-squares solvers."""

from std.collections import Optional
from std.utils.numerics import isfinite

from .loss import LossKind


struct LeastSquaresOptions(Copyable):
    """Shared nonlinear least-squares configuration.

    A tolerance set to ``None`` is disabled. Present tolerances, the loss
    scale, damping values, and an explicit finite-difference step must be
    finite and positive. Evaluation and iteration budgets are positive
    counts. The current release deliberately leaves the relationship between
    those budgets to a solver because numerical-Jacobian policies are not yet
    part of the public API.

    Fields remain writable under Mojo's value semantics. Constructors and
    problem evaluation boundaries call ``validate()`` so a mutated invalid
    configuration is rejected before a user callback runs.
    """

    var loss: LossKind
    var loss_scale: Float64
    var ftol: Optional[Float64]
    var xtol: Optional[Float64]
    var gtol: Optional[Float64]
    var max_iterations: Int
    var max_residual_evaluations: Int
    var initial_damping: Float64
    var min_damping: Float64
    var max_damping: Float64
    var finite_difference_step: Optional[Float64]

    def __init__(
        out self,
        *,
        loss: LossKind = LossKind.LINEAR,
        loss_scale: Float64 = 1.0,
        ftol: Optional[Float64] = 1.0e-8,
        xtol: Optional[Float64] = 1.0e-8,
        gtol: Optional[Float64] = 1.0e-8,
        max_iterations: Int = 100,
        max_residual_evaluations: Int = 1000,
        initial_damping: Float64 = 1.0e-3,
        min_damping: Float64 = 1.0e-15,
        max_damping: Float64 = 1.0e15,
        finite_difference_step: Optional[Float64] = None,
    ) raises:
        self.loss = loss
        self.loss_scale = loss_scale
        self.ftol = ftol.copy()
        self.xtol = xtol.copy()
        self.gtol = gtol.copy()
        self.max_iterations = max_iterations
        self.max_residual_evaluations = max_residual_evaluations
        self.initial_damping = initial_damping
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.finite_difference_step = finite_difference_step.copy()
        self.validate()

    def validate(self) raises:
        """Reject an invalid configuration, including after field mutation."""
        if not isfinite(self.loss_scale) or self.loss_scale <= 0.0:
            raise Error("loss scale must be finite and positive")

        _validate_optional_positive(self.ftol, "ftol")
        _validate_optional_positive(self.xtol, "xtol")
        _validate_optional_positive(self.gtol, "gtol")
        _validate_optional_positive(
            self.finite_difference_step, "finite-difference step"
        )

        if self.max_iterations <= 0:
            raise Error("maximum iterations must be positive")
        if self.max_residual_evaluations <= 0:
            raise Error("maximum residual evaluations must be positive")

        if not isfinite(self.min_damping) or self.min_damping <= 0.0:
            raise Error("minimum damping must be finite and positive")
        if not isfinite(self.initial_damping) or self.initial_damping <= 0.0:
            raise Error("initial damping must be finite and positive")
        if not isfinite(self.max_damping) or self.max_damping <= 0.0:
            raise Error("maximum damping must be finite and positive")
        if self.min_damping > self.initial_damping:
            raise Error("minimum damping cannot exceed initial damping")
        if self.initial_damping > self.max_damping:
            raise Error("initial damping cannot exceed maximum damping")


def _validate_optional_positive(value: Optional[Float64], name: String) raises:
    if value:
        if not isfinite(value.value()) or value.value() <= 0.0:
            raise Error(name + " must be finite and positive when enabled")
