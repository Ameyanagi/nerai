"""Statically dispatched residual-model and least-squares problem contracts."""

from std.collections import Optional
from std.utils.numerics import isfinite

from .options import LeastSquaresOptions


trait ResidualModel(Deinitable, Movable):
    """A stateful, statically dispatched residual callback.

    A problem owns its model. ``residuals()`` receives read-only parameters,
    may update model state through ``mut self``, and may raise a model-specific
    ``Error``. ``residual_count()`` must stay constant from entry to return of
    each ``residuals()`` call. Coherent direct mutation may reconfigure a
    problem between standalone evaluations, but never during one callback.
    Problem evaluation checks both the entry declaration and every returned
    residual vector, so violations fail at the callback boundary.
    """

    def residual_count(self) -> Int:
        """Return the model's currently configured number of residuals."""
        ...

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        """Evaluate residuals at one finite parameter vector."""
        ...


struct LeastSquaresProblem[M: ResidualModel](Movable):
    """An owned residual model and validated initial problem data.

    The residual count must be at least the number of parameters. Weights are
    finite and non-negative with at least one positive entry; omitting weights
    creates one unit weight per residual. Public fields support ordinary Mojo
    value use, and every evaluation revalidates them before invoking the model.
    Coherent direct mutation is explicit reconfiguration between evaluations.
    """

    var model: Self.M
    var initial_parameters: List[Float64]
    var weights: List[Float64]
    var residual_count: Int
    var options: LeastSquaresOptions

    def __init__(
        out self,
        var model: Self.M,
        initial_parameters: List[Float64],
        *,
        options: Optional[LeastSquaresOptions] = None,
    ) raises:
        self = Self(
            model^,
            initial_parameters,
            _weights=Optional[List[Float64]](None),
            options=options,
        )

    def __init__(
        out self,
        var model: Self.M,
        initial_parameters: List[Float64],
        *,
        weights: List[Float64],
        options: Optional[LeastSquaresOptions] = None,
    ) raises:
        self = Self(
            model^,
            initial_parameters,
            _weights=Optional(weights.copy()),
            options=options,
        )

    def __init__(
        out self,
        var model: Self.M,
        initial_parameters: List[Float64],
        *,
        var _weights: Optional[List[Float64]],
        options: Optional[LeastSquaresOptions] = None,
    ) raises:
        _validate_parameters(initial_parameters)
        var residual_count = model.residual_count()
        if residual_count < len(initial_parameters):
            raise Error("residual count must be at least the parameter count")
        self.model = model^
        self.initial_parameters = initial_parameters.copy()
        if _weights:
            self.weights = _weights.take()
        else:
            self.weights = List[Float64](length=residual_count, fill=1.0)
        self.residual_count = residual_count
        if options:
            self.options = options.value().copy()
        else:
            self.options = LeastSquaresOptions()
        self.validate()

    def validate(self) raises:
        """Revalidate all reachable problem state without calling the model."""
        self.options.validate()
        _validate_parameters(self.initial_parameters)

        if self.residual_count < len(self.initial_parameters):
            raise Error("residual count must be at least the parameter count")
        if self.model.residual_count() != self.residual_count:
            raise Error("model residual count must match the problem configuration")
        if len(self.weights) != self.residual_count:
            raise Error("weight count must equal the residual count")

        var any_positive = False
        for index in range(len(self.weights)):
            var weight = self.weights[index]
            if not isfinite(weight) or weight < 0.0:
                raise Error("weights must be finite and non-negative")
            if weight > 0.0:
                any_positive = True
        if not any_positive:
            raise Error("at least one weight must be positive")

    def evaluate_initial_residuals(mut self) raises -> List[Float64]:
        """Evaluate the model at the problem's validated initial parameters."""
        var parameters = self.initial_parameters.copy()
        return self.evaluate_residuals(parameters)

    def evaluate_residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        """Evaluate one parameter vector and validate the callback result.

        Model-specific errors propagate unchanged. The configured residual
        count is snapshotted at entry and must remain declared through callback
        return. Successful output must have that length and contain only finite
        values.
        """
        self.validate()
        _validate_parameters(parameters)
        if len(parameters) != len(self.initial_parameters):
            raise Error("parameter count must match the initial parameter count")

        var entry_residual_count = self.residual_count
        var residuals = self.model.residuals(parameters)
        if self.model.residual_count() != entry_residual_count:
            raise Error("model residual count changed during residual evaluation")
        if len(residuals) != entry_residual_count:
            raise Error("model returned an unexpected residual count")
        for index in range(len(residuals)):
            if not isfinite(residuals[index]):
                raise Error("model residuals must be finite")
        return residuals^


def _validate_parameters(parameters: List[Float64]) raises:
    if len(parameters) == 0:
        raise Error("least-squares problems require at least one parameter")
    for index in range(len(parameters)):
        if not isfinite(parameters[index]):
            raise Error("parameters must be finite")
