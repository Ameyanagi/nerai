"""Statically dispatched residual-model and least-squares problem contracts."""

from std.collections import Optional
from std.utils.numerics import isfinite

from .bounds import Bounds
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
    creates one unit weight per residual. Optional box bounds must match the
    parameter count, and initial parameters must be strictly inside them.
    Public fields support ordinary Mojo value use, and every evaluation
    revalidates them before invoking the model. Coherent direct mutation is
    explicit reconfiguration between evaluations.
    """

    var model: Self.M
    var initial_parameters: List[Float64]
    var weights: List[Float64]
    var residual_count: Int
    var options: LeastSquaresOptions
    var bounds: Optional[Bounds]

    def __init__(
        out self,
        var model: Self.M,
        initial_parameters: List[Float64],
        *,
        options: Optional[LeastSquaresOptions] = None,
        bounds: Optional[Bounds] = None,
    ) raises:
        self = Self(
            model^,
            initial_parameters,
            _weights=Optional[List[Float64]](None),
            options=options,
            bounds=bounds,
        )

    def __init__(
        out self,
        var model: Self.M,
        initial_parameters: List[Float64],
        *,
        weights: List[Float64],
        options: Optional[LeastSquaresOptions] = None,
        bounds: Optional[Bounds] = None,
    ) raises:
        self = Self(
            model^,
            initial_parameters,
            _weights=Optional(weights.copy()),
            options=options,
            bounds=bounds,
        )

    def __init__(
        out self,
        var model: Self.M,
        initial_parameters: List[Float64],
        *,
        var _weights: Optional[List[Float64]],
        options: Optional[LeastSquaresOptions] = None,
        bounds: Optional[Bounds] = None,
    ) raises:
        _validate_parameters(initial_parameters, "initial_parameters")
        var residual_count = model.residual_count()
        if residual_count < len(initial_parameters):
            raise Error(
                String(
                    "model declares ",
                    residual_count,
                    " residual" if residual_count == 1 else " residuals",
                    " for ",
                    len(initial_parameters),
                    (" parameter" if len(initial_parameters) == 1 else " parameters"),
                    (
                        "; a least-squares problem needs at least as "
                        "many residuals as parameters"
                    ),
                )
            )
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
        self.bounds = bounds.copy()
        self.validate()

    def validate(self) raises:
        """Revalidate all reachable problem state without calling the model."""
        self.options.validate()
        _validate_parameters(self.initial_parameters, "initial_parameters")

        if self.options.x_scale:
            var x_scale_count = len(self.options.x_scale.value())
            if x_scale_count != len(self.initial_parameters):
                raise Error(
                    String(
                        "x_scale has ",
                        x_scale_count,
                        " entries for ",
                        len(self.initial_parameters),
                        " parameters",
                    )
                )

        if self.bounds:
            var bounds = self.bounds.value().copy()
            bounds.validate()
            if bounds.parameter_count() != len(self.initial_parameters):
                raise Error(
                    String(
                        "bounds parameter count ",
                        bounds.parameter_count(),
                        " must equal initial_parameters count ",
                        len(self.initial_parameters),
                    )
                )
            for index in range(len(self.initial_parameters)):
                var parameter = self.initial_parameters[index]
                if parameter <= bounds.lower(index) or parameter >= bounds.upper(index):
                    raise Error(
                        String(
                            "initial parameter[",
                            index,
                            "] ",
                            parameter,
                            " is not strictly inside bounds [",
                            bounds.lower(index),
                            ", ",
                            bounds.upper(index),
                            "]",
                        )
                    )

        if self.residual_count < len(self.initial_parameters):
            raise Error(
                String(
                    "model declares ",
                    self.residual_count,
                    (" residual" if self.residual_count == 1 else " residuals"),
                    " for ",
                    len(self.initial_parameters),
                    (
                        " parameter" if len(self.initial_parameters)
                        == 1 else " parameters"
                    ),
                    (
                        "; a least-squares problem needs at least as "
                        "many residuals as parameters"
                    ),
                )
            )
        if self.model.residual_count() != self.residual_count:
            raise Error(
                String(
                    "model residual_count ",
                    self.model.residual_count(),
                    " must match problem residual_count ",
                    self.residual_count,
                )
            )
        if len(self.weights) != self.residual_count:
            raise Error(
                String(
                    "weights count ",
                    len(self.weights),
                    " must equal residual_count ",
                    self.residual_count,
                )
            )

        var any_positive = False
        for index in range(len(self.weights)):
            var weight = self.weights[index]
            if not isfinite(weight) or weight < 0.0:
                raise Error(
                    String(
                        "weights[",
                        index,
                        "] must be finite and non-negative; got ",
                        weight,
                    )
                )
            if weight > 0.0:
                any_positive = True
        if not any_positive:
            raise Error(
                "weights must include at least one positive entry; got all zeros"
            )

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
        _validate_parameters(parameters, "parameters")
        if len(parameters) != len(self.initial_parameters):
            raise Error(
                String(
                    "parameters count ",
                    len(parameters),
                    " must equal initial_parameters count ",
                    len(self.initial_parameters),
                )
            )

        var entry_residual_count = self.residual_count
        var residuals = self.model.residuals(parameters)
        if self.model.residual_count() != entry_residual_count:
            raise Error(
                String(
                    "model residual_count changed during residual evaluation; got ",
                    self.model.residual_count(),
                    " after entering with ",
                    entry_residual_count,
                )
            )
        if len(residuals) != entry_residual_count:
            raise Error(
                String(
                    "model returned a residual vector of length ",
                    len(residuals),
                    "; expected length ",
                    entry_residual_count,
                )
            )
        for index in range(len(residuals)):
            if not isfinite(residuals[index]):
                raise Error(
                    String(
                        "model residuals[",
                        index,
                        "] must be finite; got ",
                        residuals[index],
                    )
                )
        return residuals^


def _validate_parameters(parameters: List[Float64], name: String) raises:
    if len(parameters) == 0:
        raise Error(
            String(name, " must contain at least one parameter; got ", len(parameters))
        )
    for index in range(len(parameters)):
        if not isfinite(parameters[index]):
            raise Error(
                String(
                    name,
                    "[",
                    index,
                    "] must be finite; got ",
                    parameters[index],
                )
            )
