"""Curve-fitting front door for paired independent and observed data."""

from std.collections import List, Optional
from std.io import Writable, Writer
from std.utils.numerics import isfinite

from .bounds import Bounds
from .options import LeastSquaresOptions
from .problem import LeastSquaresProblem, ResidualModel
from .result import LeastSquaresResult
from .solve import least_squares
from .statistics import FitReport, FitStatistics, fit_statistics


trait CurveModel(Deinitable, Movable):
    """A stateful curve callback evaluated at every supplied coordinate.

    Every successful call must return exactly one value for every coordinate
    in ``t``.
    """

    def values(
        mut self, parameters: List[Float64], t: Span[Float64, ...]
    ) raises -> List[Float64]:
        """Return one modeled value per coordinate in ``t``."""
        ...


struct _CurveAdapter[M: CurveModel](ResidualModel):
    var model: Self.M
    var t: List[Float64]
    var y: List[Float64]

    def __init__(
        out self,
        var model: Self.M,
        t: Span[Float64, ...],
        y: Span[Float64, ...],
        parameter_count: Int,
    ) raises:
        if len(t) != len(y):
            raise Error(
                String("t has ", len(t), " entries but y has ", len(y), " entries")
            )
        if len(t) < 1:
            raise Error("curve fitting requires at least one t/y observation")
        if len(t) <= parameter_count:
            var degrees_of_freedom = len(t) - parameter_count
            if degrees_of_freedom == 0:
                raise Error(
                    String(
                        len(t),
                        " observations for ",
                        parameter_count,
                        (
                            " parameters leaves zero degrees of freedom; add "
                            "observations or fix parameters"
                        ),
                    )
                )
            raise Error(
                String(
                    len(t),
                    " observations for ",
                    parameter_count,
                    " parameters leaves ",
                    degrees_of_freedom,
                    " degrees of freedom; add observations or fix parameters",
                )
            )

        self.model = model^
        self.t = List[Float64](length=len(t), fill=0.0)
        self.y = List[Float64](length=len(y), fill=0.0)
        for index in range(len(t)):
            if not isfinite(t[index]):
                raise Error(String("t[", index, "] must be finite; got ", t[index]))
            if not isfinite(y[index]):
                raise Error(String("y[", index, "] must be finite; got ", y[index]))
            self.t[index] = t[index]
            self.y[index] = y[index]

    def residual_count(self) -> Int:
        return len(self.y)

    def residuals(mut self, parameters: List[Float64]) raises -> List[Float64]:
        var values = self.model.values(parameters, self.t)
        if len(values) != len(self.t):
            raise Error(
                String(
                    "model returned ",
                    len(values),
                    " values for ",
                    len(self.t),
                    " t points",
                )
            )
        for index in range(len(values)):
            values[index] -= self.y[index]
        return values^


struct CurveFit[M: CurveModel](Movable):
    """A validated, reusable nonlinear fit of a curve to paired observations.

    The model and initial parameters are owned. The two constructor overloads
    accept either no ``sigma`` or a keyword-only ``sigma`` span.

    Convention: Input ``Span`` values are validated and copied once into owned
    storage during construction, so caller buffers need not outlive the fit.
    ``CurveModel.values(parameters, t) - y`` defines the raw residuals. When
    ``sigma`` is supplied, Nerai weights are ``1 / sigma`` and multiply each
    residual once. Together with ``fit_statistics``, this matches SciPy
    ``curve_fit(..., absolute_sigma=False)`` covariance semantics.
    """

    var _problem: LeastSquaresProblem[_CurveAdapter[Self.M]]

    def __init__(
        out self,
        var model: Self.M,
        t: Span[Float64, ...],
        y: Span[Float64, ...],
        initial_parameters: List[Float64],
        *,
        bounds: Optional[Bounds] = None,
        options: Optional[LeastSquaresOptions] = None,
    ) raises:
        """Own and validate an unweighted curve-fitting problem."""
        var adapter = _CurveAdapter(model^, t, y, len(initial_parameters))
        self._problem = LeastSquaresProblem(
            adapter^,
            initial_parameters,
            bounds=bounds,
            options=options,
        )

    def __init__(
        out self,
        var model: Self.M,
        t: Span[Float64, ...],
        y: Span[Float64, ...],
        initial_parameters: List[Float64],
        *,
        sigma: Span[Float64, ...],
        bounds: Optional[Bounds] = None,
        options: Optional[LeastSquaresOptions] = None,
    ) raises:
        """Own and validate a curve fit weighted by inverse ``sigma``."""
        if len(sigma) != len(y):
            raise Error(
                String(
                    "sigma has ",
                    len(sigma),
                    " entries but y has ",
                    len(y),
                    " entries",
                )
            )
        var weights = List[Float64](length=len(sigma), fill=0.0)
        for index in range(len(sigma)):
            if not isfinite(sigma[index]) or sigma[index] <= 0.0:
                raise Error(
                    String(
                        "sigma[",
                        index,
                        "] must be finite and positive; got ",
                        sigma[index],
                    )
                )
            weights[index] = 1.0 / sigma[index]

        var adapter = _CurveAdapter(model^, t, y, len(initial_parameters))
        self._problem = LeastSquaresProblem(
            adapter^,
            initial_parameters,
            weights=weights,
            bounds=bounds,
            options=options,
        )

    def solve(mut self) raises -> CurveFitResult:
        """Fit from the stored initial values and return result statistics.

        Solver failure terminations remain inspectable in the returned result
        when statistics succeed. Model and statistics errors propagate.
        """
        var result = least_squares(self._problem)
        var statistics = fit_statistics(self._problem, result)
        return CurveFitResult(result, statistics)


struct CurveFitResult(Copyable, Writable):
    """A public least-squares result and its uncertainty statistics."""

    var result: LeastSquaresResult
    var statistics: FitStatistics

    def __init__(
        out self, result: LeastSquaresResult, statistics: FitStatistics
    ) raises:
        """Copy a result and matching statistics into one curve-fit result."""
        result.validate()
        statistics.validate()
        _ = FitReport(result, statistics)
        self.result = result.copy()
        self.statistics = statistics.copy()

    def validate(self) raises:
        """Revalidate both public fields and their matching dimensions."""
        self.result.validate()
        self.statistics.validate()
        _ = FitReport(self.result, self.statistics)

    def __str__(self) -> String:
        """Return exactly the stable stderr-aware ``FitReport`` block."""
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        """Write exactly the stable stderr-aware ``FitReport`` block."""
        try:
            var report = FitReport(self.result, self.statistics)
            report.write_to(writer)
        except:
            writer.write("invalid curve-fit result\n")
