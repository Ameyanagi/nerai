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
        if len(t) < parameter_count:
            raise Error(
                String(
                    "t/y observation count ",
                    len(t),
                    " is less than parameter count ",
                    parameter_count,
                    (
                        "; curve fitting requires at least as many observations "
                        "as parameters — add observations or fix parameters"
                    ),
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
    residual once. By default this matches SciPy
    ``curve_fit(..., absolute_sigma=False)`` covariance semantics; the sigma
    overload's ``absolute_sigma=True`` skips reduced-chi-squared rescaling for
    known measurement uncertainties.
    """

    var _problem: LeastSquaresProblem[_CurveAdapter[Self.M]]
    var _absolute_sigma: Bool

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
        self._absolute_sigma = False

    def __init__(
        out self,
        var model: Self.M,
        t: Span[Float64, ...],
        y: Span[Float64, ...],
        initial_parameters: List[Float64],
        *,
        sigma: Span[Float64, ...],
        absolute_sigma: Bool = False,
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
        self._absolute_sigma = absolute_sigma

    def solve(mut self) raises -> CurveFitResult:
        """Fit from the stored initial values and return result statistics.

        Solver errors propagate. If post-fit statistics cannot be estimated,
        the returned solver result remains available and reports the reason.
        """
        var result = least_squares(self._problem)
        var statistics: FitStatistics
        try:
            statistics = fit_statistics(
                self._problem,
                result,
                absolute_sigma=self._absolute_sigma,
            )
        except error:
            return CurveFitResult(
                result,
                Optional[FitStatistics](),
                statistics_message=String(error),
            )
        return CurveFitResult(result, Optional(statistics^))


struct CurveFitResult(Copyable, Writable):
    """A public least-squares result and optional uncertainty statistics."""

    var result: LeastSquaresResult
    var statistics: Optional[FitStatistics]
    var statistics_message: String

    def __init__(
        out self,
        result: LeastSquaresResult,
        statistics: Optional[FitStatistics],
        *,
        statistics_message: String = "",
    ) raises:
        """Copy a result and either matching statistics or their failure reason."""
        result.validate()
        if statistics:
            var present_statistics = statistics.value().copy()
            present_statistics.validate()
            _ = FitReport(result, present_statistics)
            if statistics_message.byte_length() != 0:
                raise Error(
                    String(
                        "statistics_message must be empty when statistics are ",
                        "present; got ",
                        statistics_message,
                    )
                )
        elif statistics_message.byte_length() == 0:
            raise Error(
                "statistics_message must describe why statistics are absent; "
                "got an empty string — pass the fit_statistics error message"
            )
        self.result = result.copy()
        self.statistics = statistics.copy()
        self.statistics_message = String(statistics_message)

    def validate(self) raises:
        """Revalidate both public fields and their matching state."""
        self.result.validate()
        if self.statistics:
            self.statistics.value().validate()
            _ = FitReport(self.result, self.statistics.value())
            if self.statistics_message.byte_length() != 0:
                raise Error(
                    String(
                        "statistics_message must be empty when statistics are ",
                        "present; got ",
                        self.statistics_message,
                    )
                )
        elif self.statistics_message.byte_length() == 0:
            raise Error(
                "statistics_message must describe why statistics are absent; "
                "got an empty string — pass the fit_statistics error message"
            )

    def residuals(self) -> List[Float64]:
        """Return a copy of the raw residual vector at the fitted parameters."""
        return self.result.residuals.copy()

    def __str__(self) -> String:
        """Return exactly the stable stderr-aware ``FitReport`` block."""
        var result = String()
        self.write_to(result)
        return result^

    def write_to[W: Writer](self, mut writer: W):
        """Write exactly the stable stderr-aware ``FitReport`` block."""
        if self.statistics:
            try:
                var report = FitReport(self.result, self.statistics.value())
                report.write_to(writer)
            except:
                writer.write("invalid curve-fit result\n")
            return

        writer.write("termination           ", self.result.termination, "\n")
        writer.write(
            "converged             ",
            "yes" if self.result.converged() else "no",
            "\n",
        )
        writer.write("cost                  ", self.result.cost, "\n")
        writer.write("optimality            ", self.result.optimality, "\n")
        writer.write("iterations            ", self.result.iterations, "\n")
        writer.write("residual evaluations  ", self.result.residual_evaluations, "\n")
        writer.write("jacobian evaluations  ", self.result.jacobian_evaluations, "\n")
        for index in range(len(self.result.parameters)):
            var label = String("parameters[", index, "]")
            writer.write(label)
            for _ in range(label.byte_length(), 22):
                writer.write(" ")
            writer.write(self.result.parameters[index], "\n")
        writer.write(
            "standard errors       not estimated: ",
            self.statistics_message,
            "\n",
        )
