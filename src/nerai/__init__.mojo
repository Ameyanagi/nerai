"""Optimization and nonlinear least-squares foundations for Mojo."""

from .bounds import Bounds
from .curve import CurveFit, CurveFitResult, CurveModel
from .loss import LossEvaluation, LossKind, evaluate_loss, robust_cost
from .options import JacobianScheme, LeastSquaresOptions
from .problem import LeastSquaresProblem, ResidualModel
from .result import LeastSquaresResult
from .solve import least_squares
from .statistics import FitReport, FitStatistics, fit_statistics
from .termination import TerminationReason
