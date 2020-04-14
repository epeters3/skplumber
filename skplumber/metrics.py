import typing as t
import math

import pandas as pd
from sklearn.metrics import mean_squared_error as mse, mean_squared_log_error as msle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from skplumber.consts import ProblemType, OptimizationDirection


class Metric:
    def __init__(
        self,
        name: str,
        problem_type: ProblemType,
        compute: t.Callable[[pd.Series, pd.Series], float],
        opt_dir: OptimizationDirection,
        is_better_than: t.Callable[[float, float], bool],
        best_value: float,
        worst_value: float,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            The name of this metric e.g. "RMSE", or "Accuracy".
        problem_type : ProblemType
            The problem type this metric is used to asses e.g.
            Regression for RMSE or Classification for Accuracy.
        compute : function
            The method that actually computes the score between the
            truth `y` and the `predictions`.
        opt_dir : OptimizationDirection
            The direction an optimizer would go to improve this metric e.g.
            for RMSE the goal is to minimize. For accuracy, the goal is to
            maximize.
        is_better_than : function
            Should return `True` if the first arg (a) is better than the
            second arg (b) in regards to this metric. E.g. if this metric were
            RMSE, `a=25`, and second `b=30`, then this method would
            return `True`.
        best_value : float
            The value a perfect model would get under this metric e.g. for
            accuracy it would be 1.0.
        worst_value : float
            The worst value a model could achieve for this metric e.g. for
            accuracy it would be 0.0.
        """
        self.name = name
        self.problem_type = problem_type
        self._compute = compute
        self.opt_dir = opt_dir
        self.is_better_than = is_better_than
        self.best_value = best_value
        self.worst_value = worst_value

    def __call__(self, y: pd.Series, predictions: pd.Series) -> float:
        return self._compute(y, predictions)


rmse = Metric(
    "Root Mean Squared Error",
    ProblemType.REGRESSION,
    lambda y, preds: math.sqrt(mse(y, preds)),
    OptimizationDirection.MINIMIZE,
    lambda a, b: a < b,
    0.0,
    float("inf"),
)

rmsle = Metric(
    "Root Mean Squared Log Error",
    ProblemType.REGRESSION,
    lambda y, preds: math.sqrt(msle(y, preds)),
    OptimizationDirection.MINIMIZE,
    lambda a, b: a < b,
    0.0,
    float("inf"),
)

f1macro = Metric(
    "F1 Macro",
    ProblemType.CLASSIFICATION,
    lambda y, preds: f1_score(y, preds, average="macro"),
    OptimizationDirection.MAXIMIZE,
    lambda a, b: a > b,
    1.0,
    0.0,
)

accuracy = Metric(
    "Accuracy",
    ProblemType.CLASSIFICATION,
    accuracy_score,
    OptimizationDirection.MAXIMIZE,
    lambda a, b: a > b,
    1.0,
    0.0,
)


metrics: t.Dict[str, Metric] = {
    "rmse": rmse,
    "rmsle": rmsle,
    "f1macro": f1macro,
    "accuracy": accuracy,
}

default_metrics: t.Dict[ProblemType, Metric] = {
    ProblemType.REGRESSION: rmse,
    ProblemType.CLASSIFICATION: accuracy,
}
