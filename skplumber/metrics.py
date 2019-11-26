from abc import ABC, abstractmethod
from typing import Callable, Dict
import math

import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score

from skplumber.consts import ProblemType


class Metric(ABC):
    @abstractmethod
    def __call__(self, y: pd.Series, predictions: pd.Series) -> float:
        """
        The method that actually computes the score between the
        truth `y` and the `predictions`.
        """
        pass

    @abstractmethod
    def is_better_than(self, a: float, b: float) -> bool:
        """
        Should return `True` if `a` is better than `b` in
        regards to this metric. E.g. if this metric were
        RMSE, `a=25`, and `b=30`, then this method would
        return `True`.
        """
        pass


class RMSE(Metric):
    def __call__(self, y: pd.Series, predictions: pd.Series) -> float:
        return math.sqrt(mse(y, predictions))

    def is_better_than(self, a: float, b: float) -> bool:
        return a < b


class F1MACRO(Metric):
    def __call__(self, y: pd.Series, predictions: pd.Series) -> float:
        return f1_score(y, predictions, average="macro")

    def is_better_than(self, a: float, b: float) -> bool:
        return a > b


problems_to_metrics: Dict[ProblemType, Metric] = {
    ProblemType.REGRESSION: RMSE(),
    ProblemType.CLASSIFICATION: F1MACRO(),
}


def score_output(
    y: pd.Series, predictions: pd.Series, problem_type: ProblemType
) -> float:
    """
    Scores the output of a pipeline.

    Parameters
    ----------
    y
        The vector of true target values.
    predictions
        A vector of predictions.
    problem_type
        The problem type the predictions are representing. Determines
        which performance metric to use.
    """

    return problems_to_metrics[problem_type](y, predictions)
