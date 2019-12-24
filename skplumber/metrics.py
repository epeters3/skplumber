from abc import ABC, abstractmethod
import typing as t
import math

import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from skplumber.consts import ProblemType


class Metric(ABC):
    @property
    @abstractmethod
    def problem_type(self) -> ProblemType:
        """
        The problem type this metric can be computed for.
        """
        pass

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
    problem_type = ProblemType.REGRESSION

    def __call__(self, y: pd.Series, predictions: pd.Series) -> float:
        return math.sqrt(mse(y, predictions))

    def is_better_than(self, a: float, b: float) -> bool:
        return a < b


class F1MACRO(Metric):
    problem_type = ProblemType.CLASSIFICATION

    def __call__(self, y: pd.Series, predictions: pd.Series) -> float:
        return f1_score(y, predictions, average="macro")

    def is_better_than(self, a: float, b: float) -> bool:
        return a > b


class Accuracy(Metric):
    problem_type = ProblemType.CLASSIFICATION

    def __call__(self, y: pd.Series, predictions: pd.Series) -> float:
        return accuracy_score(y, predictions)

    def is_better_than(self, a: float, b: float) -> bool:
        return a > b


metrics: t.Dict[str, Metric] = {
    "rmse": RMSE(),
    "f1macro": F1MACRO(),
    "accuracy": Accuracy(),
}

default_metrics: t.Dict[ProblemType, Metric] = {
    ProblemType.REGRESSION: RMSE(),
    ProblemType.CLASSIFICATION: Accuracy(),
}
