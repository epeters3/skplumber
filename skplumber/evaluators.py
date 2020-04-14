import typing as t
from statistics import mean

from sklearn.model_selection import KFold, train_test_split
import pandas as pd

from skplumber.pipeline import Pipeline
from skplumber.metrics import Metric


def make_kfold_evaluator(
    k: int = 5, shuffle: bool = True, random_state: int = 0
) -> t.Callable:
    """
    Returns a pipeline evaluator that performs
    `k` fold cross-validation.
    """

    def kfold_evaluate(
        pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, metric: Metric
    ) -> float:
        cv = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        # Perform cross validation, calculating the average performance
        # over the folds as this pipeline's performance.
        scores = []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pipeline.fit(X_train, y_train)
            test_predictions = pipeline.predict(X_test)
            fold_score = metric(y_test, test_predictions)
            scores.append(fold_score)
        test_score = mean(scores)
        return test_score

    return kfold_evaluate


def make_train_test_evaluator(
    test_size: float = 0.33, shuffle: bool = True, random_state: int = 0
) -> t.Callable:
    """
    Returns a pipeline evaluator that trains the pipeline on a training
    set having `1 - test_size` percent of the instances, and which computes
    the pipeline's score over `test_size` percent of the instances.
    """

    def train_test_evaluate(
        pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, metric: Metric
    ) -> float:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
        pipeline.fit(X_train, y_train)
        test_predictions = pipeline.predict(X_test)
        score = metric(y_test, test_predictions)
        return score

    return train_test_evaluate


def make_down_sample_evaluator(
    sample_ratio: float,
    test_size: float = 0.33,
    shuffle: bool = True,
    random_state: int = 0,
) -> t.Callable:
    """
    Returns a pipeline evaluator that conducts a basic
    train/test split fit/evaluation like `train_test_evaluate`
    does, only it only uses a randomly selected subset of the
    data. Useful for quick, low-fidelity estimates of a
    pipeline's performance.

    Parameters
    ----------
    sample_ratio : float
        The ratio of the data to conduct the train/test
        evaluation on.
    """

    def down_sample_evaluate(
        pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, metric: Metric
    ) -> float:
        # First down-sample the data by using `train_test_split` in a
        # perhaps unintended way.
        X_smaller, _, y_smaller, _ = train_test_split(
            X, y, train_size=sample_ratio, shuffle=shuffle, random_state=random_state
        )
        # Now make the train/test split.
        X_train, X_test, y_train, y_test = train_test_split(
            X_smaller,
            y_smaller,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        # Finally fit and evaluate the models' performance.
        pipeline.fit(X_train, y_train)
        test_predictions = pipeline.predict(X_test)
        score = metric(y_test, test_predictions)
        return score

    return down_sample_evaluate
