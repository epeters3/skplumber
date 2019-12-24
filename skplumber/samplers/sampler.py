from abc import ABC, abstractmethod
import typing as t

import pandas as pd
from sklearn.model_selection import KFold

from skplumber.primitives.primitive import Primitive
from skplumber.consts import ProblemType
from skplumber.metrics import Metric


class PipelineSampler(ABC):
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        num_samples: int,
        n_splits: int,
        models: t.List[t.Type[Primitive]],
        transformers: t.List[t.Type[Primitive]],
        problem_type: ProblemType,
        metric: Metric,
    ):
        """
        Samples `num_samples` pipelines, returning the best one
        found along the way.
        
        Returns
        -------
        Pipeline
            The fitted best pipeline trained on the problem.
        float
            The score of the best pipeline that was trained.
        """
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        for i in range(num_samples):
            print(f"sampling pipeline {i+1}/{num_samples}")
            pipeline = self.sample_pipeline(problem_type, models, transformers)

            # Perform cross validation of `n_splits` folds, calculating
            # the average performance over the folds as this pipeline's
            # performance.
            scores = []
            for train_index, test_index in cv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                pipeline.fit(X_train, y_train)
                test_predictions = pipeline.predict(X_test)
                fold_score = metric(y_test, test_predictions)
                scores.append(fold_score)

            test_score = sum(scores) / len(scores)

            print(f"achieved test score: {test_score}")
            if i == 0:
                best_pipeline = pipeline
                best_score = test_score
            else:
                if metric.is_better_than(test_score, best_score):
                    best_score = test_score
                    best_pipeline = pipeline

        return best_pipeline, best_score

    @abstractmethod
    def sample_pipeline(
        self,
        problem_type: ProblemType,
        models: t.List[t.Type[Primitive]],
        transformers: t.List[t.Type[Primitive]],
    ):
        pass
