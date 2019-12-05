from abc import ABC, abstractmethod
from typing import List, Union, Type

import pandas as pd
from sklearn.model_selection import train_test_split

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
        test_size: Union[int, float],
        models: List[Type[Primitive]],
        transformers: List[Type[Primitive]],
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        best_pipeline = None
        best_score = None

        for i in range(num_samples):
            print(f"sampling pipeline {i+1}/{num_samples}")
            pipeline = self.sample_pipeline(models, transformers)
            pipeline.fit(X_train, y_train)
            test_predictions = pipeline.predict(X_test)
            test_score = metric(y_test, test_predictions)
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
        self, models: List[Type[Primitive]], transformers: List[Type[Primitive]],
    ):
        pass
