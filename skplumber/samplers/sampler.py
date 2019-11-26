from abc import ABC, abstractmethod
from typing import List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from skplumber.primitives.primitive import Primitive
from skplumber.pipeline import run_pipeline
from skplumber.consts import ProblemType
from skplumber.metrics import problems_to_metrics


class PipelineSampler(ABC):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        models: List[Primitive],
        transformers: List[Primitive],
        problem_type: ProblemType,
        test_size: Union[float, int],
    ) -> None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = models
        self.transfomers = transformers
        self.problem_type = problem_type
        self.best_score = None
        self.best_pipeline = None

    def run(self, num_samples: int):
        """
        Samples `num_samples` pipelines, returning the best one
        found along the way.

        Parameters
        ----------
        num_samples
            The number of pipelines to sample and evaluate.
        
        Returns
        -------
        Pipeline
            The fitted best pipeline trained on the problem.
        float
            The score of the best pipeline that was trained.
        """
        problem_metric = problems_to_metrics[self.problem_type]

        for i in range(num_samples):
            print(f"sampling pipeline {i}/{num_samples}")
            pipeline = self.sample_pipeline()
            train_score = run_pipeline(
                self.X_train,
                self.y_train,
                pipeline,
                fit=True,
                problem_type=self.problem_type,
            )
            test_score = run_pipeline(
                self.X_test,
                self.y_test,
                pipeline,
                fit=False,
                problem_type=self.problem_type,
            )
            print(f"achieved train score: {train_score}, test score: {test_score}")
            if i == 0:
                self.best_pipeline = pipeline
                self.best_score = test_score
            else:
                if problem_metric.is_better_than(test_score, self.best_score):
                    self.best_score = test_score
                    self.best_pipeline = pipeline

        return self.best_pipeline, self.best_score

    @abstractmethod
    def sample_pipeline(self):
        pass
