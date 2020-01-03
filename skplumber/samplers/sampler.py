from abc import ABC, abstractmethod
import typing as t

import pandas as pd
from sklearn.model_selection import KFold

from skplumber.primitives.primitive import Primitive
from skplumber.pipeline import Pipeline
from skplumber.consts import ProblemType
from skplumber.metrics import Metric
from skplumber.utils import logger, conditional_timeout, EvaluationTimeoutError


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
        pipeline_timeout: t.Optional[int] = None,
    ) -> t.Tuple[Pipeline, float]:
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
        should_timeout = pipeline_timeout is not None
        best_score = metric.worst_value
        best_pipeline = None

        for i in range(num_samples):
            logger.info(f"sampling pipeline {i+1}/{num_samples}")
            pipeline = self.sample_pipeline(problem_type, models, transformers)

            try:

                with conditional_timeout(pipeline_timeout, should_timeout):
                    pipeline, test_score = self._evaluate_pipeline(
                        pipeline, X, y, metric, cv
                    )
                    if metric.is_better_than(test_score, best_score):
                        best_score = test_score
                        best_pipeline = pipeline

            except EvaluationTimeoutError:

                logger.info("pipeline took too long to evaluate, skipping")
                logger.debug(pipeline)

            finally:

                if best_score == metric.best_value:
                    logger.info(
                        f"found best possible score {metric.best_value} early, "
                        "stopping the search"
                    )
                    break

        return best_pipeline, best_score

    @abstractmethod
    def sample_pipeline(
        self,
        problem_type: ProblemType,
        models: t.List[t.Type[Primitive]],
        transformers: t.List[t.Type[Primitive]],
    ) -> Pipeline:
        pass

    def _evaluate_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        metric: Metric,
        cv: KFold,
    ) -> t.Tuple[Pipeline, float]:
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

        test_score = sum(scores) / len(scores)
        logger.info(f"achieved test score: {test_score}")
        return pipeline, test_score
