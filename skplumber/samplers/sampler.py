from abc import ABC, abstractmethod
import typing as t

import pandas as pd

from skplumber.primitives.primitive import Primitive
from skplumber.pipeline import Pipeline
from skplumber.consts import ProblemType
from skplumber.metrics import Metric
from skplumber.utils import logger, conditional_timeout, EvaluationTimeoutError


class SamplerState(t.NamedTuple):
    score: float
    pipeline: Pipeline
    nit: int


class PipelineSampler(ABC):
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        num_samples: int,
        models: t.List[t.Type[Primitive]],
        transformers: t.List[t.Type[Primitive]],
        problem_type: ProblemType,
        metric: Metric,
        evaluator: t.Callable,
        pipeline_timeout: t.Optional[int],
        callback: t.Optional[t.Callable] = None,
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
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        should_timeout = pipeline_timeout is not None
        best_score = metric.worst_value
        best_pipeline = None

        for i in range(1, num_samples + 1):
            logger.info(f"sampling pipeline {i+1}/{num_samples}")
            pipeline = self.sample_pipeline(problem_type, models, transformers)

            try:

                with conditional_timeout(pipeline_timeout, should_timeout):
                    test_score = evaluator(pipeline, X, y, metric)
                    logger.info(f"achieved test score: {test_score}")

                    if (
                        metric.is_better_than(test_score, best_score)
                        or best_pipeline is None
                    ):
                        best_score = test_score
                        best_pipeline = pipeline

                    if callback is not None:
                        exit_early = callback(SamplerState(test_score, pipeline, i))
                        if exit_early:
                            break

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
