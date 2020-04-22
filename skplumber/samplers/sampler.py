from abc import ABC, abstractmethod
import typing as t
from typing import NamedTuple
from time import time

import pandas as pd

from skplumber.primitives.primitive import Primitive
from skplumber.pipeline import Pipeline
from skplumber.consts import ProblemType
from skplumber.metrics import Metric
from skplumber.utils import (
    logger,
    conditional_timeout,
    EvaluationTimeoutError,
    PipelineRunError,
)


class SamplerState(NamedTuple):
    score: float
    pipeline: Pipeline
    train_time: float
    n_iters: int


class PipelineSampler(ABC):
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        models: t.List[t.Type[Primitive]],
        transformers: t.List[t.Type[Primitive]],
        problem_type: ProblemType,
        metric: Metric,
        evaluator: t.Callable,
        pipeline_timeout: t.Optional[int],
        num_samples: t.Optional[int] = None,
        callback: t.Union[None, t.Callable, t.List[t.Callable]] = None,
        exit_on_pipeline_error: bool = True,
    ) -> t.Tuple[Pipeline, float, int]:
        """Samples `num_samples` pipelines, returning the best one found along the way.
        
        Returns
        -------
        best_pipeline : Pipeline
            The fitted best pipeline trained on the problem.
        best_score : float
            The score of the best pipeline that was trained.
        n_iters : int
            The total number of iterations the sampler completed.
        """

        # Validate inputs

        if num_samples is None and callback is None:
            raise ValueError(
                "either num_samples or callback must be"
                " passed so the sampler knows when to stop"
            )

        if num_samples is not None and num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        if callback is None:
            callbacks: t.List[t.Callable] = []
        elif callable(callback):
            callbacks = [callback]
        elif isinstance(callback, list):
            callbacks = callback
        else:
            raise ValueError(f"unsupported type '{type(callback)}' for callback arg")

        # Initialize

        should_timeout = pipeline_timeout is not None
        best_score = metric.worst_value
        best_pipeline = None

        # Conduct the sampling

        i = 0
        while True:
            i += 1
            logger.info(
                f"sampling pipeline {i}"
                f"{'/' + str(num_samples) if num_samples else ''}"
            )
            pipeline = self.sample_pipeline(problem_type, models, transformers)

            try:

                with conditional_timeout(pipeline_timeout, should_timeout):

                    # Train the pipeline and check its performance.

                    start_time = time()
                    test_score = evaluator(pipeline, X, y, metric)
                    logger.info(f"achieved test score: {test_score}")

                    if (
                        metric.is_better_than(test_score, best_score)
                        or best_pipeline is None
                    ):
                        best_score = test_score
                        best_pipeline = pipeline

                    # Check to see if its time to stop sampling.

                    if callback is not None:
                        # We stop if any callback returns True.
                        train_time = time() - start_time
                        exit_early = any(
                            cb(SamplerState(test_score, pipeline, train_time, i))
                            for cb in callbacks
                        )
                        if exit_early:
                            break

                    if best_score == metric.best_value:
                        logger.info(
                            f"found best possible score {metric.best_value} early, "
                            "stopping the search"
                        )
                        break

                    if num_samples and i >= num_samples:
                        break

            except EvaluationTimeoutError:

                logger.info("pipeline took too long to evaluate, skipping")
                logger.debug(pipeline)

            except PipelineRunError as e:

                logger.exception(e)
                if exit_on_pipeline_error:
                    raise e

        return best_pipeline, best_score, i

    @abstractmethod
    def sample_pipeline(
        self,
        problem_type: ProblemType,
        models: t.List[t.Type[Primitive]],
        transformers: t.List[t.Type[Primitive]],
    ) -> Pipeline:
        pass
