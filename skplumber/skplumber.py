import typing as t

import pandas as pd
from sklearn.utils import shuffle

from skplumber.consts import ProblemType
from skplumber.samplers.sampler import PipelineSampler
from skplumber.samplers.straight import StraightPipelineSampler
from skplumber.pipeline import Pipeline
from skplumber.primitives.primitive import Primitive
from skplumber.primitives.sk_primitives.classifiers import classifiers
from skplumber.primitives.sk_primitives.regressors import regressors
from skplumber.primitives.sk_primitives.transformers import transformers
from skplumber.metrics import default_metrics, metrics
from skplumber.utils import logger
from skplumber.evaluators import make_train_test_evaluator
from skplumber.tuners.ga import ga_tune


class SKPlumber:

    models_map: t.Dict[ProblemType, t.List[t.Type[Primitive]]] = {
        ProblemType.CLASSIFICATION: list(classifiers.values()),
        ProblemType.REGRESSION: list(regressors.values()),
    }

    def crank(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem: str,
        *,
        metric: str = None,
        sampler: PipelineSampler = None,
        n: int = 10,
        evaluator: t.Callable = None,
        pipeline_timeout: t.Optional[int] = None,
        tune: bool = False,
        callback: t.Optional[t.Callable] = None,
    ) -> t.Tuple[Pipeline, float]:
        """
        The main runtime method of the package. Given a dataset, problem type,
        and sampling strategy, it tries to find, in a limited amount of time,
        the best performing pipeline it can.

        Parameters
        ----------
        X
            The features of your dataset.
        y
            The target vector of your dataset. The indices of `X` and `y`
            should match up.
        problem
            The type of problem you want to train this dataset for e.g.
            "classification". See the values of the `skplumber.consts.ProblemType`
            enum for a list of valid options.
        metric
            The type of metric you want to score the pipeline predictions with.
            See the keys of the `skplumber.metrics.metrics` dict for a list of
            valid options. If `None`, a default metric will be used.
        sampler
            An instance of a class inheriting from
            `skplumber.samplers.sampler.PipelineSampler`. Used to decide what
            types of pipelines to sample and try out on the problem. If `None`,
            the `skplumber.samplers.straight.StraightPipelineSampler` will be
            used with default arguments.
        n
            The number of pipelines to try out on the problem.
        evaluator
            The evaluation strategy to use to fit and score a pipeline to determine
            its performance. Must be a function with signature:
            ```
            f(
                pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, metric: Metric
            ) -> float:
            ```,
            meaning it must accept a `skplumber.Pipeline` object, dataset `X` with
            corresponding target vector `y`, and an instantiation of a class
            inheriting from `skplumber.metrics.Metric`. Should fit on the dataset
            and compute and return its performance on that dataset. For basic k-fold
            cross validation or train/test split strategies, see
            `skplumber.evaluators.make_kfold_evaluator` or
            `skplumber.evaluators.make_train_test_evaluator`.
            If no evaluator is provided, a default train/test split evaluation strategy
            will be used. `evaluator` will be used during both the sampling and
            hyperparameter tuning stages, if `tune==True`.
        pipeline_timeout
            The maximum number of seconds to spend evaluating any one pipeline.
            If a sampled pipeline takes longer than this to evaluate, it will
            be skipped.
        tune
            Whether to perform hyperparameter tuning on the best found pipeline.
        callback
            Optional callback function. Will be called after each sampled pipeline
            is evaluated. Should have the function signature
            `callback(state: SamplerState) -> bool`. If `callback` returns `True`, the
            sampling or hyperparameter optimization will end prematurely. `state`
            is a named tuple containing these members:
                - pipeline: The pipeline fitted in the previous iteration.
                - score: The score of `pipeline`.
                - nit: The number of iterations completed so far.
        
        Returns
        -------
        Pipeline
            The best pipeline the search strategy was able to find.
        float
            The score of the best found pipeline.
        """
        problem_type = ProblemType(problem)

        valid_metric_names = [
            name
            for name, _metric in metrics.items()
            if _metric.problem_type == problem_type
        ]
        if metric is not None and metric not in valid_metric_names:
            raise ValueError(f"metric is invalid, must be one of {valid_metric_names}")
        if metric is None:
            _metric = default_metrics[problem_type]
        else:
            _metric = metrics[metric]

        if len(X.index) != y.size:
            raise ValueError(f"X and y must have the same number of instances")

        if sampler is None:
            sampler = StraightPipelineSampler()

        if evaluator is None:
            evaluator = make_train_test_evaluator()

        best_pipeline, best_score = sampler.run(
            X,
            y,
            num_samples=n,
            models=self.models_map[problem_type],
            transformers=list(transformers.values()),
            problem_type=problem_type,
            metric=_metric,
            evaluator=evaluator,
            pipeline_timeout=pipeline_timeout,
            callback=callback,
        )

        logger.info(f"found best validation score of {best_score}")
        logger.info("best pipeline:")
        logger.info(best_pipeline)

        if tune:
            logger.info("now tuning best found pipeline...")
            ga_tune(best_pipeline, X, y, evaluator, _metric, print_every=1)

        # Now that we have the "best" model, train it on
        # the full dataset so it can see as much of the
        # dataset's distribution as possible in an effort
        # to be more ready for the wild.
        best_pipeline.fit(*shuffle(X, y))

        return best_pipeline, best_score
