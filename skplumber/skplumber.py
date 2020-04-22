import typing as t
from typing import NamedTuple
from time import time

import pandas as pd
from sklearn.utils import shuffle

from skplumber.consts import ProblemType
from skplumber.samplers.sampler import PipelineSampler, SamplerState
from skplumber.samplers.straight import StraightPipelineSampler
from skplumber.primitives.primitive import Primitive
from skplumber.primitives.sk_primitives.classifiers import classifiers
from skplumber.primitives.sk_primitives.regressors import regressors
from skplumber.primitives.sk_primitives.transformers import transformers
from skplumber.metrics import default_metrics, metrics, Metric
from skplumber.utils import logger
from skplumber.evaluators import make_train_test_evaluator
from skplumber.tuners.ga import ga_tune
from skplumber.progress import EVProgress


class SKPlumberFitState:
    def __init__(self, budget: int, metric: Metric) -> None:
        self.starttime = time()
        self.endbytime = self.starttime + budget
        self.best_pipeline_min_tune_time = 0.0
        self.best_score = metric.worst_value


class SearchResult(NamedTuple):
    # total train time in seconds
    time: float
    # total number of pipelines the sampler tried
    n_sample_iters: int
    # total number of pipelines the hyperparameter tuner tried
    n_tune_iters: int
    # the best score SKPlumber was able to find
    best_score: float


class SKPlumber:

    _models_map: t.Dict[ProblemType, t.List[t.Type[Primitive]]] = {
        ProblemType.CLASSIFICATION: list(classifiers.values()),
        ProblemType.REGRESSION: list(regressors.values()),
    }

    def __init__(
        self,
        problem: str,
        budget: int,
        *,
        metric: str = None,
        sampler: PipelineSampler = None,
        evaluator: t.Callable = None,
        pipeline_timeout: t.Optional[int] = None,
        exit_on_pipeline_error: bool = True,
        callback: t.Optional[t.Callable] = None,
        # TODO: make True by default. Requires being able to control the
        # amount of time it runs for (for tests and the budget)
        block_size: int = 10,
        tune: bool = False,
        tuning_mult_factor: int = 10,
        log_level: str = "INFO",
    ) -> None:
        """
        Parameters
        ----------
        problem : str
            The type of problem you want to train this dataset for e.g.
            "classification". See the values of the `skplumber.consts.ProblemType`
            enum for a list of valid options.
        budget : int
            How much time in seconds `fit` is allowed to search for a good solution.
        metric : Metric, optional
            The type of metric you want to score the pipeline predictions with.
            See the keys of the `skplumber.metrics.metrics` dict for a list of
            valid options. If `None`, a default metric will be used.
        sampler : PipelineSampler, optional
            An instance of a class inheriting from
            `skplumber.samplers.sampler.PipelineSampler`. Used to decide what
            types of pipelines to sample and try out on the problem. If `None`,
            the `skplumber.samplers.straight.StraightPipelineSampler` will be
            used with default arguments.
        evaluator : function, optional
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
        pipeline_timeout : int, optional
            If supplied, the maximum number of seconds to spend evaluating any
            one pipeline. If a sampled pipeline takes longer than this to evaluate,
            it will be skipped.
        exit_on_pipeline_error : bool, optional
            Whether to exit if a specific pipeline errors out while training. If
            `False`, the pipeline will just be skipped and `SKPlumber` will continue.
        callback : function, optional
            Optional callback function. Will be called after each sampled pipeline
            is evaluated. Should have the function signature
            `callback(state: SamplerState) -> bool`. If `callback` returns `True`, the
            sampling or hyperparameter optimization will end prematurely. `state`
            is a named tuple containing these members:
                - pipeline: The pipeline fitted in the previous iteration.
                - score: The score of `pipeline`.
                - train_time: The number of seconds it took to train and
                  evaluate `pipeline`.
                - n_iters: The number of iterations completed so far.
        block_size : int, optional
            The block size to take extrema from when using the block maxima approach
            to calculate return times in the Generalized Extreme Value (GEV)
            distribution fit to the pipeline sample results as the sampler progresses.
        tune : bool, optional
            Whether to perform hyperparameter optimization on the best found pipeline.
        tuning_mult_factor : int, optional
            Each hyperparameter tuning generation will have a population size equal to
            the number of hyerparameters on the pipeline being optimized times this
            value.
        log_level : {'INFO', 'DEBUG', 'NOTEST', 'WARNING', 'ERROR', 'CRITICAL'}, optional
            Log level SKPlumber should use while running. Defaults to `"INFO"`.
        """
        logger.setLevel(log_level)

        # Validate inputs

        self.problem_type = ProblemType(problem)

        valid_metric_names = [
            name
            for name, met in metrics.items()
            if met.problem_type == self.problem_type
        ]

        if metric is not None and metric not in valid_metric_names:
            raise ValueError(f"metric is invalid, must be one of {valid_metric_names}")

        # Set defaults

        if metric is None:
            self.metric = default_metrics[self.problem_type]
        else:
            self.metric = metrics[metric]

        if sampler is None:
            self.sampler: PipelineSampler = StraightPipelineSampler()
        else:
            self.sampler = sampler

        if evaluator is None:
            self.evaluator = make_train_test_evaluator()
        else:
            self.evaluator = evaluator

        # Set other members

        self.budget = budget
        self.pipeline_timeout = pipeline_timeout
        self.tune = tune
        self.progress = EVProgress(block_size, self.metric.opt_dir)
        self.is_fitted = False
        self.tuning_mult_factor = tuning_mult_factor
        self.exit_on_pipeline_error = exit_on_pipeline_error

        self.sampler_cbs = [self._sampler_cb]
        if callback:
            self.sampler_cbs.append(callback)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SearchResult:
        """
        The main runtime method of the package. Given a dataset, problem type,
        and sampling strategy, it tries to find, in a limited amount of time,
        the best performing pipeline it can.

        Parameters
        ----------
        X : pandas.DataFrame
            The features of your dataset.
        y : pandas.Series
            The target vector of your dataset. The indices of `X` and `y`
            should match up.

        Returns
        -------
        result : SearchResult
            A named tuple containing data about how the fit process went.
        """

        # Initialize

        if len(X.index) != y.size:
            raise ValueError(f"X and y must have the same number of instances")

        # A little encapsulation to make this `fit` method's code less huge.
        self.state = SKPlumberFitState(self.budget, self.metric)

        # Run

        self.progress.start()
        best_pipeline, best_score, n_sample_iters = self.sampler.run(
            X,
            y,
            models=self._models_map[self.problem_type],
            transformers=list(transformers.values()),
            problem_type=self.problem_type,
            metric=self.metric,
            evaluator=self.evaluator,
            pipeline_timeout=self.pipeline_timeout,
            callback=self.sampler_cbs,
            exit_on_pipeline_error=self.exit_on_pipeline_error,
        )
        self.best_pipeline = best_pipeline
        self.state.best_score = best_score

        logger.info(f"found best validation score of {best_score}")
        logger.info("best pipeline:")
        logger.info(self.best_pipeline)

        if self.tune:
            logger.info(
                "now performing hyperparameter tuning on best found pipeline..."
            )
            best_tuning_score, best_tuning_params, n_tune_iters = ga_tune(
                self.best_pipeline,
                X,
                y,
                self.evaluator,
                self.metric,
                self.exit_on_pipeline_error,
                population_size=(
                    self.best_pipeline.num_params * self.tuning_mult_factor
                ),
                callback=self._tuner_callback,
            )
            if self.metric.is_better_than(best_tuning_score, self.state.best_score):
                # The hyperparameter tuning was able to find an
                # improvement.
                self.state.best_score = best_tuning_score
                self.best_pipeline.set_params(best_tuning_params)
        else:
            n_tune_iters = 0

        # Now that we have the "best" model, train it on
        # the full dataset so it can see as much of the
        # dataset's distribution as possible in an effort
        # to be more ready for the wild.
        self.best_pipeline.fit(*shuffle(X, y))

        logger.info(
            "finished. total execution time: "
            f"{time() - self.state.starttime:.2f} seconds."
        )
        logger.info(f"final best score found: {self.state.best_score}")

        result = SearchResult(
            time() - self.state.starttime,
            n_sample_iters,
            n_tune_iters,
            self.state.best_score,
        )

        # Fitting completed successfully
        self.is_fitted = True
        return result

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes a prediction for each instance in `X`, returning the predictions.
        """
        if not self.is_fitted:
            raise ValueError(
                "`SKPlumber.fit` must be called and finish "
                "before `predict` can be called."
            )
        return self.best_pipeline.predict(X)

    def _sampler_cb(self, sampler_state: SamplerState) -> bool:
        # Decide how much time is left available to us in
        # the sampling phase.
        if self.tune:
            # We want to leave enough time in the budget to be able
            # to complete at least one generation of hyperparameter tuning.
            if self.metric.is_better_than(sampler_state.score, self.state.best_score):
                self.state.best_score = sampler_state.score
                # An estimate of how long it will take to complete one
                # generation of hyperparameter tuning on this current
                # best pipeline.
                self.state.best_pipeline_min_tune_time = (
                    sampler_state.train_time
                    * sampler_state.pipeline.num_params
                    * self.tuning_mult_factor
                )
            sampling_endtime = (
                self.state.endbytime - self.state.best_pipeline_min_tune_time
            )
        else:
            sampling_endtime = self.state.endbytime

        now = time()
        logger.info(f"{sampling_endtime - now:.2f} seconds left in sampling budget")

        # Logic for tracking sampler progress and exiting when the cost
        # of finding a new best score is too great.
        self.progress.observe(sampler_state.score)

        exit_early = False
        if now > sampling_endtime:

            exit_early = True

        elif self.progress.can_report:

            logger.info(
                f"estimated time to new best: {self.progress.return_time:.2f} seconds"
            )
            if now + self.progress.return_time > sampling_endtime:  # type: ignore
                exit_early = True

        if exit_early:
            logger.info(
                "not enough time is left in the budget to find a new "
                "best score, so no more sampling will be done"
            )

        return exit_early

    def _tuner_callback(self, tuner_state: dict) -> bool:
        now = time()
        logger.info(
            f"candidate pipeline in generation {tuner_state['nit']} finished. "
            f"{self.state.endbytime - now:.2f} seconds left in budget."
        )
        logger.info(f"best score found so far: {tuner_state['fun']}")
        logger.info(
            f"best hyperparameter config found so far: {tuner_state['kwargs_opt']}"
        )
        # We need to quit early if our time budget is used up.
        return True if time() > self.state.endbytime else False
