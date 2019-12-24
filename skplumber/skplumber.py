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


class SKPlumber:

    models_map: t.Dict[ProblemType, t.List[t.Type[Primitive]]] = {
        ProblemType.CLASSIFICATION: list(classifiers.values()),
        ProblemType.REGRESSION: list(regressors.values()),
    }

    def crank(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        problem: str,
        metric: str = None,
        sampler: PipelineSampler = None,
        n_splits: int = 3,
        n: int = 10,
    ) -> t.Tuple[Pipeline, float]:
        """
        The main runtime method of the package. Give a dataset, problem type,
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
        n_splits
            The number of folds to conduct when evaluating the performance via
            cross validation of each sampled pipeine. 
        n
            The number of pipelines to try out on the problem.
        
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

        best_pipeline, best_score = sampler.run(
            X,
            y,
            num_samples=n,
            n_splits=n_splits,
            models=self.models_map[problem_type],
            transformers=list(transformers.values()),
            problem_type=problem_type,
            metric=_metric,
        )

        print(f"found best test score of {best_score}")
        print("pipeline steps of best model:")
        for step in best_pipeline.steps:
            print(step.primitive.__class__.__name__)

        # Now that we have the "best" model, train it on
        # the full dataset so it can see as much of the
        # dataset's distribution as possible in an effort
        # to be more ready for the wild.
        best_pipeline.fit(*shuffle(X, y))

        return best_pipeline, best_score
