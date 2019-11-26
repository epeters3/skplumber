from typing import Dict, List, Union

import pandas as pd

from skplumber.consts import ProblemType
from skplumber.samplers.sampler import PipelineSampler
from skplumber.samplers.straight import StraightPipelineSampler
from skplumber.primitives.primitive import Primitive
from skplumber.primitives.sk_primitives.classifiers import classifier_primitives
from skplumber.primitives.sk_primitives.regressors import regressor_primitives
from skplumber.primitives.sk_primitives.transformers import transformer_primitives


class SKPlumber:

    sampler_map: Dict[str, PipelineSampler] = {"straight": StraightPipelineSampler}
    models_map: Dict[ProblemType, List[Primitive]] = {
        ProblemType.CLASSIFICATION: classifier_primitives,
        ProblemType.REGRESSION: regressor_primitives,
    }

    def crank(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        problem: str,
        sampler_name: str = "straight",
        test_size: Union[float, int] = 0.25,
        n: int = 10,
    ):
        """
        The main runtime method of the package. Give a dataset, problem type,
        and sampling strategy, it tries to find pipelines that give good
        performance.

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
        sampler_name
            The name of the pipeline sampling strategy you wish to use on
            the problem. See the keys of the `SKPlumber.sampler_map` dict
            for a list of valid options.
        test_size
            Used to determine the size of the test set. Passed as the
            `test_size` argument to `sklearn.model_selection.train_test_split`.
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
        if sampler_name not in self.sampler_map:
            raise ValueError(
                f"invalid sampler {sampler_name} -- "
                f"must be one of {self.sampler_map.keys()}"
            )
        if len(X.index) != y.size:
            raise ValueError(f"X and y must have the same number of instances")

        sampler_cls = self.sampler_map[sampler_name]
        sampler = sampler_cls(
            X,
            y,
            models=self.models_map[problem_type],
            transformers=transformer_primitives,
            problem_type=problem_type,
            test_size=test_size,
        )
        best_pipeline, best_score = sampler.run(n)

        print(f"found best test score of {best_score}")
        print("pipeline steps of best model:")
        for step in best_pipeline.steps:
            print(step.primitive.__class__.__name__)

        return best_pipeline, best_score
