import pandas as pd

from skplumber.consts import ProblemType
from skplumber.samplers.straight import StraightPipelineSampler

sampler_map = {
    "straight": StraightPipelineSampler
}

class SKPlumber:

    def crank(self, X: pd.DataFrame, y: pd.Series, *, problem: str, sampler: str = "straight"):
        """
        The main runtime method of the package. Accepts features, a target,
        and a problem type and tries to find pipelines that give good
        performance.
        """
        if problem not in ProblemType:
            raise ValueError(
                f"invalid problem type {problem} -- must be one of {ProblemType}"
            )
        if sampler not in sampler_map:
            raise ValueError(
                f"invalid sampler {sampler} -- must be one of {sampler_map.keys()}"
            )
        if len(X.index) != y.size:
            raise ValueError(f"X and y must have the same number of instances")