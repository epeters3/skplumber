import random
import typing as t

from skplumber.primitives.primitive import Primitive
from skplumber.samplers.sampler import PipelineSampler
from skplumber.pipeline import Pipeline
from skplumber.consts import ProblemType


class StraightPipelineSampler(PipelineSampler):
    def __init__(self, preprocessors: int = 1) -> None:
        self.preprocessors = preprocessors

    def sample_pipeline(
        self,
        problem_type: ProblemType,
        models: t.List[t.Type[Primitive]],
        transformers: t.List[t.Type[Primitive]],
    ) -> Pipeline:
        pipeline = Pipeline()
        for _ in range(self.preprocessors):
            pipeline.add_step(random.choice(transformers))
        pipeline.add_step(random.choice(models))
        return pipeline
