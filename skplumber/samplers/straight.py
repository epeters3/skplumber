import random
from typing import List, Type

from skplumber.primitives.primitive import Primitive
from skplumber.samplers.sampler import PipelineSampler
from skplumber.pipeline import Pipeline


class StraightPipelineSampler(PipelineSampler):
    def __init__(self, preprocessors: int = 1) -> None:
        self.preprocessors = preprocessors

    def sample_pipeline(
        self, models: List[Type[Primitive]], transformers: List[Type[Primitive]],
    ) -> Pipeline:
        pipeline = Pipeline()
        for _ in range(self.preprocessors):
            pipeline.add_step(random.choice(transformers))
        pipeline.add_step(random.choice(models))
        return pipeline
