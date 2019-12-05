import random

from skplumber.samplers.sampler import PipelineSampler
from skplumber.pipeline import Pipeline


class StraightPipelineSampler(PipelineSampler):
    def sample_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_step(random.choice(self.transfomers))
        pipeline.add_step(random.choice(self.transfomers))
        pipeline.add_step(random.choice(self.models))
        return pipeline
