import random
import typing as t

from skplumber.primitives.primitive import Primitive
from skplumber.samplers.sampler import PipelineSampler
from skplumber.pipeline import Pipeline
from skplumber.consts import ProblemType


class OneStackPipelineSampler(PipelineSampler):
    """
    Each pipeline this strategy samples routes the
    input data to `self.width` randomly sampled primitives
    (both models and transformers), concatentates all their
    output, and feeds it into a final randomly sampled
    model. Thus, it is a pipeline of width `self.width` that
    uses a single layer of stacking.
    """

    def __init__(self, width: int = 3) -> None:
        self.width = width

    def sample_pipeline(
        self,
        problem_type: ProblemType,
        models: t.List[t.Type[Primitive]],
        transformers: t.List[t.Type[Primitive]],
    ) -> Pipeline:
        all_primitives = models + transformers
        pipeline = Pipeline()
        stack_input = pipeline.curr_step_i
        stack_outputs = []
        for _ in range(self.width):
            primitive = random.choice(all_primitives)
            pipeline.add_step(primitive, [stack_input])
            stack_outputs.append(pipeline.curr_step_i)
        pipeline.add_step(random.choice(models), stack_outputs)
        return pipeline
