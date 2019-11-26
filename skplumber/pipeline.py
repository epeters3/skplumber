from typing import List, Callable, Dict, Type

import pandas as pd

from skplumber.primitives.primitive import Primitive
from skplumber.consts import ProblemType
from skplumber.metrics import score_output
from skplumber.primitives.sk_primitives.preprocessing import MeanValueImputer


class PrimitiveStep:
    def __init__(self, primitive_cls: Type[Primitive], inputs: List[int]):
        """
        Parameters
        ----------
        primitive_cls
            The primitive class this step is associated with.
        inputs
            The indices of the pipeline steps for whose output
            this step will use as its input.
        """
        self.primitive = primitive_cls()
        self.inputs = inputs


class Pipeline:
    def __init__(self) -> None:
        """
        Initializes the pipeline, including some preliminary
        common data preprocessing.
        """
        self.steps: List[PrimitiveStep] = []
        self.add_step(MeanValueImputer)

    def add_step(
        self, primitive_cls: Type[Primitive], inputs: List[int] = None
    ) -> None:
        """
        Adds `primitive` as the next step to this pipeline. If `inputs` is `None`,
        the outputs of the most recent step will be used as `inputs`.
        """
        if inputs is None:
            inputs = [len(self.steps) - 1]
        step = PrimitiveStep(primitive_cls, inputs)
        self.steps.append(step)


def run_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    *,
    fit: bool,
    problem_type: ProblemType,
):
    """
    Runs a pipeline of primitive steps, finally scoring its output.
    """
    all_step_outputs: List[pd.DataFrame] = []

    for step_i, step in enumerate(pipeline.steps):
        print(f"step {step_i}, {step.primitive.__class__.__name__}")
        if step_i == 0:
            step_inputs = X
        else:
            step_inputs = pd.concat([all_step_outputs[i] for i in step.inputs], axis=1)
        if fit:
            step.primitive.fit(step_inputs, y)
        step_outputs = step.primitive.produce(step_inputs)
        if isinstance(step_outputs, pd.Series) and step_i < len(pipeline.steps) - 1:
            # Every step's output but the last step must be a dataframe, since it
            # might be used as the `X` input for a future step.
            step_outputs = pd.DataFrame({"output": step_outputs})
        all_step_outputs.append(step_outputs)

    final_predictions = all_step_outputs[-1]
    if not isinstance(final_predictions, pd.Series):
        raise ValueError(
            f"final pipeline step {pipeline.steps[-1].primitive} "
            "did not output a pandas Series"
        )

    return score_output(y, final_predictions, problem_type)
