import typing as t

import pandas as pd

from skplumber.primitives.primitive import Primitive
from skplumber.primitives.custom_primitives.preprocessing import (
    OneHotEncoder,
    RandomImputer,
)


class PrimitiveStep:
    def __init__(self, primitive_cls: t.Type[Primitive], inputs: t.List[int]):
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
        self.steps: t.List[PrimitiveStep] = []
        self.add_step(RandomImputer)
        self.add_step(OneHotEncoder)

    @property
    def curr_step_i(self) -> int:
        return len(self.steps) - 1

    def add_step(
        self, primitive_cls: t.Type[Primitive], inputs: t.List[int] = None
    ) -> None:
        """
        Adds `primitive` as the next step to this pipeline. If `inputs` is `None`,
        the outputs of the most recent step will be used as `inputs`.
        """
        if inputs is None:
            inputs = [len(self.steps) - 1]
        step = PrimitiveStep(primitive_cls, inputs)
        self.steps.append(step)

    def _run(
        self, X: pd.DataFrame, y: t.Optional[pd.Series], *, fit: bool,
    ) -> pd.DataFrame:
        all_step_outputs: t.List[pd.DataFrame] = []

        for step_i, step in enumerate(self.steps):
            if step_i == 0:
                step_inputs = X
            else:
                step_inputs = pd.concat(
                    [all_step_outputs[i] for i in step.inputs], axis=1
                )
            if fit:
                if y is None:
                    raise ValueError("`y` cannot be `None` when fitting a pipeline")
                step.primitive.fit(step_inputs, y)
            step_outputs = step.primitive.produce(step_inputs)
            if isinstance(step_outputs, pd.Series) and step_i < len(self.steps) - 1:
                # Every step's output but the last step must be a dataframe, since it
                # might be used as the `X` input for a future step.
                step_outputs = pd.DataFrame({"output": step_outputs})
            all_step_outputs.append(step_outputs)

        final_predictions = all_step_outputs[-1]
        if not isinstance(final_predictions, pd.Series):
            raise ValueError(
                f"final pipeline step {self.steps[-1].primitive} "
                "did not output a pandas Series"
            )
        return final_predictions

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fits the pipeline on `X` and `y`, meaning, learns how to use `X`
        to predict `y`.
        
        Parameters
        ----------
        X
            The dataframe of features.
        y
            The series of targets to learn to predict.
        """
        self._run(X, y, fit=True)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes a prediction for each instance in `X`, returning the predictions.
        """
        return self._run(X, None, fit=False)
