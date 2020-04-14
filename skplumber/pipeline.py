import typing as t

import pandas as pd

from skplumber.primitives.primitive import Primitive
from skplumber.primitives.custom_primitives.preprocessing import (
    OneHotEncoder,
    RandomImputer,
)
from skplumber.primitives.parammeta import ParamMeta


class PrimitiveStep:
    def __init__(self, primitive_cls: t.Type[Primitive], inputs: t.List[int], **params):
        """
        Parameters
        ----------
        primitive_cls : class inheriting Primitive
            The primitive class this step is associated with.
        inputs : list of int
            The indices of the pipeline steps for whose output
            this step will use as its input.
        params : kwargs
            Any hyperparameters to set in the primitive.
        """
        self.primitive = primitive_cls(**params)
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

    @property
    def param_metas(self) -> t.Dict[int, t.Dict[str, ParamMeta]]:
        return {i: step.primitive.param_metas for i, step in enumerate(self.steps)}

    @property
    def num_params(self) -> int:
        return sum(len(step.primitive.param_metas) for step in self.steps)

    def param_metas_with_data(
        self, X: pd.DataFrame
    ) -> t.Dict[int, t.Dict[str, ParamMeta]]:
        return {
            i: step.primitive.param_metas_with_data(X)
            for i, step in enumerate(self.steps)
        }

    def get_params(self) -> t.Dict[int, t.Dict[str, t.Any]]:
        """
        Get all the pipeline's tunable hyperparameters. A given
        param for a given step can be accessed via e.g.:
        ```
        params = pipeline.get_params()
        params[0]["criterion"]
        ```
        That yields the value of the `"criterion"` param of
        the 0th step in the pipeline.
        """
        return {i: step.primitive.get_params() for i, step in enumerate(self.steps)}

    def set_params(self, params: t.Dict[int, t.Dict[str, t.Any]]) -> None:
        """
        Sets any tunable hyperparameters on one or more steps in the
        pipeline. E.g. to set the `"criterion"` param of the 0th step:
        ```
        pipeline.set_params({0: {"criterion": "gini"}})
        ```
        """
        for i, step_params in params.items():
            if i < 0 or i >= len(self.steps):
                raise ValueError(f"pipeline does not have a step at index {i}")
            self.steps[i].primitive.set_params(**step_params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fits the pipeline on `X` and `y`, meaning, learns how to use `X`
        to predict `y`.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe of features.
        y : pandas.Series
            The series of targets to learn to predict.
        """
        self._run(X, y, fit=True)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes a prediction for each instance in `X`, returning the predictions.
        """
        return self._run(X, None, fit=False)

    def __str__(self) -> str:
        string = f"Pipeline object with {len(self.steps)} steps:"
        for step in self.steps:
            string += "\n\t" + str(step.primitive)
        return string
