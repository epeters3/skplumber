from typing import Callable, List

import pandas as pd

from skplumber.consts import ProblemType, PrimitiveType, SKLearnAPIType

class Primitive:
    def __init__(
        self,
        f: Callable,
        *,
        primitive_type: PrimitiveType,
        sklearn_api_type: SKLearnAPIType,
        supported_problem_types: List[ProblemType],
    ) -> None:
        """
        TODO: Document
        """
        self.f = f # The actual underlying sklearn primitive
        self.primitive_type = primitive_type
        self.sklearn_api_type = sklearn_api_type
        self.supported_problem_types = supported_problem_types
    
    def fit(self, X: pd.DataFrame, target_name: str):
        """
        This method has the same purpose as the sklearn `fit`
        method, only uses a `target_name` variable as well.
        Note that `target_name` isn't always used, depending
        on the underlying sklearn primitive type.
        """
        pass
    
    def produce(self, X: pd.DataFrame, target_name: str):
        """
        This method calls the method of the underlying
        sklearn primitive that gives its output. For
        transformers its the `transform` method. For
        predictors its the `predict` method, etc.
        """
        pass