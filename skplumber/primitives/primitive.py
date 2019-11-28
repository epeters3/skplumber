from abc import ABC, abstractmethod

import pandas as pd
import scipy as sp

from skplumber.consts import ProblemType, PrimitiveType


class Primitive(ABC):
    """
    Inherit this class to make a primitive that can work with the
    skplumber framework.
    """

    def __init__(self, primitive_type: PrimitiveType) -> None:
        self.primitive_type = primitive_type
        if primitive_type == PrimitiveType.REGRESSOR:
            self.supported_problem_types = [ProblemType.REGRESSION]
        elif primitive_type == PrimitiveType.CLASSIFIER:
            self.supported_problem_types = [ProblemType.CLASSIFICATION]
        elif primitive_type == PrimitiveType.TRANSFORMER:
            self.supported_problem_types = [
                ProblemType.REGRESSION,
                ProblemType.CLASSIFICATION,
            ]

    @abstractmethod
    def fit(self, X, y) -> None:
        pass

    @abstractmethod
    def produce(self, X):
        pass


def make_sklearn_primitive(sklearn_cls, primitive_type: PrimitiveType):
    class SKPrimitive(Primitive):
        f"""
        An automatically generated `Primitive` implementing wrapper for the
        '{sklearn_cls.__name__}' class in the `sklearn` package.
        """

        def __init__(self) -> None:
            super().__init__(primitive_type)
            self.sk_primitive = sklearn_cls()

        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            """
            This method has the same purpose as the sklearn `fit`
            method. Note that `y` isn't always used, depending
            on the underlying sklearn primitive type.
            """
            self.sk_primitive.fit(X, y)

        def produce(self, X: pd.DataFrame):
            """
            This method calls the method of the underlying
            sklearn primitive that gives its output. For
            transformers its the `transform` method. For
            predictors its the `predict` method, etc.
            """
            if self.primitive_type in [
                PrimitiveType.REGRESSOR,
                PrimitiveType.CLASSIFIER,
            ]:
                outputs = self.sk_primitive.predict(X)
            elif self.primitive_type == PrimitiveType.TRANSFORMER:
                outputs = self.sk_primitive.transform(X)

            print(outputs.shape)

            if sp.sparse.issparse(outputs):
                outputs = outputs.todense()

            if len(outputs.shape) == 1:
                # One column output
                return pd.Series(outputs)
            else:
                return pd.DataFrame(outputs)

    SKPrimitive.__name__ = f"{sklearn_cls.__name__}Primitive"

    return SKPrimitive
