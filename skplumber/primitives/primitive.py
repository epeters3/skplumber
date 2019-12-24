from abc import ABC, abstractmethod
import typing as t

import pandas as pd
import scipy as sp

from skplumber.consts import ProblemType, PrimitiveType


class Primitive(ABC):
    """
    Inherit this class to make a primitive that can work with the
    skplumber framework.
    """

    @property
    @abstractmethod
    def primitive_type(self) -> PrimitiveType:
        pass

    @property
    def supported_problem_types(self) -> t.List[ProblemType]:
        if self.primitive_type == PrimitiveType.REGRESSOR:
            return [ProblemType.REGRESSION]
        elif self.primitive_type == PrimitiveType.CLASSIFIER:
            return [ProblemType.CLASSIFICATION]
        elif self.primitive_type == PrimitiveType.TRANSFORMER:
            return [
                ProblemType.REGRESSION,
                ProblemType.CLASSIFICATION,
            ]
        else:
            raise ValueError(
                f"class {self.__class__.__name__} has an invalid primitive type"
            )

    @abstractmethod
    def fit(self, X, y) -> None:
        pass

    @abstractmethod
    def produce(self, X):
        pass


def make_sklearn_primitive(sk_cls, prim_type: PrimitiveType):
    class SKPrimitive(Primitive):
        f"""
        An automatically generated `Primitive` implementing wrapper for the
        '{sk_cls.__name__}' class in the `sklearn` package.
        """

        primitive_type = prim_type
        sklearn_cls = sk_cls

        def __init__(self) -> None:
            self.sk_primitive = self.sklearn_cls()

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
            elif self.primitive_type in [
                PrimitiveType.TRANSFORMER,
                PrimitiveType.PREPROCESSOR,
            ]:
                outputs = self.sk_primitive.transform(X)

            if sp.sparse.issparse(outputs):
                outputs = outputs.todense()

            if len(outputs.shape) == 1:
                # One column output
                return pd.Series(outputs)
            else:
                return pd.DataFrame(outputs)

    SKPrimitive.__name__ = f"{sk_cls.__name__}Primitive"

    return SKPrimitive
