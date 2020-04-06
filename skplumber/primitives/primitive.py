from abc import ABC, abstractmethod
import typing as t

import pandas as pd
import scipy as sp
from sklearn.base import BaseEstimator

from skplumber.consts import ProblemType, PrimitiveType
from skplumber.primitives.parammeta import ParamMeta


class Primitive(ABC):
    """
    Inherit this class to make a primitive that can work with the
    skplumber framework. A few methods are given for free.
    """

    def __init__(self, *args, **params):
        self.set_params(**params)

    @property
    @abstractmethod
    def primitive_type(self) -> PrimitiveType:
        pass

    @property
    @abstractmethod
    def param_metas(self) -> t.Dict[str, ParamMeta]:
        pass

    def param_metas_with_data(self, X: pd.DataFrame) -> t.Dict[str, ParamMeta]:
        return {
            key: param_meta.with_data(X) for key, param_meta in self.param_metas.items()
        }

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

    def get_params(self) -> t.Dict[str, t.Any]:
        """
        Get all the primitive's tunable hyperparameters.
        """
        return {key: getattr(self, key) for key in self.param_metas.keys()}

    def set_params(self, **params) -> None:
        valid_params = self.param_metas.keys()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for primitive {self}. "
                    "Check the list of available parameters "
                    "with `primitive.param_metas.keys()`."
                )
            setattr(self, key, value)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += ", ".join(
            [f"{key}={repr(value)}" for key, value in self.get_params().items()]
        )
        s += ")"
        return s

    @abstractmethod
    def fit(self, X, y) -> None:
        pass

    @abstractmethod
    def produce(self, X):
        pass


def make_sklearn_primitive(
    sk_cls: BaseEstimator,
    prim_type: PrimitiveType,
    param_metas_dict: t.Dict[str, ParamMeta],
):
    """
    A class factory for automatically creating `Primitive`s out of
    Scikit-Learn estimators.
    """

    class SKPrimitive(Primitive):
        f"""
        An automatically generated `Primitive` implementing wrapper for the
        '{sk_cls.__name__}' class in the `sklearn` package.
        """

        primitive_type = prim_type
        sklearn_cls = sk_cls
        param_metas = param_metas_dict

        def __init__(self, *args, **params) -> None:
            self.sk_primitive = self.sklearn_cls()
            params_from_sk_primitive = {
                key: value
                for key, value in self.sk_primitive.get_params().items()
                if key in self.param_metas.keys()
            }
            params_from_sk_primitive.update(params)
            self.set_params(**params_from_sk_primitive)

        def set_params(self, **params) -> None:
            super().set_params(**params)
            self.sk_primitive.set_params(**params)

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
