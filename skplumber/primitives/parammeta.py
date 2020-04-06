from abc import ABC, abstractmethod
import typing as t

import pandas as pd


class ParamMeta(ABC):
    """
    An asbtract base class for describing the metadata of
    a primitive hyperparameter. Allows hyperparameters to
    be more easily sampled or optimized. A given
    hyperparameter only needs to be described in this way
    if the implementer wants it to be automatically samplable
    and optimizable by the SKPlumber package. Any hyperparameters
    not described in this way can still be set manually by the
    user.
    """

    @abstractmethod
    def with_data(self, X: pd.DataFrame) -> "ParamMeta":
        """
        Should return a copy of self but with all attributes
        represented in hard, literal values, given that `X` is
        being passed in. E.g. an `int` instead of a function
        that computes an `int`, given `X`.
        """
        pass


class NumericParamMeta(ParamMeta):
    """
    Abstract base class for describing the metadata of a
    numeric primitive hyperparameter (e.g. a float or int).
    """

    def __init__(
        self,
        lbound: t.Union[int, float, t.Callable],
        ubound: t.Union[int, float, t.Callable],
    ) -> None:
        """
        Parameters
        ----------
        lbound
            The lower bound that the hyperparameter's value can
            take on.
        ubound
            The upper bound that the hyperparameter's value can
            take on.

        Sometimes the bounds of a hyperparameter are dependent on
        the dataset being trained on. If `lbound` or `ubound` is
        a function, it will be called, passing the training data
        (i.e. the `X` Pandas DataFrame) as the first and only
        argument. This so the bound can be computed having the
        context of the input dataset.
        """
        self.lbound: t.Union[int, float, t.Callable] = lbound
        self.ubound: t.Union[int, float, t.Callable] = ubound

    def with_data(self, X: pd.DataFrame) -> "NumericParamMeta":
        lbound = self.lbound(X) if callable(self.lbound) else self.lbound
        ubound = self.ubound(X) if callable(self.ubound) else self.ubound
        # `type(self)` ensures a `IntParamMeta` or `FloatParamMeta` is
        # returned. Source:
        # https://stackoverflow.com/questions/7840911/python-inheritance-return-subclass
        return type(self)(lbound, ubound)


class CategoricalParamMeta(ParamMeta):
    def __init__(self, options: t.Iterable[t.Any]) -> None:
        """
        Parameters
        ----------
        options
            The options that this hyperparameter can take on.
        """
        self.options = set(options)

    def with_data(self, X: pd.DataFrame) -> "CategoricalParamMeta":
        options = [o(X) if callable(o) else o for o in self.options]
        return CategoricalParamMeta(options)


class BoolParamMeta(ParamMeta):
    def with_data(self, X: pd.DataFrame) -> "BoolParamMeta":
        """
        Bools don't use `X` in any way, so no copy needed.
        """
        return self


class IntParamMeta(NumericParamMeta):
    def __init__(
        self, lbound: t.Union[int, t.Callable], ubound: t.Union[int, t.Callable]
    ) -> None:
        super().__init__(lbound, ubound)


class FloatParamMeta(NumericParamMeta):
    def __init__(
        self, lbound: t.Union[float, t.Callable], ubound: t.Union[float, t.Callable]
    ) -> None:
        super().__init__(lbound, ubound)
