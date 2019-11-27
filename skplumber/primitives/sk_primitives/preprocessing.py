from sklearn.impute import SimpleImputer
import pandas as pd

from skplumber.primitives.primitive import make_sklearn_primitive, Primitive
from skplumber.consts import PrimitiveType

MeanValueImputer = make_sklearn_primitive(SimpleImputer, PrimitiveType.TRANSFORMER)


class OneHotEncoder(Primitive):
    """
    A naive one hot encoder. Just wraps
    `pandas.get_dummies`.
    """

    def __init__(self) -> None:
        super().__init__(PrimitiveType.TRANSFORMER)

    def fit(self, X, y) -> None:
        pass

    def produce(self, X):
        return pd.get_dummies(X)
