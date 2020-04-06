import typing as t

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType
from skplumber.primitives.parammeta import ParamMeta, CategoricalParamMeta

_preprocessors: t.List[t.Tuple[BaseEstimator, t.Dict[str, ParamMeta]]] = [
    (
        SimpleImputer,
        {"strategy": CategoricalParamMeta(["mean", "median", "most_frequent"])},
    )
]

preprocessors = {}
for est, param_metas in _preprocessors:
    primitive = make_sklearn_primitive(est, PrimitiveType.PREPROCESSOR, param_metas)
    preprocessors[primitive.__name__] = primitive
