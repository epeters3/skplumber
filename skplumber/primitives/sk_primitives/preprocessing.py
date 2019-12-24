from sklearn.impute import SimpleImputer

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType

_preprocessors = [SimpleImputer]

preprocessors = {}
for est in _preprocessors:
    primitive = make_sklearn_primitive(est, PrimitiveType.PREPROCESSOR)
    preprocessors[primitive.__name__] = primitive
