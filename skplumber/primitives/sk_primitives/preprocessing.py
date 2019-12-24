from sklearn.impute import SimpleImputer

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType

_preprocessors = [SimpleImputer]

preprocessing_primitives = {}
for est in _preprocessors:
    primitive = make_sklearn_primitive(est, PrimitiveType.PREPROCESSOR)
    preprocessing_primitives[primitive.__name__] = primitive
