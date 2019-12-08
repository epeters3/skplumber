from sklearn.impute import SimpleImputer

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType

MeanValueImputer = make_sklearn_primitive(SimpleImputer, PrimitiveType.PREPROCESSOR)
