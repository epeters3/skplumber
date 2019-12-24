from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.feature_selection import (
    SelectPercentile,
    SelectKBest,
    SelectFpr,
    SelectFwe,
    VarianceThreshold,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
    TSNE,
)

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType

_transformers = [
    MinMaxScaler,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    SelectPercentile,
    # Default hyperparameters don't always work
    # SelectKBest,
    SelectFpr,
    SelectFwe,
    VarianceThreshold,
    PCA,
    RandomTreesEmbedding,
    Isomap,
    LocallyLinearEmbedding,
    # These have no `transform` method
    # MDS,
    # SpectralEmbedding,
    # TSNE,
]

transformers = {}
for est in _transformers:
    primitive = make_sklearn_primitive(est, PrimitiveType.TRANSFORMER)
    transformers[primitive.__name__] = primitive
