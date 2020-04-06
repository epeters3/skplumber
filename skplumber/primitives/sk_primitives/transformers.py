import typing as t

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
from sklearn.base import BaseEstimator

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType
from skplumber.primitives.parammeta import (
    ParamMeta,
    BoolParamMeta,
    IntParamMeta,
    FloatParamMeta,
    CategoricalParamMeta,
)

_transformers: t.List[t.Tuple[BaseEstimator, t.Dict[str, ParamMeta]]] = [
    (MinMaxScaler, {}),
    (
        PolynomialFeatures,
        {"interaction_only": BoolParamMeta(), "include_bias": BoolParamMeta()},
    ),
    (
        QuantileTransformer,
        {
            "n_quantiles": IntParamMeta(2, lambda X: X.shape[0]),
            "output_distribution": CategoricalParamMeta(["uniform", "normal"]),
        },
    ),
    (RobustScaler, {},),
    (StandardScaler, {}),
    (SelectPercentile, {"percentile": IntParamMeta(10, 100)}),
    # Default hyperparameters don't always work
    # (SelectKBest, {}),
    (SelectFpr, {"alpha": FloatParamMeta(0.05, 1.0)}),
    (SelectFwe, {"alpha": FloatParamMeta(0.05, 1.0)}),
    (
        VarianceThreshold,
        # The range is from 0 (no variance) to about everything but the
        # column in `X` with highest variance.
        {"threshold": FloatParamMeta(0.0, lambda X: X.var().max() - 1e-10)},
    ),
    (
        PCA,
        {
            "n_components": IntParamMeta(1, lambda X: min(X.shape)),
            "whiten": BoolParamMeta(),
        },
    ),
    (
        RandomTreesEmbedding,
        {
            "n_estimators": IntParamMeta(2, 200),
            "max_depth": IntParamMeta(1, lambda X: X.shape[0] // 2),
            "min_samples_split": IntParamMeta(2, lambda X: X.shape[0]),
            # TODO: more
        },
    ),
    (
        Isomap,
        {
            "n_neighbors": IntParamMeta(1, lambda X: X.shape[0] // 2),
            "n_components": IntParamMeta(1, lambda X: min(X.shape)),
            "metric": CategoricalParamMeta(
                [
                    "cosine",
                    "euclidean",
                    "manhattan",
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "correlation",
                ]
            ),
        },
    ),
    (
        LocallyLinearEmbedding,
        {
            "n_neighbors": IntParamMeta(1, lambda X: X.shape[0] // 2),
            "n_components": IntParamMeta(1, lambda X: min(X.shape)),
            "reg": FloatParamMeta(0.0, 1e4),
        },
    ),
    # These have no `transform` method
    # (MDS, {}),
    # (SpectralEmbedding, {}),
    # (TSNE, {}),
]

transformers = {}
for est, param_metas in _transformers:
    primitive = make_sklearn_primitive(est, PrimitiveType.TRANSFORMER, param_metas)
    transformers[primitive.__name__] = primitive
