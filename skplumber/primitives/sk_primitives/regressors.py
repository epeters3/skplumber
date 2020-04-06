import typing as t

from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType
from skplumber.primitives.parammeta import (
    ParamMeta,
    IntParamMeta,
    FloatParamMeta,
    CategoricalParamMeta,
    BoolParamMeta,
)

_regressors: t.List[t.Tuple[BaseEstimator, t.Dict[str, ParamMeta]]] = [
    (
        ARDRegression,
        {
            "n_iter": IntParamMeta(100, 1000),
            "tol": FloatParamMeta(1e-8, 1e-2),
            "alpha_1": FloatParamMeta(1e-8, 1e4),
            # TODO: more
        },
    ),
    (
        BayesianRidge,
        {
            "n_iter": IntParamMeta(100, 1000),
            "tol": FloatParamMeta(1e-8, 1e-2),
            "alpha_1": FloatParamMeta(1e-8, 1e4),
            # TODO: more
        },
    ),
    (
        ElasticNet,
        {
            "alpha": FloatParamMeta(0.0, 1e4),
            "l1_ratio": FloatParamMeta(0.01, 1.0),
            "selection": CategoricalParamMeta(["cyclic", "random"]),
            # TODO: more
        },
    ),
    (
        HuberRegressor,
        {
            "epsilon": FloatParamMeta(1.0 + 1e-8, 1e4),
            "max_iter": IntParamMeta(100, int(1e6)),
            "alpha": FloatParamMeta(0.0, 1e4),
        },
        # TODO: more
    ),
    (Lars, {"n_nonzero_coefs": IntParamMeta(1, int(1e10))}),
    (
        Lasso,
        {
            "alpha": FloatParamMeta(1e-10, 1e4),
            "selection": CategoricalParamMeta(["cyclic", "random"]),
        },
        # TODO: more
    ),
    (
        LassoLars,
        {"alpha": FloatParamMeta(1e-10, 1e4), "max_iter": IntParamMeta(100, int(1e6))},
        # TODO: more
    ),
    (LinearRegression, {}),
    (
        PassiveAggressiveRegressor,
        {
            "early_stopping": BoolParamMeta(),
            "loss": CategoricalParamMeta(
                ["epsilon_insensitive", "squared_epsilon_insensitive"]
            ),
            # TODO: more
        },
    ),
    (
        RANSACRegressor,
        {"loss": CategoricalParamMeta(["absolute_loss", "squared_loss"])},
        # TODO: more
    ),
    (
        Ridge,
        {
            "alpha": FloatParamMeta(0.0, 1e4),
            "tol": FloatParamMeta(1e-8, 1e-2),
            "solver": CategoricalParamMeta(
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
            ),
        },
    ),
    (
        SGDRegressor,
        {
            "loss": CategoricalParamMeta(
                [
                    "squared_loss",
                    "huber",
                    "epsilon_insensitive",
                    "squared_epsilon_insensitive",
                ]
            ),
            "penalty": CategoricalParamMeta(["l2", "l1", "elasticnet"]),
            "alpha": FloatParamMeta(0.0, 1e4),
            # TODO: more
        },
    ),
    (
        TheilSenRegressor,
        {"max_iter": IntParamMeta(100, int(1e6)), "tol": FloatParamMeta(1e-8, 1e-2)},
    ),
    (
        RandomForestRegressor,
        {
            "n_estimators": IntParamMeta(2, 200),
            "criterion": CategoricalParamMeta(["mse", "mae"]),
            "max_depth": IntParamMeta(1, lambda X: X.shape[0] // 2),
            # TODO: more
        },
    ),
    (
        ExtraTreesRegressor,
        {
            "n_estimators": IntParamMeta(2, 200),
            "criterion": CategoricalParamMeta(["mse", "mae"]),
            "max_depth": IntParamMeta(1, lambda X: X.shape[0] // 2),
            # TODO: more
        },
    ),
    (
        AdaBoostRegressor,
        {
            "n_estimators": IntParamMeta(2, 200),
            "loss": CategoricalParamMeta(["linear", "square", "exponential"]),
        },
    ),
    (
        GradientBoostingRegressor,
        {
            "loss": CategoricalParamMeta(["ls", "lad", "huber", "quantile"]),
            "n_estimators": IntParamMeta(2, 200),
            "min_samples_split": IntParamMeta(2, lambda X: X.shape[0]),
            # TODO: more
        },
    ),
    (
        DecisionTreeRegressor,
        {
            "criterion": CategoricalParamMeta(["mse", "friedman_mse", "mae"]),
            "splitter": CategoricalParamMeta(["best", "random"]),
            "max_depth": IntParamMeta(1, lambda X: X.shape[0] // 2),
            # TODO: more
        },
    ),
    (
        KNeighborsRegressor,
        {
            "n_neighbors": IntParamMeta(1, lambda X: X.shape[0] // 2),
            "weights": CategoricalParamMeta(["uniform", "distance"]),
            "metric": CategoricalParamMeta(
                [
                    "euclidean",
                    "manhattan",
                    "chebyshev",
                    "wminkowski",
                    "seuclidean",
                    "mahalanobis",
                    "minkowski",
                ]
            ),
        },
    ),
    (
        SVR,
        {
            "C": FloatParamMeta(1e-10, 1e4),
            "kernel": CategoricalParamMeta(["linear", "poly", "rbf", "sigmoid"]),
            "shrinking": BoolParamMeta(),
            # TODO: more
        },
    ),
    (
        LinearSVR,
        {
            "epsilon": FloatParamMeta(0.0, 1e4),
            "loss": CategoricalParamMeta(
                ["epsilon_insensitive", "squared_epsilon_insensitive"]
            ),
            "C": FloatParamMeta(1e-10, 1e4),
            # TODO: more
        },
    ),
    (
        KernelRidge,
        {
            "alpha": FloatParamMeta(0.0, 1e4),
            "kernel": CategoricalParamMeta(
                [
                    "additive_chi2",
                    "chi2",
                    "linear",
                    "poly",
                    "polynomial",
                    "rbf",
                    "laplacian",
                    "sigmoid",
                    "cosine",
                ]
            ),
        },
    ),
    (
        MLPRegressor,
        {
            "activation": CategoricalParamMeta(
                ["identity", "logistic", "tanh", "relu"]
            ),
            "solver": CategoricalParamMeta(["lbfgs", "sgd", "adam"]),
            "alpha": FloatParamMeta(0.0, 1e4),
            "learning_rate": CategoricalParamMeta(
                ["constant", "invscaling", "adaptive"]
            ),
            "tol": FloatParamMeta(1e-8, 1e-2),
            "early_stopping": BoolParamMeta(),
        },
    ),
]

regressors = {}
for est, param_metas in _regressors:
    primitive = make_sklearn_primitive(est, PrimitiveType.REGRESSOR, param_metas)
    regressors[primitive.__name__] = primitive
