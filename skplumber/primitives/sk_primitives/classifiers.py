import typing as t

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.base import BaseEstimator

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType
from skplumber.primitives.parammeta import (
    ParamMeta,
    CategoricalParamMeta,
    BoolParamMeta,
    IntParamMeta,
    FloatParamMeta,
)

_classifiers: t.List[t.Tuple[BaseEstimator, t.Dict[str, ParamMeta]]] = [
    (
        DecisionTreeClassifier,
        {
            "criterion": CategoricalParamMeta(["gini", "entropy"]),
            "splitter": CategoricalParamMeta(["best", "random"]),
            "min_samples_split": IntParamMeta(2, lambda X: X.shape[0]),
            # TODO: more
        },
    ),
    (
        MLPClassifier,
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
    # Radius hyperparam needs to be problem-specific to be useful
    # (RadiusNeighborsClassifier, {}),
    (
        KNeighborsClassifier,
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
        SGDClassifier,
        {
            "loss": CategoricalParamMeta(
                ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
            ),
            "penalty": CategoricalParamMeta(["l2", "l1", "elasticnet"]),
            "alpha": FloatParamMeta(0.0, 1e4),
            # TODO: more
        },
    ),
    (
        RidgeClassifier,
        {
            "alpha": FloatParamMeta(0.0, 1e4),
            "solver": CategoricalParamMeta(
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
            ),
        },
    ),
    (
        PassiveAggressiveClassifier,
        {
            "early_stopping": BoolParamMeta(),
            "loss": CategoricalParamMeta(["hinge", "squared_hinge"]),
        },
    ),
    (GaussianProcessClassifier, {}),
    (AdaBoostClassifier, {"n_estimators": IntParamMeta(2, 200)}),
    (
        GradientBoostingClassifier,
        {
            "min_samples_split": IntParamMeta(2, lambda X: X.shape[0]),
            "min_samples_leaf": IntParamMeta(1, lambda X: X.shape[0]),
            "max_depth": IntParamMeta(1, lambda X: X.shape[0] // 2),
            # TODO: more
        },
    ),
    (
        BaggingClassifier,
        {
            "n_estimators": IntParamMeta(2, 200),
            "bootstrap": BoolParamMeta(),
            "bootstrap_features": BoolParamMeta(),
            # TODO: more
        },
    ),
    (
        ExtraTreesClassifier,
        {
            "n_estimators": IntParamMeta(2, 200),
            "criterion": CategoricalParamMeta(["gini", "entropy"]),
            "max_depth": IntParamMeta(1, lambda X: X.shape[0] // 2),
            # TODO: more
        },
    ),
    (
        RandomForestClassifier,
        {
            "n_estimators": IntParamMeta(2, 200),
            "criterion": CategoricalParamMeta(["gini", "entropy"]),
            "max_depth": IntParamMeta(1, lambda X: X.shape[0] // 2),
            # TODO: more
        },
    ),
    (
        BernoulliNB,
        {
            "alpha": FloatParamMeta(0.0, 1e4),
            "binarize": FloatParamMeta(0.0, 1.0),
            "fit_prior": BoolParamMeta(),
        },
    ),
    (GaussianNB, {"var_smoothing": FloatParamMeta(1e-12, 1e-2)}),
    (
        LinearDiscriminantAnalysis,
        {
            "solver": CategoricalParamMeta(["svd", "lsqr", "eigen"]),
            "shrinkage": FloatParamMeta(0.0, 1.0),
        },
    ),
    (
        LinearSVC,
        {
            "penalty": CategoricalParamMeta(["l1", "l2"]),
            "loss": CategoricalParamMeta(["hinge", "squared_hinge"]),
            "C": FloatParamMeta(1e-10, 1e4),
            # TODO: more
        },
    ),
    (
        LogisticRegression,
        {
            "penalty": CategoricalParamMeta(["l1", "l2", "elasticnet"]),
            "C": FloatParamMeta(1e-10, 1e4),
        },
    ),
    # Can't handle negative data
    # (MultinomialNB, {}),
    (
        NearestCentroid,
        {
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
            "shrink_threshold": FloatParamMeta(0.0, 1.0),
        },
    ),
    # Default params throw errors in some cases
    # (NuSVC, {}),
    (
        Perceptron,
        {
            "penalty": CategoricalParamMeta(["l2", "l1", "elasticnet"]),
            "alpha": FloatParamMeta(0.0, 1e4),
            "early_stopping": BoolParamMeta(),
            # TODO: more
        },
    ),
    (
        QuadraticDiscriminantAnalysis,
        {"reg_param": FloatParamMeta(0.0, 1.0), "tol": FloatParamMeta(1e-10, 1e-2)},
    ),
    (
        SVC,
        {
            "C": FloatParamMeta(1e-10, 1e4),
            "kernel": CategoricalParamMeta(["linear", "poly", "rbf", "sigmoid"]),
            "shrinking": BoolParamMeta(),
            # TODO: more
        },
    ),
]

classifiers = {}
for est, param_metas in _classifiers:
    primitive_cls = make_sklearn_primitive(est, PrimitiveType.CLASSIFIER, param_metas)
    classifiers[primitive_cls.__name__] = primitive_cls
