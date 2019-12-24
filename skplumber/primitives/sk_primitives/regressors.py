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

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType

_regressors = [
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
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    DecisionTreeRegressor,
    KNeighborsRegressor,
    SVR,
    LinearSVR,
    KernelRidge,
    MLPRegressor,
]

regressors = {}
for est in _regressors:
    primitive = make_sklearn_primitive(est, PrimitiveType.REGRESSOR)
    regressors[primitive.__name__] = primitive
