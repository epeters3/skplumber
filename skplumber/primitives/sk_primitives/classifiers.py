from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
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
from sklearn.mixture import GaussianMixture

from skplumber.primitives.primitive import make_sklearn_primitive
from skplumber.consts import PrimitiveType

_classifiers = [
    ExtraTreeClassifier,
    DecisionTreeClassifier,
    MLPClassifier,
    # Radius hyperparam needs to be problem-specific to be useful
    # RadiusNeighborsClassifier,
    KNeighborsClassifier,
    SGDClassifier,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    GaussianProcessClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    BernoulliNB,
    GaussianNB,
    LinearDiscriminantAnalysis,
    LinearSVC,
    LogisticRegression,
    # Can't handle negative data
    # MultinomialNB,
    NearestCentroid,
    NuSVC,
    Perceptron,
    QuadraticDiscriminantAnalysis,
    SVC,
    GaussianMixture,
]

classifiers = {}
for est in _classifiers:
    primitive = make_sklearn_primitive(est, PrimitiveType.CLASSIFIER)
    classifiers[primitive.__name__] = primitive
