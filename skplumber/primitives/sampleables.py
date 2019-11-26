"""
The full collection of primitives that can be sampled i.e. all the
transformers, regressors, and classifiers.
"""
from sklearn.base import RegressorMixin, ClassifierMixin, TransformerMixin

from skplumber.primitives.sk_primitives.classifiers import classifier_primitives
from skplumber.primitives.sk_primitives.regressors import regressor_primitives
from skplumber.primitives.sk_primitives.transformers import transformer_primitives


def validate_primitives(prim_list: list, subclass):
    for prim in prim_list:
        if not isinstance(prim.sk_primitive, subclass):
            print(f"{prim.sk_primitive} should be a {subclass} but is not")


validate_primitives(classifier_primitives, ClassifierMixin)
validate_primitives(regressor_primitives, RegressorMixin)
validate_primitives(transformer_primitives, TransformerMixin)
