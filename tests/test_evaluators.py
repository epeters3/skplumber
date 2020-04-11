from unittest import TestCase

from tests.utils import load_dataset
from skplumber.skplumber import SKPlumber
from skplumber.evaluators import (
    make_kfold_evaluator,
    make_train_test_evaluator,
    make_down_sample_evaluator,
)


class TestEvaluators(TestCase):
    def test_default_works(self):
        """
        `SKPlumber.fit`'s default evaluator should work
        out of the box.
        """
        plumber = SKPlumber("classification", 1)
        X, y = load_dataset("iris")
        plumber.fit(X, y)

    def test_can_do_train_test(self):
        """
        The evaluator returned by `make_train_test_evaluator`
        should work, and should cupport custom test size.
        """
        plumber = SKPlumber(
            "classification", 1, evaluator=make_train_test_evaluator(0.2)
        )
        X, y = load_dataset("iris")
        plumber.fit(X, y)

    def test_can_do_k_fold_cv(self):
        """
        The evaluator returned by `make_kfold_evaluator`
        should work, and should cupport custom number of folds.
        """
        plumber = SKPlumber("classification", 1, evaluator=make_kfold_evaluator(3))
        X, y = load_dataset("iris")
        # Should be able to do k-fold cross validation.
        plumber.fit(X, y)

    def test_can_do_down_sample_evaluation(self):
        """
        The evaluator returned by `make_down_sample_evaluator`
        should work, and should cupport custom test size.
        """
        plumber = SKPlumber(
            "classification", 1, evaluator=make_down_sample_evaluator(0.8, 0.2)
        )
        X, y = load_dataset("iris")
        # Should be able to do down-sampled train/test validation.
        plumber.fit(X, y)
