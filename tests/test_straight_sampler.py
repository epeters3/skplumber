from unittest import TestCase

from skplumber.skplumber import SKPlumber
from tests.utils import load_sk_dataset


class TestStraightSampler(TestCase):
    def test_can_sample_for_classification(self) -> None:
        plumber = SKPlumber()
        X, y = load_sk_dataset("iris")
        plumber.crank(X, y, problem="classification", n=10)

    def test_can_sample_for_regression(self) -> None:
        plumber = SKPlumber()
        X, y = load_sk_dataset("boston")
        plumber.crank(X, y, problem="regression", n=10)
