from unittest import TestCase

from skplumber.skplumber import SKPlumber
from skplumber.samplers.onestack import OneStackPipelineSampler
from tests.utils import load_dataset


class TestOneStackSampler(TestCase):
    def test_can_sample_for_classification(self) -> None:
        plumber = SKPlumber()
        sampler = OneStackPipelineSampler()
        X, y = load_dataset("titanic")
        plumber.crank(X, y, problem="classification", sampler=sampler, n=3)

    def test_can_sample_for_regression(self) -> None:
        plumber = SKPlumber()
        sampler = OneStackPipelineSampler()
        X, y = load_dataset("boston")
        plumber.crank(X, y, problem="regression", sampler=sampler, n=3)

