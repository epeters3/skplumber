from unittest import TestCase

from skplumber.skplumber import SKPlumber
from skplumber.samplers.onestack import OneStackPipelineSampler
from tests.utils import load_dataset


class TestOneStackSampler(TestCase):
    def test_can_sample_for_classification(self) -> None:
        sampler = OneStackPipelineSampler()
        plumber = SKPlumber("classification", 1, sampler=sampler)
        X, y = load_dataset("titanic")
        plumber.fit(X, y)

    def test_can_sample_for_regression(self) -> None:
        sampler = OneStackPipelineSampler()
        plumber = SKPlumber("regression", 1, sampler=sampler)
        X, y = load_dataset("boston")
        plumber.fit(X, y)

