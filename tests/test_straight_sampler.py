from unittest import TestCase

from skplumber.skplumber import SKPlumber
from skplumber.samplers.straight import StraightPipelineSampler
from tests.utils import load_dataset


class TestStraightSampler(TestCase):
    def test_can_sample_for_classification(self) -> None:
        sampler = StraightPipelineSampler()
        X, y = load_dataset("titanic")
        plumber = SKPlumber("classification", 1, sampler=sampler)
        plumber.fit(X, y)

    def test_can_sample_for_regression(self) -> None:
        sampler = StraightPipelineSampler()
        X, y = load_dataset("boston")
        plumber = SKPlumber("regression", 1, sampler=sampler)
        plumber.fit(X, y)

    def test_can_sample_multiple_preprocessors(self) -> None:
        sampler = StraightPipelineSampler(preprocessors=2)
        X, y = load_dataset("boston")
        plumber = SKPlumber("regression", 1, sampler=sampler)
        plumber.fit(X, y)
        self.assertEqual(len(plumber.best_pipeline.steps), 5)
