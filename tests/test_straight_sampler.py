from unittest import TestCase

from skplumber.skplumber import SKPlumber
from skplumber.samplers.straight import StraightPipelineSampler
from tests.utils import load_sk_dataset


class TestStraightSampler(TestCase):
    def test_can_sample_for_classification(self) -> None:
        plumber = SKPlumber()
        sampler = StraightPipelineSampler()
        X, y = load_sk_dataset("iris")
        plumber.crank(X, y, problem="classification", sampler=sampler, n=3)

    def test_can_sample_for_regression(self) -> None:
        plumber = SKPlumber()
        sampler = StraightPipelineSampler()
        X, y = load_sk_dataset("boston")
        plumber.crank(X, y, problem="regression", sampler=sampler, n=3)

    def test_can_sample_multiple_preprocessors(self) -> None:
        plumber = SKPlumber()
        sampler = StraightPipelineSampler(preprocessors=2)
        X, y = load_sk_dataset("boston")
        best_pipeline, _ = plumber.crank(
            X, y, problem="regression", sampler=sampler, n=1
        )
        self.assertEqual(len(best_pipeline.steps), 5)
