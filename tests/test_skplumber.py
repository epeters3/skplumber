from unittest import TestCase

from tests.utils import load_dataset
from skplumber.skplumber import SKPlumber


class TestSKPlumber(TestCase):
    def test_args_are_validated(self) -> None:
        # metric should be valid
        with self.assertRaises(ValueError):
            SKPlumber("classification", 1, metric="foobar")

        # metric should be valid for problem type
        with self.assertRaises(ValueError):
            SKPlumber("classification", 1, metric="rmse")

        # problem type should be valid
        with self.assertRaises(Exception):
            SKPlumber("foobar", 1)

    def test_can_run(self) -> None:
        X, y = load_dataset("iris")

        # Should be able to run with the most basic configuration
        plumber = SKPlumber("classification", 1)
        plumber.fit(X, y)

        # Should be able to run using a non-default metric
        plumber = SKPlumber("classification", 1, metric="f1macro")
        plumber.fit(X, y)

    def test_can_take_callback(self) -> None:
        self.n_iters = 0
        X, y = load_dataset("iris")

        def cb(state) -> bool:
            self.n_iters = state.n_iters
            return True if state.n_iters == 2 else False

        plumber = SKPlumber("classification", 100, callback=cb)
        plumber.fit(X, y)
        assert self.n_iters < 3 and self.n_iters > 0
