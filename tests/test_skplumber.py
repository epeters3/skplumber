from unittest import TestCase

from tests.utils import load_dataset
from skplumber.skplumber import SKPlumber


class TestSKPlumber(TestCase):
    def test_args_are_validated(self) -> None:
        plumber = SKPlumber()
        X, y = load_dataset("iris")

        # metric should be valid
        with self.assertRaises(ValueError):
            plumber.crank(X, y, problem="classification", metric="foobar")

        # metric should be valid for problem type
        with self.assertRaises(ValueError):
            plumber.crank(X, y, problem="classification", metric="rmse")

        # problem type should be valid
        with self.assertRaises(Exception):
            plumber.crank(X, y, problem="foobar")

    def test_can_run(self) -> None:
        plumber = SKPlumber()
        X, y = load_dataset("iris")

        # Should be able to run with the most basic configuration
        plumber.crank(X, y, problem="classification", n=1)

        # Should be able to run using a non-default metric
        plumber.crank(X, y, problem="classification", metric="f1macro", n=1)

    def test_can_take_callback(self) -> None:
        self.n_iters = 0
        plumber = SKPlumber()
        X, y = load_dataset("iris")

        def cb(state) -> bool:
            self.n_iters = state.nit
            return True if state.nit == 2 else False

        plumber.crank(X, y, problem="classification", n=10, callback=cb)
        assert self.n_iters < 3 and self.n_iters > 0
