from unittest import TestCase

from skplumber.pipeline import Pipeline
from skplumber.primitives import classifiers, transformers
from skplumber.tuners.ga import ga_tune
from skplumber.metrics import f1macro
from skplumber.evaluators import make_train_test_evaluator
from skplumber.utils import logger
from tests.utils import load_dataset


class TestGATune(TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = X, y = load_dataset("iris")
        cls.X = X
        cls.y = y

    def test_can_run_basic(self):
        """
        The flexga tuner should be able to complete without
        erroring.
        """
        pipeline = Pipeline()
        pipeline.add_step(classifiers["DecisionTreeClassifierPrimitive"])

        evaluate = make_train_test_evaluator()
        logger.info(f"baseline score: {evaluate(pipeline, self.X, self.y, f1macro)}")
        ga_tune(pipeline, self.X, self.y, evaluate, f1macro, iters=2)

    def test_can_tune_multiple_primitives(self):
        """
        The flexga tuner should be able to tune the hyperparameters
        of all primitives in a pipeline at once.
        """
        pipeline = Pipeline()
        pipeline.add_step(transformers["PCAPrimitive"])
        pipeline.add_step(classifiers["DecisionTreeClassifierPrimitive"])

        evaluate = make_train_test_evaluator()
        logger.info(f"baseline score: {evaluate(pipeline, self.X, self.y, f1macro)}")
        ga_tune(pipeline, self.X, self.y, evaluate, f1macro, iters=2)
