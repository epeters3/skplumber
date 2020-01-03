from unittest import TestCase
import time

from skplumber.utils import conditional_timeout, EvaluationTimeoutError


class TestConditionalTimeout(TestCase):
    def test_timeout_raises_when_too_long(self):
        with self.assertRaises(EvaluationTimeoutError):
            with conditional_timeout(1):
                time.sleep(2)

    def test_timeout_doesnt_raise_when_not_too_long(self):
        try:
            with conditional_timeout(2):
                time.sleep(1)
        except Exception:
            self.fail()

    def test_timeout_does_nothing_when_condition_false(self):
        try:
            with conditional_timeout(1, False):
                time.sleep(2)
            with conditional_timeout(2, False):
                time.sleep(1)
        except Exception:
            self.fail()
