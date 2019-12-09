from unittest import TestCase
import random

import pandas as pd
import numpy as np

from skplumber.primitives.custom_primitives.preprocessing import OneHotEncoder


class TestOneHotEncoder(TestCase):
    def test_can_encode_basic(self):
        X = pd.DataFrame({"cat": ["a", "b", "c", "a"], "num": [1, 4, 3, 5]})
        encoded = self._fit_produce(X)

        self.assertTrue((encoded.cat_a == pd.Series([1, 0, 0, 1])).all())
        self.assertTrue((encoded.cat_b == pd.Series([0, 1, 0, 0])).all())
        self.assertTrue((encoded.cat_c == pd.Series([0, 0, 1, 0])).all())
        self.assertTrue((encoded.num == pd.Series([1, 4, 3, 5])).all())
        self.assertEqual(len(encoded.columns), 4)

    def test_only_encodes_top_n(self):
        X = pd.DataFrame(
            {
                "cat": [str(num) for num in range(1000)],
                "num": [num for num in range(1000)],
            }
        )
        encoded = self._fit_produce(X)
        cats_only = encoded.drop("num", axis=1)

        self.assertEqual(len(encoded.columns), OneHotEncoder.top_n + 1)
        for col in cats_only:
            self.assertEqual(cats_only[col].sum(), 1)

    def test_doesnt_break_on_nans(self):
        X = pd.DataFrame({"cat": ["a", np.nan, "c", "a"], "num": [1, 4, 3, 5]})
        encoded = self._fit_produce(X)

        self.assertTrue((encoded.cat_a == pd.Series([1, 0, 0, 1])).all())
        self.assertTrue((encoded.cat_c == pd.Series([0, 0, 1, 0])).all())
        self.assertTrue((encoded.num == pd.Series([1, 4, 3, 5])).all())
        self.assertEqual(len(encoded.columns), 3)

    def test_can_handle_new_data(self):
        """
        The output columns should always be the same, even when the
        encoder produces on new data that includes unseen labels or
        which doesn't have previously seen labels.
        """
        X = pd.DataFrame({"cat": ["a", "b", "c", "a"], "num": [1, 4, 3, 5]})
        X_new = pd.DataFrame({"cat": ["a", "b", "b", "d"], "num": [2, 4, 6, 5]})
        encoder = OneHotEncoder()
        encoder.fit(X, None)
        encoded = encoder.produce(X_new)

        self.assertTrue((encoded.cat_a == pd.Series([1, 0, 0, 0])).all())
        self.assertTrue((encoded.cat_b == pd.Series([0, 1, 1, 0])).all())
        self.assertTrue((encoded.cat_c == pd.Series([0, 0, 0, 0])).all())
        self.assertTrue((encoded.num == pd.Series([2, 4, 6, 5])).all())

    def _fit_produce(self, X: pd.DataFrame) -> pd.DataFrame:
        encoder = OneHotEncoder()
        encoder.fit(X, None)
        return encoder.produce(X)
