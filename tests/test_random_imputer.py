from unittest import TestCase

import pandas as pd
import numpy as np

from skplumber.primitives.custom_primitives.preprocessing import RandomImputer


class TestRandomImputer(TestCase):
    def test_can_impute_basic(self):
        X = pd.DataFrame({"cat": ["a", "b", None, "a"], "num": [1, np.nan, np.nan, 1]})
        imputed = self._fit_produce(X)
        self.assertFalse(imputed.isna().all().all())

    def test_can_impute_with_empty_columns(self):
        """
        If a column has no values, the imputer will not have any
        values to sample from to fill null values in that column,
        so should just drop the columnn.
        """
        X_with_all_nans = pd.DataFrame(
            {
                "cat": ["a", "b", None, "a"],
                "num": [1, np.nan, np.nan, 1],
                "nans": [np.nan, np.nan, np.nan, np.nan],
            }
        )
        imputed = self._fit_produce(X_with_all_nans)
        self.assertFalse(imputed.isna().all().all())  # no null values
        self.assertEqual(
            tuple(imputed.columns), ("cat", "num")
        )  # empty column was dropped

        X_with_all_nones = pd.DataFrame(
            {
                "cat": ["a", "b", None, "a"],
                "num": [1, np.nan, np.nan, 1],
                "nones": [None, None, None, None],
            }
        )
        imputed = self._fit_produce(X_with_all_nones)
        self.assertFalse(imputed.isna().all().all())
        self.assertEqual(tuple(imputed.columns), ("cat", "num"))

    def test_can_fit_on_subsequent_dataset(self):
        """
        Once fit on a dataset, it should be able to be fit on a totally new
        and different dataset without any side effects from the old dataset.
        """
        X1 = pd.DataFrame({"cat1": ["a", None, "c", "a"], "num1": [np.nan, 4, 3, 5]})
        X2 = pd.DataFrame({"cat2": ["d", "f", None, "g"], "num2": [2, 6, np.nan, 7]})
        imputer = RandomImputer()

        # Fit on first dataset
        imputer.fit(X1, None)
        imputer.produce(X1)
        # Fit on a different second dataset
        imputer.fit(X2, None)
        imputer.produce(X2)

    def _fit_produce(self, X: pd.DataFrame) -> pd.DataFrame:
        imputer = RandomImputer()
        imputer.fit(X, None)
        return imputer.produce(X)
