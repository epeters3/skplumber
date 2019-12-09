from unittest import TestCase

import pandas as pd
import numpy as np

from skplumber.primitives.custom_primitives.preprocessing import RandomImputer


class TestRandomImputer(TestCase):
    def test_can_impute_basic(self):
        X = pd.DataFrame({"cat": ["a", "b", None, "a"], "num": [1, np.nan, np.nan, 1]})
        imputed = self._fit_produce(X)
        self.assertFalse(imputed.isna().all().all())

    def _fit_produce(self, X: pd.DataFrame) -> pd.DataFrame:
        imputer = RandomImputer()
        imputer.fit(X, None)
        return imputer.produce(X)
