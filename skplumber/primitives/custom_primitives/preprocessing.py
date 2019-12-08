from typing import Dict

import pandas as pd
import numpy as np

from skplumber.primitives.primitive import Primitive
from skplumber.consts import PrimitiveType


class OneHotEncoder(Primitive):
    """
    One-hot encodes any `object` or `category` columns. If the number of
    unique values is large, it just encodes the most common ones. NaN values
    are not encoded. This primitive is heavily inspired by USC ISI's DSBOX
    encoder primitive used in the D3M ecosystem. See:
    https://github.com/usc-isi-i2/dsbox-primitives
    """

    # the max number of most common values to
    # one-hot encode for each column
    top_n = 10

    def __init__(self) -> None:
        super().__init__(PrimitiveType.PREPROCESSOR)
        self.onehot_col_names_to_vals: Dict[str, pd.Series] = {}

    def fit(self, X, y) -> None:
        # Get the categorical columns
        categoricals = X.select_dtypes(include=["object", "category"])
        for col_name in categoricals.columns:
            # Get the `self.top_n` values that occur most frequently
            # in the column.
            top_n_vals = pd.Series(
                categoricals[col_name].value_counts().nlargest(self.top_n).index
            )
            self.onehot_col_names_to_vals[col_name] = top_n_vals

    def produce(self, X):
        # Use pd.get_dummies() to do the encoding then only keep columns
        # who are found in the map created in `self.fit`.
        categoricals = X.select_dtypes(include=["object", "category"])
        one_hotted = pd.get_dummies(categoricals)
        result = X.copy()

        for col_name, vals_to_onehot in self.onehot_col_names_to_vals.items():
            # get rid of the un-encoded column, then add the
            # one-hot encoded ones.
            result = result.drop(col_name, axis=1)
            for val in vals_to_onehot:
                onehot_col_name = f"{col_name}_{val}"
                result[onehot_col_name] = one_hotted[onehot_col_name]

        return result
