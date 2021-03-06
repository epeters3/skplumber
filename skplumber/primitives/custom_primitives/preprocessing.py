import typing as t

import pandas as pd
import numpy as np

from skplumber.primitives.primitive import Primitive
from skplumber.consts import PrimitiveType


class OneHotEncoder(Primitive):
    """
    One-hot encodes any `object` or category` columns. If the number of
    unique values is large, it just encodes the most common ones. `nan` values
    are not encoded. This primitive is heavily inspired by USC ISI's DSBOX
    encoder primitive used in the D3M ecosystem. See:
    https://github.com/usc-isi-i2/dsbox-primitives
    """

    primitive_type = PrimitiveType.PREPROCESSOR
    param_metas = {}

    def __init__(self, top_n: int = 10) -> None:
        # the max number of most common values to
        # one-hot encode for each column
        self.top_n = top_n
        self.onehot_col_names_to_vals: t.Dict[str, pd.Series] = {}

    def fit(self, X, y) -> None:
        # We could be fitting on a new dataset with different columns,
        # so forget everything from the last dataset.
        self.onehot_col_names_to_vals.clear()

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
        if len(self.onehot_col_names_to_vals) == 0:
            # This dataset does not need one hot encoding
            return X

        # Use pd.get_dummies() to do the encoding then only keep columns
        # who are found in the map created in `self.fit`.
        categoricals = X.select_dtypes(include=["object", "category"])
        one_hotted = pd.get_dummies(categoricals)
        result = X.copy()

        for col_name, vals_to_onehot in self.onehot_col_names_to_vals.items():
            # get rid of the un-encoded column, then add the
            # one-hot encoded ones, only adding the ones that
            # were created in `self.fit`.
            result = result.drop(col_name, axis=1)
            for val in vals_to_onehot:
                onehot_col_name = f"{col_name}_{val}"
                if onehot_col_name in one_hotted.columns:
                    result[onehot_col_name] = one_hotted[onehot_col_name]
                else:
                    result[onehot_col_name] = 0

        return result


class RandomImputer(Primitive):
    """
    Imputes missing values for each column by randomly sampling
    from the known values of that column. Has the benefit of
    preserving the column's distribution.
    """

    primitive_type = PrimitiveType.PREPROCESSOR
    param_metas = {}

    def __init__(self) -> None:
        self.col_names_to_known_vals: t.Dict[str, pd.Series] = {}
        self.cols_to_drop: t.Set[str] = set()

    def fit(self, X, y) -> None:
        # We could be fitting on a new dataset with different columns,
        # so forget everything from the last dataset.
        self.col_names_to_known_vals.clear()
        self.cols_to_drop.clear()

        for col in X:
            if X[col].isna().all():
                # This column has no values, so we won't have any values
                # to sample from when imputing values for it. So we drop
                # it at produce time.
                self.cols_to_drop.add(col)
                continue

            # The index of a series returned by `pd.Series.value_counts`
            # holds the values, and the actual entries of the series hold
            # the proportions those values have in `X`.
            self.col_names_to_known_vals[col] = X[col].value_counts(normalize=True)

    def produce(self, X):
        result = X.copy()
        result.drop(self.cols_to_drop, axis="columns", inplace=True)

        # Impute missing values using the known values found
        # in `self.fit`
        for col, known_vals in self.col_names_to_known_vals.items():

            # Fill all missing values with values sampled from the
            # distribution observed for this column in the `self.fit`
            # method.
            fill_vals = pd.Series(
                np.random.choice(known_vals.index, p=known_vals, size=len(result.index))
            )

            # The indices of fill_vals and result need to match so
            # every NaN in result can have a companion value in
            # `fill_vals` to be filled with.
            fill_vals.index = result.index
            result[col].fillna(
                fill_vals, inplace=True,
            )

        return result
