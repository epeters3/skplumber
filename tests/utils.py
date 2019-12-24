import typing as t
import os

import pandas as pd
from sklearn.datasets import load_iris, load_boston


def sk_data_to_pd(dataset: dict) -> t.Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(data=dataset["data"], columns=dataset["feature_names"])
    y = pd.Series(dataset["target"])
    return X, y


def load_dataset(name: str) -> t.Tuple[pd.DataFrame, pd.Series]:
    loaders: t.Dict[str, t.Callable] = {
        "iris": lambda: sk_data_to_pd(load_iris()),  # classification
        "boston": lambda: sk_data_to_pd(load_boston()),  # regression
        # Classification with missing values and mixed datatypes
        "titanic": lambda: load_csv(
            os.path.abspath("tests/data/titanic.csv"), "Survived"
        ),
        # Synthetic classification dataset with string targets
        "string-targets": lambda: load_csv(
            os.path.abspath("tests/data/string-targets.csv"), "Class"
        ),
    }
    assert name in loaders
    return loaders[name]()


def load_csv(path: str, target_name: str) -> t.Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(path)
    y = X[target_name]
    X = X.drop(target_name, axis=1)
    return X, y
