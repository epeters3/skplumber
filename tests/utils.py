from typing import Dict, Callable

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_boston


def load_sk_dataset(name: str):
    loaders: Dict[str, Callable] = {"iris": load_iris, "boston": load_boston}
    assert name in loaders
    dataset = loaders[name]()
    X = pd.DataFrame(data=dataset["data"], columns=dataset["feature_names"])
    y = pd.Series(dataset["target"])
    return X, y
