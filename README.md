# skplumber

[![Build Status](https://travis-ci.org/epeters3/skplumber.svg?branch=master)](https://travis-ci.org/epeters3/skplumber)

A package for automatically sampling, training, and scoring machine learning pipelines on classification or regression problems. The base constructs (pipelines, primitives, etc.) take heavily from the [Data Driven Discovery of Models (D3M)](https://docs.datadrivendiscovery.org/) core package.

## Getting Started

### Installation

```shell
pip install skplumber
```

### Usage

```python
from skplumber import SKPlumber
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
X = pd.DataFrame(data=dataset["data"], columns=dataset["feature_names"])
y = pd.Series(dataset["target"])

plumber = SKPlumber()
best_pipeline, best_score = plumber.crank(X, y, problem="classification")
print(f"The best test set score the model found was: {best_score}")

# To use the best pipeline on unseen data:
predictions = best_pipeline.predict(unseen_X)
```

## Package Opinions

- A pipeline's final step must be the step that produces the pipeline's final output.
- All missing values are imputed
