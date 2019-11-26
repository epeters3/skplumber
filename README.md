# skplumber

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

X = pd.DataFrame({"feature": [1, 4, 5, 6, 3, 2, 4]})
y = pd.Series({"class": [1, 0, 0, 1 ,1, 1, 0]})
plumber = SKPlumber()
best_pipeline, best_score = plumber.crank(X, y)
print(f"The best test set score the model found was: {best_score})
```

## Package Opinions

- A pipeline's final step must be the step that produces the pipeline's final output.
