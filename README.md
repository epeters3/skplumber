# skplumber

[![Build Status](https://travis-ci.org/epeters3/skplumber.svg?branch=master)](https://travis-ci.org/epeters3/skplumber)

A package for automatically sampling, training, and scoring machine learning pipelines on classification or regression problems. The base constructs (pipelines, primitives, etc.) take heavily from the [Data Driven Discovery of Models (D3M)](https://docs.datadrivendiscovery.org/) core package.

## Getting Started

### Installation

```shell
pip install skplumber
```

### Usage

#### The `SKPlumber` AutoML System

The top-level API of the package is the `SKPlumber` class. You instantiate the class, then use it's `fit` method to perform a search for an optimal machine learning (ML) pipeline, given your input data `X`, and `y` (a `pandas.DataFrame` and `pandas.Series` respectively). Here is an example using the classic iris dataset:

```python
from skplumber import SKPlumber
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
X = pd.DataFrame(data=dataset["data"], columns=dataset["feature_names"])
y = pd.Series(dataset["target"])

# Ask plumber to find the best machine learning pipeline it
# can for the problem in 60 seconds.
plumber = SKPlumber(problem="classification", budget=60)
plumber.fit(X, y)

# To use the best found machine learning pipeline on unseen data:
predictions = plumber.predict(unseen_X)
```

#### `Pipeline`

The `Pipeline` class is a slightly lower level API for the package that can be used to build, fit, and predict arbitrarily shaped machine learning pipelines. For example, we can create a basic single level stacking pipeline, where the output from predictors are fed into another predictor to ensemble in a learned way:

```python
from skplumber import Pipeline
from skplumber.primitives import transformers, classifiers
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
X = pd.DataFrame(data=dataset["data"], columns=dataset["feature_names"])
y = pd.Series(dataset["target"])

# A random imputation of missing values step and one hot encoding of
# non-numeric features step are automatically added.
pipeline = Pipeline()
# Preprocess the inputs
pipeline.add_step(transformers["StandardScalerPrimitive"])
# Save the pipeline step index of the preprocessor's outputs
stack_input = pipeline.curr_step_i
# Add three classifiers to the pipeline that all take the
# preprocessor's outputs as inputs
stack_outputs = []
for clf_name in [
    "LinearDiscriminantAnalysisPrimitive",
    "DecisionTreeClassifierPrimitive",
    "KNeighborsClassifierPrimitive"
]:
    pipeline.add_step(classifiers[clf_name], [stack_input])
    stack_outputs.append(pipeline.curr_step_i)
# Add a final classifier that takes the outputs of all the previous
# three classifiers as inputs
pipeline.add_step(classifiers["RandomForestClassifierPrimitive"], stack_outputs)

# Train the pipeline
pipeline.fit(X, y)

# Have fitted pipeline make predictions
pipeline.predict(X)
```

## Package Opinions

- A pipeline's final step must be the step that produces the pipeline's final output.
- All missing values are imputed.
- All columns of type `object` and `category` are one hot encoded.
