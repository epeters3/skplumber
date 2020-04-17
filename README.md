[![Build Status](https://travis-ci.com/epeters3/skplumber.svg?branch=master)](https://travis-ci.com/github/epeters3/skplumber)

```
       ______         ______                 ______
__________  /____________  /___  ________ ______  /______________
__  ___/_  //_/__  __ \_  /_  / / /_  __ `__ \_  __ \  _ \_  ___/
_(__  )_  ,<  __  /_/ /  / / /_/ /_  / / / / /  /_/ /  __/  /
/____/ /_/|_| _  .___//_/  \__,_/ /_/ /_/ /_//_.___/\___//_/
              /_/
```

skplumber is a Machine Learning (ML) package with two core things to offer:

- An **Automated Machine Learning (AutoML) system** for automatically sampling, training, scoring, and tuning machine learning pipelines on classification or regression problems. This is available as the `skplumber.skplumber.SKPlumber` class.
- A **lightweight ML framework** for composing ML primitives into pipelines (`skplumber.pipeline.Pipeline`) of arbitrary shape, and for training and fitting those pipelines using various evaluation techniques (e.g. train/test split, k-fold cross validation, and down-sampling). Also, all primitive hyperparameters come pre-annotated with types and range information so hyperparameters can be more easily interacted with. Additionally, an existing hyperparameter tuning technique is provided by `skplumber.tuners.ga.ga_tune`.

The base pipeline and primitive constructs take heavily from the same constructs as they exist in the [Data Driven Discovery of Models (D3M)](https://docs.datadrivendiscovery.org/) core package.

API documentation for the project is located [here](https://epeters3.github.io/skplumber/).

## Installation

```shell
pip install skplumber
```

## Usage

### The `SKPlumber` AutoML System

The top-level API of the package is the `skplumber.skplumber.SKPlumber` class. You instantiate the class, then use it's `fit` method to perform a search for an optimal machine learning (ML) pipeline, given your input data `X`, and `y` (a `pandas.DataFrame` and `pandas.Series` respectively). Here is an example using the classic iris dataset:

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

### `Pipeline`

The `skplumber.pipeline.Pipeline` class is a slightly lower level API for the package that can be used to build, fit, and predict arbitrarily shaped machine learning pipelines. For example, we can create a basic single level stacking pipeline, where the output from predictors are fed into another predictor to ensemble in a learned way:

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
