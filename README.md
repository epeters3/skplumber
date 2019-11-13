# skplumber

## Development Guide:

1. For initial 1.0.0 release, this will be a python package that can take a pandas DF, a target column, and a problem type (i.e. classification, regression), and will build and try out ML pipelines on the data, delivering or storing the best pipeline as a useable model. It will use a common data preprocessing preamble, and try out different feature preprocessors and extractors, and different models as well.

## Releasing

1. Bump the version in `setup.py`
1. Run:
   ```shell
   python setup.py sdist bdist_wheel
   twine check dist/*
   twine upload dist/*
   ```
1. Pip install the package to make sure the version was bumped:
   ```shell
   pip install skplumber
   pip freeze | grep skplumber
   ```
