## Releasing a New Version

1. Bump the version in `setup.py`
1. Run:
   ```shell
   rm -r build dist
   python setup.py sdist bdist_wheel
   twine check dist/*
   twine upload dist/* --verbose
   ```
1. Pip install the package to verify the version was bumped:
   ```shell
   pip install skplumber
   pip freeze | grep skplumber
   ```
