rm -r build dist
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/* --verbose