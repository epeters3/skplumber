python3 -m unittest
rm -r build dist
python3 setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/* --verbose