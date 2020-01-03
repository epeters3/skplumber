from setuptools import setup, find_packages

setup(
    name="skplumber",
    version="0.4.6dev",
    packages=find_packages(include=["skplumber", "skplumber.*"]),
    license="MIT",
    url="https://github.com/epeters3/skplumber",
    author="Evan Peterson",
    author_email="evanpeterson17@gmail.com",
    description="A scikit-learn based AutoML tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "scikit-learn>=0.21.3",
        "pandas>=0.25.3",
        "pytest>=5.2.4",
        "Cython==0.29.14",
        "scipy>=1.3.2",
        "colorlog>=4.0.2",
    ],
)
