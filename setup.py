from setuptools import setup

setup(
    name='skplumber',
    version='0.1dev',
    packages=['skplumber'],
    license='MIT',
    url='https://github.com/epeters3/skplumber',
    author='Evan Peterson',
    author_email='evanpeterson17@gmail.com',
    description='A scikit-learn based AutoML tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['scikit-learn>=0.21.3', 'pandas>=0.25.3']
)