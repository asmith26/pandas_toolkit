import os

from setuptools import setup

import pandas_toolkit

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pandas_toolkit",
    version=pandas_toolkit.__version__,
    description="A collection of pandas accessors to help with common machine learning related functionality.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="asmith26",
    url="https://github.com/asmith26/pandas_toolkit.git",
    license="Apache-2.0",
    packages=["pandas_toolkit"],
    install_requires=["pandas"],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
