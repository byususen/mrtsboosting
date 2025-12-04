from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="mrtsboosting",
    version="0.1.0",
    description="MRTSBoosting: Multivariate Robust Time Series Boosting",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Bayu Suseno",
    author_email="bayu.suseno@outlook.com",
    url="https://github.com/byususen/mrtsboosting",
    packages=find_packages(exclude=("experiment",)),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "numba",
        "joblib",
        "astropy",
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
