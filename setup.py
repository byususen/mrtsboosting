from setuptools import setup, find_packages

setup(
    name='mrtsboosting',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 'pandas', 'xgboost', 'scikit-learn', 'numba', 'astropy', 'joblib'
    ],
    author='Bayu Suseno',
    description='Multivariate Robust Time Series Boosting for cloud-prone remote sensing classification',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)