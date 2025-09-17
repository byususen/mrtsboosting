# MRTSBoosting: Multivariate Robust Time Series Boosting

MRTS-Boosting is a fast and robust time series classification (TSC) framework designed for noisy and temporally irregular data. It combines full-series and interval-based feature extraction with an XGBoost ensemble classifier, enabling accurate classification under challenging conditions such as cloud contamination and variable planting schedules.

The method is tailored for multisensor satellite data, including optical and radar vegetation indices (VIs), which often differ in acquisition frequency and temporal alignment. By treating each VI as an independent series on its own temporal grid, MRTS-Boosting avoids the need for resampling while fully exploiting complementary information.

## Key Features

- Handles multivariate, misaligned, and unequal-length time series (e.g., Sentinel-1 radar and Sentinel-2 optical VIs).
- Quality-aware feature extraction using observation weights (e.g., CloudScore+) to mitigate noise such as cloud contamination.
- Full-series features: weighted slope, dominant period and spectral power (Lomb–Scargle), entropy, and weighted lag-1 autocorrelation.
- Interval-based features: weighted quartiles (Q1, median, Q3), IQR, MAD, and local slope, adaptively selected based on observation quality.
- Scalable and efficient, with parallelized feature extraction using joblib and numba.
- Seamless integration with sktime: direct conversion from nested time series format to the required input dictionary.
- Powered by XGBoost for high-performance, regularized classification.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/byususen/mrtsboosting.git
cd mrtsboosting
pip install -r requirements.txt
```

## Usage Example

```python
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sktime.datasets import load_UCR_UEA_dataset

#from mrtsboosting import MRTSBoostingClassifier

# 1) Load UCR dataset (univariate, nested DataFrame)
X_train, y_train = load_UCR_UEA_dataset("CBF", split="train", return_X_y=True)
X_test,  y_test  = load_UCR_UEA_dataset("CBF", split="test",  return_X_y=True)

# Ensure class labels are 0-indexed
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# 2) Convert nested sktime format -> flat dicts (per your class helper)
model = MRTSBoostingClassifier()
x_train_flat, y_train_dict = model.from_sktime_nested_uni(X_train, y_train, id_prefix='train')
x_test_flat,  y_test_dict  = model.from_sktime_nested_uni(X_test,  y_test, id_prefix='test')

# 3) Group by sample id (what extract_features expects)
x_train = model.preprocess_x_data_dict(x_train_flat)
x_test  = model.preprocess_x_data_dict(x_test_flat)

# 4) Fit & predict
model.fit(x_train, y_train_dict)
y_pred = model.predict(x_test)

# 5) Evaluate
acc = accuracy_score(y_test_dict["label"], y_pred)
kappa = cohen_kappa_score(y_test_dict["label"], y_pred)
print(f"[CBF] Accuracy: {acc:.3f} | Cohen’s κ: {kappa:.3f}")
```

## License

MIT License
