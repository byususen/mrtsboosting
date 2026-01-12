# 🌾 MRTSBoosting: Multivariate Robust Time Series Boosting

MRTS-Boosting is a quality-aware and computationally efficient time series classification (TSC) framework designed for multivariate, irregular, and unequal-length time series, with a particular focus on satellite-based agricultural monitoring.

The method combines global (full-series) and local (interval-based) time series features with an XGBoost classifier, enabling robust classification under realistic observational conditions such as:

- persistent cloud contamination in optical satellite data,
- temporally misaligned acquisitions from multisensor platforms (e.g., Sentinel-1 SAR vs. Sentinel-2 optical),
- irregular sampling, missing observations, and heterogeneous data quality,
- asynchronous crop growth cycles and staggered planting schedules.

Although originally developed for rice detection using multisensor vegetation indices, MRTS-Boosting is a general-purpose TSC framework applicable to a wide range of multivariate time series classification problems.

## 🔑 Key Features

- Multivariate time series support with unequal lengths and unsynchronized timestamps (e.g., Sentinel-2 optical vs. Sentinel-1 radar observations).
- Quality-aware feature construction, allowing explicit incorporation of observation reliability (e.g., CloudScore+, STMS-derived quality weights, or user-defined metrics).
- Full-series (global) features, including: quality-weighted trend (slope), dominant period and spectral power via Lomb–Scargle periodogram, weighted entropy, weighted lag-1 autocorrelation.
- Interval-based (local) features with adaptive interval selection: weighted quartiles (Q1, median, Q3), weighted interquartile range (IQR), weighted median absolute deviation (MAD), local weighted trend.
- Fast and scalable, suitable for large-area satellite monitoring tasks.
- Flexible input handling, including direct conversion from sktime nested time series format or structured dictionaries.
- Powered by XGBoost.

## 📦 Installation

```bash
pip install mrtsboosting
```

## 🚀 Quick Start Example (UCR / sktime)
This example demonstrates how to train MRTSBoosting on a standard UCR dataset using nested DataFrame → dict conversion, which your class already supports.

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sktime.datasets import load_UCR_UEA_dataset

from mrtsboosting import MRTSBoostingClassifier

# 1) Load UCR dataset (nested DataFrame)
X_train, y_train = load_UCR_UEA_dataset("CBF", split="train", return_X_y=True)
X_test,  y_test  = load_UCR_UEA_dataset("CBF", split="test",  return_X_y=True)

# Encode to 0-indexed labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

# 2) Convert nested sktime → flat dict format
model = MRTSBoostingClassifier(n_jobs=-1)
x_train_flat, y_train_dict = model.from_sktime_nested_uni(X_train, y_train, id_prefix='train')
x_test_flat,  y_test_dict  = model.from_sktime_nested_uni(X_test,  y_test, id_prefix='test')

# 3) Group by sample ID (required by extractor)
x_train = model.preprocess_x_data_dict(x_train_flat)
x_test  = model.preprocess_x_data_dict(x_test_flat)

# 4) Fit and predict
model.fit(x_train, y_train_dict)
y_pred = model.predict(x_test)

# 5) Evaluate
acc = accuracy_score(y_test_dict["label"], y_pred)
kappa = cohen_kappa_score(y_test_dict["label"], y_pred)

print(f"[CBF] Accuracy: {acc:.3f} | Cohen’s κ: {kappa:.3f}")
```

## 🌱 Example Workflow with Satellite Vegetation Indices (NDVI, VH, NDRE, etc.)
MRTSBoosting accepts multivariate inputs as:

```python
x_data_dict = {
  "NDVI": {
      "sample_1": {"time": [...], "value": [...], "weight": [...]},
      "sample_2": {...},
      ...
  },
  "VH": {
      "sample_1": {...},
      ...
  }
}

y_data_dict = {
  "id": ["sample_1", "sample_2", ...],
  "label": ["rice", "non_rice", ...]
}
```

Example usage:
```python
model = MRTSBoostingClassifier(
    random_state=123,
    sampling="adaptive",
    n_jobs=-1
)

model.fit(x_data_dict, y_data_dict)
predicted_labels = model.predict(x_data_dict)

```

## 📜 License

MIT License
