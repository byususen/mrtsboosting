# 🌾 MRTSBoosting: Multivariate Robust Time Series Boosting

MRTS-Boosting is a fast, noise-resistant, and highly flexible time series classification (TSC) framework designed to handle multivariate, irregular, and unequal-length satellite time series.
It combines global (full-series) and local (interval-based) weighted features with an XGBoost classifier, enabling robust performance under difficult conditions such as:

- cloud-contaminated optical observations,
- temporally misaligned radar vs optical acquisition dates,
- irregular sampling, missing values, and varying quality scores,
- asynchronous crop growth phases or planting dates.

Originally designed for rice-field mapping using multisensor satellite vegetation indices, the framework is general and can be applied to any multivariate time-series classification task.

## 🔑 Key Features

- Multivariate support with unequal length and misaligned timestamps (e.g., Sentinel-2 optical vs Sentinel-1 radar time series).
- Quality-aware processing (e.g., CloudScore+ computed from STMS reconstruction or other quality metrics).
- Full-series global features: weighted slope / trend, dominant period + spectral power (Lomb–Scargle), weighted entropy, weighted autocorrelation (lag-1).
- Local interval-based features (adaptive interval selection): weighted median, Q1, Q3, weighted IQR, weighted MAD, local weighted slope. Intervals chosen based on quality weights to avoid cloudy / noisy segments.
- Fast & scalable.
- Flexible input formats: direct conversion from nested time series format to the required input dictionary.
- Powered by XGBoost, enabling: strong performance on noisy data, multiclass classification, powerful built-in regularization.

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
    n_jobs=-1
)

model.fit(x_data_dict, y_data_dict)
predicted_labels = model.predict(x_data_dict)

```

## 📜 License

MIT License
