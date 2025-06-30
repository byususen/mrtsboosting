# MRTSBoosting: Multivariate Robust Time Series Boosting

MRTSBoosting is a multivariate time series classification algorithm tailored for periodic data under noisy conditions. It incorporates a time series weighting strategy—such as CloudScore+—to handle data contamination (e.g., clouds in remote sensing) and ensures robust feature extraction. The method is designed to support multisource data with unequal time series lengths and temporal misalignment, which are common in satellite-derived vegetation indices for agricultural monitoring.

MRTSBoosting extracts both:
Global features over the full series and Interval-based weighted features from multiple sensors, and uses XGBoost for high-performance classification.

## Features

- Handles multivariate time series with variable time stamps
- Integrates quality weights for robust feature extraction
- Extracts full series features: weighted slope, period, entropy, power spectrum, and lag 1 autocorrelation
- Extracts interval-based features: weighted median, IQR, MAD, Q1, and Q3
- Supports direct conversion from sktime nested time series format to the required input dictionary structure
- Scalable and parallelized using joblib and Numba
- Powered by XGBoost for high-performance classification

## Installation

```bash
pip install -r requirements.txt
```

## Usage Example

```python
from mrtsboosting import MRTSBoosting

model = MRTSBoostingClassifier(n_window=3, window_min=3)
model.fit(x_data_dict_train, y_data_dict_train, time_weight=cloudscore_dict)
y_pred = model.predict(x_data_dict_test)
```

## Citation

If you use this package in your research, please cite:

Bayu Suseno et al., "Multivariate Robust Time Series Boosting for Remote Sensing-Based Ricefield Classification", 2025.

## License

MIT License
