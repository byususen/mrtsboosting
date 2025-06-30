# MRTS-Boost: Multivariate Robust Time Series Boosting

MRTS-Boost is a robust and flexible time series classification algorithm designed for satellite-derived vegetation indices, particularly in cloud-prone regions. It integrates global and interval-based weighted features (e.g., slope, MAD, entropy, periodicity) from multiple sources (e.g., NDVI, RVI, VH) using XGBoost.

## Features

- Handles multivariate time series with variable time stamps
- Integrates CloudScore+ quality weights for robust feature extraction
- Extracts global features: slope, period, autocorrelation, entropy
- Extracts interval-based features: weighted median, IQR, MAD
- Scalable and parallelized using joblib and Numba
- Powered by XGBoost for high-performance classification

## Installation

```bash
pip install -r requirements.txt
```

## Usage Example

```python
from mrtsboosting import MRTSBoost

model = MRTSBoost(n_window=3)
model.fit(x_data_dict_train, y_data_dict_train, time_weight=cloudscore_dict)
y_pred = model.predict(x_data_dict_test)
```

## Citation

If you use this package in your research, please cite:

Bayu Suseno et al., "Multivariate Robust Time Series Boosting for Remote Sensing-Based Ricefield Classification", 2025.

## License

MIT License