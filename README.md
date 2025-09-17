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

## Usage Example

```python
from mrtsboosting import MRTSBoosting

model = MRTSBoostingClassifier(n_window=3, window_min=3)
model.fit(x_data_dict_train, y_data_dict_train, time_weight=cloudscore_dict)
y_pred = model.predict(x_data_dict_test)
```

## License

MIT License
