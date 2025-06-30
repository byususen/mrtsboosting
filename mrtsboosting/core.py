
"""
MRTSBoost - Multivariate Robust Time Series Boosting

This implementation performs robust classification using weighted global and local features
extracted from multivariate satellite time series (e.g., NDVI, VH) with XGBoost.

Author: Bayu Suseno
Date: 2025
"""

import time
from joblib import Parallel, delayed
import numpy as np
from xgboost import XGBClassifier
from astropy.timeseries import LombScargle
from numba import jit
from collections import defaultdict

# ------------------------------
# JIT-optimized weighted metrics
# ------------------------------

@jit(nopython=True, fastmath=True)
def weighted_slope(x, y, w):
    """Compute the weighted slope (trend) of y over x using weights w."""
    if np.sum(w) == 0 or len(x) < 2:
        return 0.0
    w = w / np.sum(w)
    x_mean = np.sum(w * x)
    y_mean = np.sum(w * y)
    return np.sum(w * (x - x_mean) * (y - y_mean)) / np.sum(w * (x - x_mean)**2)

@jit(nopython=True, fastmath=True)
def weighted_mad(x, w):
    """Compute the weighted Median Absolute Deviation (MAD)."""
    median = weighted_percentile(x, w, 50)
    abs_dev = np.abs(x - median)
    return weighted_percentile(abs_dev, w, 50)

@jit(nopython=True, fastmath=True)
def weighted_iqr(x, w):
    """Compute the weighted Interquartile Range (IQR)."""
    q1 = weighted_percentile(x, w, 25)
    q3 = weighted_percentile(x, w, 75)
    return q3 - q1

@jit(nopython=True, fastmath=True)
def weighted_percentile(x, w, q):
    """Compute the q-th weighted percentile of x using weights w."""
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    w_sorted = w[sort_idx]
    total_weight = np.sum(w_sorted)

    cum_weight = 0.0
    target = q / 100.0 * total_weight

    for i in range(len(x_sorted)):
        cum_weight += w_sorted[i]
        if cum_weight >= target:
            return x_sorted[i]
    return x_sorted[-1]

@jit(nopython=True, fastmath=True)
def weighted_autocorr_lag1(x, w):
    """Compute weighted autocorrelation with lag 1."""
    if len(x) < 3:
        return 0.0
    w = w[:len(x)-1]
    x0 = x[:-1]
    x1 = x[1:]
    mean = np.sum(w * x0) / np.sum(w)
    num = np.sum(w * (x0 - mean) * (x1 - mean))
    den = np.sum(w * (x0 - mean)**2)
    return num / den if den > 0 else 0.0

@jit(nopython=True, fastmath=True)
def weighted_entropy(x, w, bins=10):
    """Compute weighted entropy from histogram of values."""
    if len(x) < 3:
        return 0.0
    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max:
        return 0.0

    hist = np.zeros(bins)
    bin_width = (x_max - x_min) / bins
    for i in range(len(x)):
        bin_idx = int((x[i] - x_min) / bin_width)
        if bin_idx == bins:
            bin_idx -= 1
        hist[bin_idx] += w[i]
    p = hist / np.sum(hist)
    entropy = 0.0
    for pi in p:
        if pi > 0:
            entropy -= pi * np.log2(pi)
    return entropy

# ------------------------------
# MRTSBoost Class
# ------------------------------

class MRTSBoostingClassifier:
    """Multivariate Robust Time Series Boosting Class."""

    def __init__(self, n_window=3, window_min=30, window_max=None, min_period=3, max_period=None):
        self.model_name = "MRTS-Boost"
        self.n_window = n_window
        self.window_min = window_min
        self.window_max = window_max
        self.min_period = min_period
        self.max_period = max_period

    def preprocess_x_data_dict(self, x_data_dict):
        """Restructure x_data_dict to be indexed by sample ID."""
        out_dict = {}
        for index_name, data in x_data_dict.items():
            out_dict[index_name] = {}
            ids = np.asarray(data['ids'])
            days = np.asarray(data['days'])
            values = np.asarray(data['values'])
            weights = np.asarray(data['weights'])
            unique_ids = np.unique(ids)
            for uid in unique_ids:
                mask = ids == uid
                out_dict[index_name][uid] = {
                    'days': days[mask],
                    'values': values[mask],
                    'weights': weights[mask]
                }
        return out_dict

    def extract_global_features(self, x_data_dict, days_max):
        """Extract global features (slope, period, entropy, etc.) for each sample."""
        global_feats = {}
        for series_name, series_data in x_data_dict.items():
            for k, data in series_data.items():
                days = np.asarray(data['days'], dtype=np.float64)
                values = np.asarray(data['values'], dtype=np.float64)
                weights = np.asarray(data['weights'], dtype=np.float64)
    
                if len(values) < 3:
                    continue
    
                slope = weighted_slope(days, values, weights)
                period, power = self._get_period(days, values, weights, days_max)
                autocorr = weighted_autocorr_lag1(values, weights)
                entropy = weighted_entropy(values, weights)
    
                if k not in global_feats:
                    global_feats[k] = {}
    
                global_feats[k][f'{series_name}_slope'] = slope
                global_feats[k][f'{series_name}_period'] = period
                global_feats[k][f'{series_name}_period_power'] = power
                global_feats[k][f'{series_name}_autocorr'] = autocorr
                global_feats[k][f'{series_name}_entropy'] = entropy
        return global_feats

    def extract_features(self, x_data_dict, y_data_dict, time_weight=None):
        """Extract both global and local features, weighted by cloudscore if provided."""
        idsamp_unique = np.unique(y_data_dict['ids'])
        
        all_days = np.concatenate([entry['days'] for v in x_data_dict.values() for entry in v.values()])
        days_min, days_max = np.min(all_days), np.max(all_days)
        time_steps = np.arange(days_min, days_max + 1)
        
        # === Fix: use fallback if self.window_max is None
        window_max = self.window_max if self.window_max is not None else days_max - days_min
    
        self.start_all = []
        self.window_all = []
    
        for _ in range(self.n_window):
            win_len = np.random.randint(self.window_min, window_max + 1)
            max_start = days_max - win_len
    
            if time_weight is None:
                # === RANDOM start (no weighting)
                chosen_start = np.random.choice(np.arange(days_min, max_start + 1))
            else:
                # === WEIGHTED interval selection using time_weight
                weights = []
                valid_starts = []
    
                for start in np.arange(days_min, max_start + 1):
                    end = start + win_len
                    total_weight = 0.0
    
                    for sid in idsamp_unique:
                        if sid not in time_weight:
                            continue
                        days_i = x_data_dict[list(x_data_dict.keys())[0]][sid]['days']
                        mask = (days_i >= start) & (days_i < end)
                        total_weight += np.sum(time_weight[sid][mask])
    
                    weights.append(total_weight)
                    valid_starts.append(start)
    
                weights = np.array(weights)
                weights = weights - np.min(weights)  # ensure all values >= 0
                if np.sum(weights) == 0:
                    chosen_start = np.random.choice(valid_starts)
                else:
                    norm_weights = weights / np.sum(weights)
                    chosen_start = np.random.choice(valid_starts, p=norm_weights)
    
            self.start_all.append(chosen_start)
            self.window_all.append(win_len)
    
        # === Global features
        global_feat_dict = self.extract_global_features(x_data_dict, days_max)
    
        # === Local features
        tasks = []
        for j in range(len(self.start_all)):
            start, end = self.start_all[j], self.start_all[j] + self.window_all[j]
            col_name = f'feat_{j+1}'
            for k in idsamp_unique:
                for series_name, series_data in x_data_dict.items():
                    tasks.append((k, series_name, series_data, start, end, col_name))
    
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self._compute_local_features)(*args) for args in tasks
        )
        results = [r for r in results if r is not None]
    
        return self._assemble_features(results, y_data_dict, global_feat_dict)
        
    def extract_predict(self, x_data_dict):
        """Extract features for prediction (no labels required)."""
        idsamp_unique = np.unique(
            np.concatenate([list(series_data.keys()) for series_data in x_data_dict.values()])
        )
        all_days = np.concatenate([entry['days'] for v in x_data_dict.values() for entry in v.values()])
        days_max = np.max(all_days)
    
        global_feat_dict = self.extract_global_features(x_data_dict, days_max)
    
        tasks = []
        for j in range(self.n_window):
            start, end = self.start_all[j], self.start_all[j] + self.window_all[j]
            col_name = f'feat_{j+1}'
            for k in idsamp_unique:
                for series_name, series_data in x_data_dict.items():
                    tasks.append((k, series_name, series_data, start, end, col_name))
    
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self._compute_local_features)(*args) for args in tasks
        )
        results = [r for r in results if r is not None]
    
        return self._assemble_features(results, None, global_feat_dict)

    def fit(self, x_data_dict, y_data_dict, time_weight=None, xgb_params=None):
        """Fit XGBoost classifier to extracted features."""
        if xgb_params is None:
            xgb_params = {
                'objective': 'multi:softprob',
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.9,
                'colsample_bytree': 0.8,
                'n_jobs': -1
            }
        if time_weight == None:
            self.X_new, self.y_new = self.extract_features(x_data_dict, y_data_dict)
        else:
            self.X_new, self.y_new = self.extract_features(x_data_dict, y_data_dict, time_weight = time_weight)
        self.clf = XGBClassifier(**xgb_params)
        self.clf.fit(self.X_new, self.y_new)
        return self

    def predict(self, x_data_dict):
        """Generate predictions using trained model."""
        return self.clf.predict(self.extract_predict(x_data_dict))

    def _compute_local_features(self, k, series_name, series_data, start, end, col_name):
        """Extract interval-based features (e.g., median, IQR, MAD) from selected window."""
        if k not in series_data:
            return None
        data = series_data[k]
        mask = (data['days'] >= start) & (data['days'] <= end)
        if not np.any(mask):
            return None
        x = np.asarray(data['values'][mask], dtype=np.float64)
        w = np.asarray(data['weights'][mask], dtype=np.float64)  # ✅ Correct
    
        if np.sum(w) == 0:
            w = np.ones_like(w)
        else:
            w = w / np.sum(w)
    
        weighted_median = weighted_percentile(x, w, 50)
        iqr_val = weighted_iqr(x, w)
        weighted_q1 = weighted_percentile(x, w, 25)
        weighted_q3 = weighted_percentile(x, w, 75)
        mad_val = weighted_mad(x, w)
    
        return {
            'id': k, 'index_name': series_name, 'feature': col_name,
            'weighted_median': weighted_median,
            'weighted_iqr': iqr_val,
            'weighted_q1': weighted_q1,
            'weighted_q3': weighted_q3,
            'weighted_mad': mad_val
        }
        
    def _assemble_features(self, results, y_data_dict, global_feat_dict):
        """Assemble global and local features into final feature matrix."""
        feature_map = defaultdict(dict)
        for r in results:
            fprefix = f"{r['index_name']}_{r['feature']}"
            feature_map[r['id']][f"{fprefix}_wmedian"] = r['weighted_median']
            feature_map[r['id']][f"{fprefix}_wiqr"] = r['weighted_iqr']
            feature_map[r['id']][f"{fprefix}_wq1"] = r['weighted_q1']
            feature_map[r['id']][f"{fprefix}_wq3"] = r['weighted_q3']
            feature_map[r['id']][f"{fprefix}_wmad"] = r['weighted_mad']

        for k, gfeat in global_feat_dict.items():
            feature_map[k].update(gfeat)

        unique_ids = sorted(feature_map.keys())
        all_keys = sorted(set(k for feats in feature_map.values() for k in feats))
        self.feature_names = all_keys
        X = np.array([[feature_map[uid].get(k, np.nan) for k in all_keys] for uid in unique_ids])

        if y_data_dict is not None:
            y_map = dict(zip(y_data_dict['ids'], y_data_dict['labels']))
            y = np.array([y_map.get(uid, np.nan) for uid in unique_ids])
            mask = ~np.isnan(y)
            return X[mask], y[mask]
        return X

    @staticmethod
    def convert_sktime_mrtsboosting(X_nested, y_array, time_weight_raw=None):
        """
        Convert sktime nested format (UEA/UCR) to MRTSBoosting-compatible dictionaries.
    
        Parameters
        ----------
        X_nested : pd.DataFrame
            Nested DataFrame (n_instances x n_features)
        y_array : array-like
            Class labels
        time_weight_raw : dict[str → np.ndarray] or None
            Optional weight array (n_samples, n_timepoints) per feature
    
        Returns
        -------
        x_data_dict : dict
            Dictionary in MRTSBoosting format
        y_data_dict : dict
            {'ids': ..., 'labels': ...}
        """
        x_data_dict = {}
        n_samples = X_nested.shape[0]
        sample_ids = np.array([f"id_{i}" for i in range(n_samples)])
    
        for i, col in enumerate(X_nested.columns):
            vi_name = f"vi_{i}"
            x_data_dict[vi_name] = {'ids': [], 'days': [], 'values': [], 'weights': []}
    
            for j, series in enumerate(X_nested[col]):
                values = series.values
                length = len(values)
                x_data_dict[vi_name]['ids'].extend([sample_ids[j]] * length)
                x_data_dict[vi_name]['days'].extend(np.arange(length))  # assume day index as 0...T-1
                x_data_dict[vi_name]['values'].extend(values)
    
                if time_weight_raw is not None:
                    weight_matrix = time_weight_raw[vi_name]
                    weights = weight_matrix[j]
                else:
                    weights = np.ones(length)
    
                x_data_dict[vi_name]['weights'].extend(weights)
    
        y_data_dict = {
            'ids': sample_ids,
            'labels': np.array(y_array)
        }
    
        return x_data_dict, y_data_dict
    
    def _get_period(self, days, values, weights, days_max):
        """Estimate dominant period of time series using Lomb-Scargle periodogram."""
        try:
            if len(values) < 4 or np.all(values == values[0]):
                return days_max, 0
    
            weights = weights if weights is not None else np.ones_like(values)
            weights = weights / np.max(weights) if np.max(weights) > 0 else np.ones_like(weights)
            dy = 1 / (weights + 1e-8)
    
            min_period = self.min_period if self.min_period is not None else 3
            max_period = self.max_period if self.max_period is not None else 5 * days_max
            min_freq = 1 / max_period
            max_freq = 1 / min_period
    
            freq, power = LombScargle(days, values, dy=dy).autopower(
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
                samples_per_peak=10
            )
    
            if len(freq) == 0:
                return min_period, 0
    
            best_freq = freq[np.argmax(power)]
            return 1 / best_freq, power[np.argmax(power)]
    
        except Exception:
            return min_period, 0
