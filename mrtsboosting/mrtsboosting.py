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
import pandas as pd
from xgboost import XGBClassifier
from astropy.timeseries import LombScargle
from numba import jit
from collections import defaultdict
from pandas import DataFrame as PandasDataFrame
from sklearn.preprocessing import LabelEncoder

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

    def __init__(self, n_window=None, window_min=None, window_max=None,
             min_period=None, max_period=None, tau=None,
             sampling="adaptive", n_ensemble=5, random_state=None, n_jobs=1):

        self.model_name = "MRTSBoosting"
        self.n_window = n_window
        self.window_min = window_min
        self.window_max = window_max
        self.min_period = min_period
        self.max_period = max_period
        self.tau = tau
    
        self.sampling = sampling
        if self.sampling not in ("adaptive", "random"):
            raise ValueError("sampling must be 'adaptive' or 'random'")
    
        self.n_ensemble = int(n_ensemble)
        if self.n_ensemble < 1:
            raise ValueError("n_ensemble must be >= 1")
    
        self.random_state = random_state
        self.n_jobs = n_jobs


    def _interval_sampling_weights(self, w):
        """
        Return weights used for *interval selection/scoring*.
        - adaptive: real weights
        - random: all ones (same shape)
        """
        if (w is None) or (self.sampling == "adaptive"):
            return w
        return np.ones_like(w, dtype=np.float32)

    def preprocess_x_data_dict(self, x_data_dict):
        """
        Ensure x_data_dict is grouped by sample ID per variable.
        
        Converts input format:
        {
            'vi_name': {
                'id': [...], 'time': [...], 'value': [...], 'weight': [...]
            }
        }
    
        Into:
        {
            'vi_name': {
                sample_id: {
                    'time': [...], 'value': [...], 'weight': [...]
                }
            }
        }
        """
        out_dict = {}
        for vi_name, data in x_data_dict.items():
            out_dict[vi_name] = {}
            ids = np.asarray(data['id'])
            times = np.asarray(data['time'])
            values = np.asarray(data['value'])
            weights = np.asarray(data['weight'])
    
            unique_ids = np.unique(ids)
            for uid in unique_ids:
                mask = ids == uid
                out_dict[vi_name][uid] = {
                    'time': times[mask],
                    'value': values[mask],
                    'weight': weights[mask]
                }
        
        return out_dict


    def extract_global_features(self, x_data_dict, time_max):
        """Extract global features (slope, period, entropy, etc.) for each sample."""
        global_feats = {}
    
        for series_name, series_data in x_data_dict.items():
            for sample_id, data in series_data.items():
                time = np.asarray(data['time'], dtype=np.float64)
                value = np.asarray(data['value'], dtype=np.float64)
                weight = np.asarray(data['weight'], dtype=np.float64)
    
                if len(value) < 3:
                    continue
    
                try:
                    slope = weighted_slope(time, value, weight)
                except Exception:
                    slope = np.nan
    
                try:
                    period, power = self._get_period(time, value, weight, time_max)
                except Exception:
                    period, power = np.nan, np.nan
    
                try:
                    autocorr = weighted_autocorr_lag1(value, weight)
                except Exception:
                    autocorr = np.nan
    
                try:
                    entropy = weighted_entropy(value, weight)
                except Exception:
                    entropy = np.nan
    
                if sample_id not in global_feats:
                    global_feats[sample_id] = {}
    
                global_feats[sample_id][f'{series_name}_slope'] = slope
                global_feats[sample_id][f'{series_name}_period'] = period
                global_feats[sample_id][f'{series_name}_period_power'] = power
                global_feats[sample_id][f'{series_name}_autocorr'] = autocorr
                global_feats[sample_id][f'{series_name}_entropy'] = entropy
    
        return global_feats


    def extract_features(self, x_data_dict, y_data_dict):
        """
        Extracts interval-based (local) and full-series (global) features from multivariate time series data.
    
        This method identifies a set of informative intervals across all series based on quality weights 
        and extracts features within each selected interval. Additionally, global (full-series) features 
        are computed for each sample and variable.
    
        Parameters:
        ----------
        x_data_dict : dict
            A nested dictionary structured as {vi_name: {sample_id: {'time', 'value', 'weight'}}}, 
            containing multivariate time series and associated quality weights.
        
        y_data_dict : dict
            A dictionary with 'id' (sample IDs) and 'label' (class labels), where each ID corresponds 
            to a unique time series sample.
    
        Returns:
        -------
        Pandas DataFrame
            A combined feature matrix with interval-based and global features for all samples,
            ready for classifier input.
        """
        idsamp_unique = np.unique(y_data_dict['id'])

        series_lengths = [len(s['time']) for vi_data in x_data_dict.values() for s in vi_data.values()]
        median_len = np.median(series_lengths)
        
        if self.n_window is None:
            self.n_window = int(np.sqrt(median_len))
            print(f"[INFO] Auto-setting n_window = {self.n_window}")
        
        if self.window_min is None:
            self.window_min = max(3, int(median_len * 0.1))
            print(f"[INFO] Auto-setting window_min = {self.window_min}")
        
        if self.min_period is None:
            self.min_period = max(3, int(median_len * 0.1))
            print(f"[INFO] Auto-setting min_period = {self.min_period}")
    
        all_times = np.concatenate([entry['time'] for v in x_data_dict.values() for entry in v.values()])
        time_min, time_max = np.min(all_times), np.max(all_times)
        window_max = self.window_max if self.window_max is not None else time_max - time_min
    
        self.start_all = []
        self.window_all = []
        all_results = []
    
        # STEP 1 — Identify VIs with fluctuating weights
        vi_weight_fluctuation = {}
        for vi_name, series_data in x_data_dict.items():
            all_weights = []
            for sid in idsamp_unique:
                if sid in series_data:
                    all_weights.extend(series_data[sid]['weight'].tolist())
            std_dev = np.std(all_weights) if len(all_weights) > 0 else 0
            vi_weight_fluctuation[vi_name] = 1 if std_dev > 1e-6 else 0
        
        if self.tau is None:
            all_weights = []
            for vi_name, series_data in x_data_dict.items():
                if vi_weight_fluctuation.get(vi_name, 0) == 1:
                    for sid in idsamp_unique:
                        if sid in series_data:
                            all_weights.extend(series_data[sid]['weight'].tolist())
            mean_w = np.mean(all_weights) if len(all_weights) > 0 else 1.0
            self.tau = 0.05 * mean_w  
            print(f"[INFO] Auto-setting tau = {self.tau:.4f} (5% of mean fluctuating weight)")

        # STEP 2 — Compute global average weight using only fluctuating VIs
        all_weights = []
        for vi_name, series_data in x_data_dict.items():
            if vi_weight_fluctuation.get(vi_name, 0) == 1:
                for sid in idsamp_unique:
                    if sid in series_data:
                        all_weights.extend(series_data[sid]['weight'].tolist())
        prev_mean_weight = np.mean(all_weights) if len(all_weights) > 0 else 0.0
    
        # STEP 3 — Select interval lengths
        possible_lengths = np.arange(self.window_min, window_max)
        n_unique_lengths = min(self.n_window, len(possible_lengths))
        interval_lengths = self.rng.choice(possible_lengths, size=n_unique_lengths, replace=False)
        interval_lengths = sorted(interval_lengths, reverse=True)
    
        # STEP 4 — For each interval length, select start location adaptively
        for i, win_len in enumerate(interval_lengths):
            while True:
                max_start = time_max - win_len
                start = self.rng.integers(time_min, max_start + 1)
                end = start + win_len
                has_data = False
                weights_all = []
    
                for vi_name, series_data in x_data_dict.items():
                    if (any(vi_weight_fluctuation.values()) and vi_weight_fluctuation.get(vi_name, 0) == 1) or \
                       (not any(vi_weight_fluctuation.values())):
                        for sid in idsamp_unique:
                            if sid in series_data:
                                times = series_data[sid]['time']
                                weights = series_data[sid]['weight']
                                w_sel = self._interval_sampling_weights(weights)
                                mask = (times >= start) & (times < end)
                                if np.any(mask):
                                    has_data = True
                                    weights_all.extend(w_sel[mask].tolist())
   
                if not has_data:
                    continue
    
                mean_weight = np.mean(weights_all) if len(weights_all) > 0 else 0.0
    
                if mean_weight >= prev_mean_weight:
                    break
                else:
                    prob_accept = np.exp((mean_weight - prev_mean_weight) / self.tau)
                    if self.rng.random() < prob_accept:
                        break
                    else:
                        continue
    
            self.start_all.append(start)
            self.window_all.append(win_len)
    
            col_name = f'feat_{i+1}'
            for series_name, series_data in x_data_dict.items():
                for sid in idsamp_unique:
                    all_results.append((sid, series_name, series_data, start, end, col_name))
    
            prev_mean_weight = mean_weight
    
        # STEP 5 — Global features
        global_feat_dict = self.extract_global_features(x_data_dict, time_max)
    
        # STEP 6 — Parallel local feature extraction
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._compute_local_features)(*args) for args in all_results
        )
        results = [r for r in results if r is not None]
    
        return self._assemble_features(results, y_data_dict, global_feat_dict)

        
    def extract_predict(self, x_data_dict, feature_names=None):
        """Extract prediction features using shared intervals across all series."""
    
        idsamp_unique = np.unique(
            np.concatenate([list(series_data.keys()) for series_data in x_data_dict.values()])
        )
    
        all_times = np.concatenate([entry['time'] for v in x_data_dict.values() for entry in v.values()])
        time_max = np.max(all_times)
    
        global_feat_dict = self.extract_global_features(x_data_dict, time_max)
    
        tasks = []
        for j in range(len(self.start_all)):
            start = self.start_all[j]
            end = start + self.window_all[j]
            col_name = f'feat_{j+1}'
    
            for series_name, series_data in x_data_dict.items():
                for id_val in series_data.keys():
                    tasks.append((id_val, series_name, series_data, start, end, col_name))
    
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._compute_local_features)(*args) for args in tasks
        )
        results = [r for r in results if r is not None]
    
        return self._assemble_features(results, None, global_feat_dict, feature_names=feature_names)


    def fit(self, X, y=None, xgb_params=None):
        """Fit the MRTS-Boost model. Accepts nested DataFrame, 3D NumPy array, or dict format."""
    
        if isinstance(X, np.ndarray) and X.ndim == 3:
            x_data_dict, y_data_dict = self._convert_3d_numpy_to_dict(X, y)
            x_data_dict = self.preprocess_x_data_dict(x_data_dict)
    
        elif isinstance(X, pd.DataFrame):
            x_data_dict, y_data_dict = self.from_sktime_nested(X, y)
    
        else:
            x_data_dict, y_data_dict = X, y
    
        self.label_encoder = LabelEncoder()
        y_data_dict['label'] = self.label_encoder.fit_transform(y_data_dict['label'])
        num_classes = len(np.unique(y_data_dict['label']))
    
        # Default XGBoost params
        if xgb_params is None:
            xgb_params = {
                'objective': 'multi:softmax',
                'num_class': num_classes,
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.9,
                'colsample_bytree': 0.8,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state
            }
    
        self.clfs_ = []
        self.intervals_ = []         # list of (start_all, window_all)
        self.feature_names_list_ = []  # list of feature-name lists
    
        base_seed = 0 if self.random_state is None else int(self.random_state)
    
        for e in range(self.n_ensemble):
            self.rng = np.random.default_rng(base_seed + e)
    
            X_new, y_new = self.extract_features(x_data_dict, y_data_dict)
    
            params_e = dict(xgb_params)
            params_e['random_state'] = (base_seed + e) if params_e.get('random_state', None) is not None else (base_seed + e)
    
            clf = XGBClassifier(**params_e)
            clf.fit(X_new, y_new)
    
            self.clfs_.append(clf)
            self.intervals_.append((list(self.start_all), list(self.window_all)))
            self.feature_names_list_.append(list(self.feature_names))
    
        self.clf = self.clfs_[0]
        return self

    
    def predict(self, X):
        """Predict using nested DataFrame, 3D NumPy array, or already grouped dict format."""
    
        if isinstance(X, np.ndarray) and X.ndim == 3:
            x_data_dict, _ = self._convert_3d_numpy_to_dict(X, y_array=np.zeros(X.shape[0]))
            x_data_dict = self.preprocess_x_data_dict(x_data_dict)
    
        elif isinstance(X, pd.DataFrame):
            x_data_dict, _ = self.from_sktime_nested(X, y_array=np.zeros(X.shape[0]))
    
        else:
            x_data_dict = X
    
        if not hasattr(self, "clfs_") or len(self.clfs_) == 0:
            X_features = self.extract_predict(x_data_dict)
            y_pred_encoded = self.clf.predict(X_features)
            return self.label_encoder.inverse_transform(y_pred_encoded)
    
        preds = []
        for clf, (starts, windows), feat_names in zip(self.clfs_, self.intervals_, self.feature_names_list_):
            # use this member's intervals
            self.start_all = starts
            self.window_all = windows
    
            X_features = self.extract_predict(x_data_dict, feature_names=feat_names)
            preds.append(clf.predict(X_features))
    
        preds = np.vstack(preds)  # shape: (k, n_samples)
    
        # Majority vote (multi-class)
        n_samples = preds.shape[1]
        y_vote = np.empty(n_samples, dtype=int)
        for j in range(n_samples):
            counts = np.bincount(preds[:, j].astype(int))
            y_vote[j] = np.argmax(counts)
    
        return self.label_encoder.inverse_transform(y_vote)

    
    def _compute_local_features(self, sample_id, series_name, series_data, start, end, col_name):
        """Extract interval-based features including weighted stats and slope."""
        
        data = series_data[sample_id]
        mask = (data['time'] >= start) & (data['time'] <= end)
    
        if not np.any(mask):
            return None
    
        time = np.asarray(data['time'][mask], dtype=np.float64)
        x = np.asarray(data['value'][mask], dtype=np.float64)
        w = np.asarray(data['weight'][mask], dtype=np.float64)
    
        if np.sum(w) == 0:
            w = np.ones_like(w)
        else:
            w = w / np.sum(w)
    
        try:
            weighted_mean_val = np.sum(w * x)
            weighted_std_val = np.sqrt(np.sum(w * (x - weighted_mean_val)**2))
        except Exception:
            weighted_mean_val = np.nan
            weighted_std_val = np.nan
    
        try:
            weighted_median = weighted_percentile(x, w, 50)
        except Exception:
            weighted_median = np.nan
    
        try:
            iqr_val = weighted_iqr(x, w)
        except Exception:
            iqr_val = np.nan
    
        try:
            weighted_q1 = weighted_percentile(x, w, 25)
            weighted_q3 = weighted_percentile(x, w, 75)
            mad_val = weighted_mad(x, w)
        except Exception:
            weighted_q1, weighted_q3, mad_val = np.nan, np.nan, np.nan
    
        try:
            slope_val = weighted_slope(time, x, w)
        except Exception:
            slope_val = np.nan
    
        return {
            'id': sample_id,
            'index_name': series_name,
            'feature': col_name,
            'weighted_mean': weighted_mean_val,
            'weighted_std': weighted_std_val,
            'weighted_median': weighted_median,
            'weighted_iqr': iqr_val,
            'weighted_q1': weighted_q1,
            'weighted_q3': weighted_q3,
            'weighted_mad': mad_val,
            'weighted_slope': slope_val
        }

    
    def _assemble_features(self, results, y_data_dict, global_feat_dict, feature_names=None):
        """Assemble global and local features into final feature matrix."""
    
        feature_map = defaultdict(dict)
    
        # === Collect local features
        for r in results:
            prefix = f"{r['index_name']}_{r['feature']}"
            sample_id = r['id']
            feature_map[sample_id][f"{prefix}_wmedian"] = r['weighted_median']
            feature_map[sample_id][f"{prefix}_wmean"] = r['weighted_mean']
            feature_map[sample_id][f"{prefix}_wstd"] = r['weighted_std']
            feature_map[sample_id][f"{prefix}_wiqr"] = r['weighted_iqr']
            feature_map[sample_id][f"{prefix}_wq1"] = r['weighted_q1']
            feature_map[sample_id][f"{prefix}_wq3"] = r['weighted_q3']
            feature_map[sample_id][f"{prefix}_wmad"] = r['weighted_mad']
            feature_map[sample_id][f"{prefix}_wslope"] = r['weighted_slope']
    
        # === Append global features
        for sample_id, global_feats in global_feat_dict.items():
            feature_map[sample_id].update(global_feats)
    
        # === Final list of sample IDs and features
        unique_ids = sorted(feature_map.keys())
    
        if feature_names is None:
            all_feature_names = sorted({feat for feats in feature_map.values() for feat in feats})
        else:
            all_feature_names = list(feature_names)
    
        self.feature_names = all_feature_names
    
        # === Build feature matrix X
        X = np.array([
            [feature_map[sid].get(fname, np.nan) for fname in all_feature_names]
            for sid in unique_ids
        ])
    
        # === Build label vector y if available
        if y_data_dict is not None:
            y_lookup = dict(zip(y_data_dict['id'], y_data_dict['label']))
            y = np.array([y_lookup.get(sid, np.nan) for sid in unique_ids])
            valid_mask = ~np.isnan(y)
            return X[valid_mask], y[valid_mask]
    
        return X
    
    def _convert_3d_numpy_to_dict(self, X_3d, y_array):
        """
        Convert a 3D NumPy array (n_instances × n_channels × n_timepoints)
        into a dictionary structure compatible with MRTSBoosting.
        """
        n_instances, n_channels, n_timepoints = X_3d.shape
        sample_ids = [f"id_{i}" for i in range(n_instances)]
    
        x_data_dict = {}
    
        for ch in range(n_channels):
            vi_name = f"vi_{ch}"
            x_data_dict[vi_name] = {}
    
            for i in range(n_instances):
                sid = sample_ids[i]
                x_data_dict[vi_name][sid] = {
                    'time': np.arange(n_timepoints, dtype=np.float64),
                    'value': X_3d[i, ch, :].astype(np.float64),
                    'weight': np.ones(n_timepoints, dtype=np.float64)
                }
    
        y_data_dict = {
            'id': sample_ids,
            'label': np.array(y_array)
        }
    
        return x_data_dict, y_data_dict


    def _group_by_sample(self, flat_dict):
        """
        Convert flat x_data_dict[vi] with keys 'id', 'time', 'value', 'weight'
        into a grouped structure indexed by sample ID.
        """
        
        grouped = defaultdict(lambda: {'time': [], 'value': [], 'weight': []})
    
        for i in range(len(flat_dict['id'])):
            sid = flat_dict['id'][i]
            grouped[sid]['time'].append(flat_dict['time'][i])
            grouped[sid]['value'].append(flat_dict['value'][i])
            grouped[sid]['weight'].append(flat_dict['weight'][i])
    
        return {
            sid: {k: np.asarray(v) for k, v in series.items()}
            for sid, series in grouped.items()
        }


    @staticmethod
    def from_sktime_nested_uni(X_nested, y_array, id_prefix="id_", weight=None):
        
        # 1) Check all series lengths (should all match)
        lens = {len(s) for s in X_nested.iloc[:, 0]}
        
        # 2) Stack into a 2D float array
        X_2d = np.vstack(
            X_nested.iloc[:, 0].apply(lambda s: np.asarray(s, dtype=float)).to_numpy()
        )

        le = LabelEncoder()
        y_array = le.fit_transform(y_array)
        data = X_2d

        n_samples, n_timestamps = data.shape
        id_samples = [f"{id_prefix}{i+1}" for i in range(n_samples)]
        id_times = np.arange(1, n_timestamps + 1)
        
        df = pd.DataFrame(data, index=id_samples, columns=id_times)
        
        X_df = df.reset_index().melt(id_vars='index', var_name='id_time', value_name='id_value')
        X_df = X_df.rename(columns={'index': 'id_sample'})
        X_df = X_df.sort_values(by=['id_sample', 'id_time'])

        if weight is not None:
            weights_df = pd.DataFrame(weight, index=id_samples, columns=id_times)
            weights_long = weights_df.reset_index().melt(id_vars='index', var_name='id_time', value_name='weight')
            weights_long = weights_long.rename(columns={'index': 'id_sample'})
            weights_long = weights_long.sort_values(by=['id_sample', 'id_time'])
            X_df['weight'] = weights_long['weight'].values
        else:
            X_df['weight'] = 1.0
        
        x_data_dict = {
            'signal': {
                'id': X_df['id_sample'].values,
                'time': X_df['id_time'].values,
                'value': X_df['id_value'].values,
                'weight': X_df['weight'].values
            }
        }

        labels = y_array
        
        id_samples = [f"{id_prefix}{i+1}" for i in range(len(labels))]
        
        y_df = pd.DataFrame({
            'id_sample': id_samples,
            'label': labels
        })
        
        y_df = y_df.sort_values(by=['id_sample'])
        y_data_dict = {
            'id': y_df['id_sample'].values,
            'label': y_df['label'].values
        }
        return x_data_dict, y_data_dict
    

    def _get_period(self, time, value, weight, time_max):
        """
        Estimate the dominant period of a weighted time series using Lomb-Scargle periodogram.
    
        Parameters
        ----------
        time : np.ndarray
            Time points of the series.
        value : np.ndarray
            Observed values.
        weight : np.ndarray
            Observation weights.
        time_max : float
            Maximum time used to bound the frequency search.
    
        Returns
        -------
        best_period : float
            Dominant period detected.
        best_power : float
            Associated power of the dominant frequency.
        """
        try:
            if len(value) < 4 or np.all(value == value[0]):
                return time_max, 0.0
    
            weight = np.asarray(weight)
            weight = weight / np.max(weight) if np.max(weight) > 0 else np.ones_like(weight)
            dy = 1 / (weight + 1e-8)
    
            min_period = self.min_period
            max_period = self.max_period if self.max_period is not None else 5 * time_max
            min_freq = 1 / max_period
            max_freq = 1 / min_period
    
            freq, power = LombScargle(time, value, dy=dy).autopower(
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
                samples_per_peak=10
            )
    
            if len(freq) == 0 or np.all(np.isnan(power)):
                return min_period, 0.0
    
            best_idx = np.nanargmax(power)
            best_freq = freq[best_idx]
            best_power = power[best_idx]
    
            best_period = 1 / best_freq if best_freq > 0 else min_period
            if not np.isfinite(best_period) or best_period < min_period:
                best_period = min_period
    
            return best_period, best_power if np.isfinite(best_power) else 0.0
    
        except Exception as e:
            print(f"[WARN] _get_period exception: {e}")
            return self.min_period, 0.0
