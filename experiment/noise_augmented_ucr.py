#OK NOISED

import random

import os
SEED = 1
np.random.seed(SEED)
random.seed(SEED)


import time
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.interval_based import DrCIF as DrCIFClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.hybrid import HIVECOTEV2
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from pyts.classification import TimeSeriesForest
from sktime.datatypes import convert_to, convert
from sklearn.ensemble import RandomForestClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.datasets import list_datasets


# === Paths and Globals ===
run_mode = "all"
# List all UCR/UEA datasets available online
dataset_list = list_datasets()
skipped_models = []
failed_datasets = []
results = []

def load_ucr_data(dataset_name, noise_prop=0.5):
    # Load raw data
    # 1) Load UCR dataset (univariate, nested DataFrame)
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
    X_test,  y_test  = load_UCR_UEA_dataset(dataset_name, split="test",  return_X_y=True)
    
    # Ensure class labels are 0-indexed
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # Combine to compute global min for noise baseline
    all_data = np.concatenate([X_train, X_test])
    global_min = np.min(all_data)

    def add_random_noise_and_weights(X):
        np.random.seed(SEED)
        n_samples, n_timestamps = X.shape
        weights = np.ones_like(X, dtype=np.float64)

        for i in range(n_samples):
            series = X[i].astype(np.float64)
            sigma = np.std(series)

            # Randomly select 50% indices
            noisy_indices = np.random.choice(
                n_timestamps,
                size=int(noise_prop * n_timestamps),
                replace=False
            )

            # Generate noise
            noise = global_min + np.random.normal(0, sigma, len(noisy_indices))

            # Apply noise
            X[i, noisy_indices] = noise

            # Assign low weights to noisy points
            weights[i, noisy_indices] = np.random.uniform(0.01, 0.1, len(noisy_indices))

        return X, weights

    # Apply random noise to train and test
    X_train_noisy, W_train = add_random_noise_and_weights(X_train.copy())
    X_test_noisy, W_test = add_random_noise_and_weights(X_test.copy())

    return X_train_noisy, y_train, W_train, X_test_noisy, y_test, W_test


def log_result(model_name, y_true, y_pred, duration):
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"[RESULT] {model_name} | Accuracy: {acc:.4f} | Kappa: {kappa:.4f} | Duration: {duration:.2f} sec")

def run_all_models(X_train, y_train, X_test, y_test, dataset_name, W_train=None, W_test=None):
    global results

    print(f"\n[INFO] Running models for dataset: {dataset_name}")
    try:
        data = X_train

        # Generate IDs
        n_samples, n_timestamps = data.shape
        id_samples = [f"train_{i+1}" for i in range(n_samples)]
        id_times = np.arange(1, n_timestamps + 1)
        
        # Create the DataFrame
        df = pd.DataFrame(data, index=id_samples, columns=id_times)
        
        # Melt to long format
        X_train_df = df.reset_index().melt(id_vars='index', var_name='id_time', value_name='id_value')
        X_train_df = X_train_df.rename(columns={'index': 'id_sample'})
        # Sort by id_sample and id_time
        X_train_df = X_train_df.sort_values(by=['id_sample', 'id_time'])

        # Reset the index
        X_train_arr_df = X_train_df.reset_index(drop=True)
        
        X_train_arr_df.set_index(['id_sample', 'id_time'], inplace=True)
        
        X_train_arr = convert_to(X_train_arr_df, to_type="numpy3D")

        # Add weights if provided (otherwise default to 1)
        if W_train is not None:
            weights_train_df = pd.DataFrame(W_train, index=id_samples, columns=id_times)
            weights_long = weights_train_df.reset_index().melt(id_vars='index', var_name='id_time', value_name='weight')
            weights_long = weights_long.rename(columns={'index': 'id_sample'})
            weights_long = weights_long.sort_values(by=['id_sample', 'id_time'])
            X_train_df['weight'] = weights_long['weight'].values
        else:
            X_train_df['weight'] = 1.0
        
        x_data_dict_train = {
            'signal': {
                'id': X_train_df['id_sample'].values,
                'time': X_train_df['id_time'].values,
                'value': X_train_df['id_value'].values,
                'weight': X_train_df['weight'].values
            }
        }
                
        data = X_test

        # Generate IDs
        n_samples, n_timestamps = data.shape
        id_samples = [f"test_{i+1}" for i in range(n_samples)]
        id_times = np.arange(1, n_timestamps + 1)
        
        # Create the DataFrame
        df = pd.DataFrame(data, index=id_samples, columns=id_times)
        
        # Melt to long format
        X_test_df = df.reset_index().melt(id_vars='index', var_name='id_time', value_name='id_value')
        X_test_df = X_test_df.rename(columns={'index': 'id_sample'})
        # Sort by id_sample and id_time
        X_test_df = X_test_df.sort_values(by=['id_sample', 'id_time'])

        # Reset the index
        X_test_arr_df = X_test_df.reset_index(drop=True)
        
        X_test_arr_df.set_index(['id_sample', 'id_time'], inplace=True)
        
        X_test_arr = convert_to(X_test_arr_df, to_type="numpy3D")

        # Add weights if provided
        if W_test is not None:
            weights_test_df = pd.DataFrame(W_test, index=id_samples, columns=id_times)
            weights_long = weights_test_df.reset_index().melt(id_vars='index', var_name='id_time', value_name='weight')
            weights_long = weights_long.rename(columns={'index': 'id_sample'})
            weights_long = weights_long.sort_values(by=['id_sample', 'id_time'])
            X_test_df['weight'] = weights_long['weight'].values
        else:
            X_test_df['weight'] = 1.0
            
        x_data_dict_test = {
            'signal': {
                'id': X_test_df['id_sample'].values,
                'time': X_test_df['id_time'].values,
                'value': X_test_df['id_value'].values,
                'weight': X_test_df['weight'].values
            }
        }
        
        
        
        # === Example label array ===
        labels = y_train
        
        # === Create corresponding id_sample ===
        id_samples = [f"train_{i+1}" for i in range(len(labels))]
        
        # === Create DataFrame ===
        y_train_df = pd.DataFrame({
            'id_sample': id_samples,
            'label': labels
        })
        
        # Sort by id_sample
        y_train_df = y_train_df.sort_values(by=['id_sample'])
        y_data_dict_train = {
            'id': y_train_df['id_sample'].values,
            'label': y_train_df['label'].values
        }
        
        y_train_arr = np.asarray(y_train_df.label)

        # === Example label array ===
        labels = y_test
        
        # === Create corresponding id_sample ===
        id_samples = [f"test_{i+1}" for i in range(len(labels))]
        
        # === Create DataFrame ===
        y_test_df = pd.DataFrame({
            'id_sample': id_samples,
            'label': labels
        })
        
        # Sort by id_sample
        y_test_df = y_test_df.sort_values(by=['id_sample'])
        y_data_dict_test = {
            'id': y_test_df['id_sample'].values,
            'label': y_test_df['label'].values
        }
        
        y_test_arr = np.asarray(y_test_df.label)

               
    except Exception as e:
        print(f"[ERROR] Preprocessing failed for {dataset_name}: {e}")
        return

    # --- TSF ---
    try:
        print(f"[INFO] Running TSF on {dataset_name}")
        clf = TimeSeriesForest(random_state=SEED)
        start = time.time()
        clf.fit(X_train_arr.reshape(X_train_arr.shape[0], -1), y_train_arr)
        y_pred = clf.predict(X_test_arr.reshape(X_test_arr.shape[0], -1))
        duration = time.time() - start
        results.append((dataset_name, "TSF", cohen_kappa_score(y_test_arr, y_pred), accuracy_score(y_test_arr, y_pred), duration))
        log_result("TSF", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] TSF failed on {dataset_name}: {e}")

    # --- MRTSBoost ---
    try:
        print(f"[INFO] Running MRTSBoosting on {dataset_name}")
        start = time.time()  # Add this line to start timing
        model = MRTSBoostingClassifier(random_state=SEED)
        model.fit(model.preprocess_x_data_dict(x_data_dict_train), y_data_dict_train)
        y_pred = model.predict(model.preprocess_x_data_dict(x_data_dict_test))
        duration = time.time() - start
        results.append((dataset_name, "MRTSBoost", cohen_kappa_score(y_data_dict_test['label'], y_pred), accuracy_score(y_data_dict_test['label'], y_pred), duration))
        log_result("MRTS-Boosting", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] MRTSBoost failed on {dataset_name}: {e}")

    
    # --- ROCKET ---
    try:
        print(f"[INFO] Running ROCKET on {dataset_name}")
        clf = RocketClassifier(random_state=SEED)
        start = time.time()
        clf.fit(X_train_arr.reshape(X_train_arr.shape[0], -1), y_train_arr)
        y_pred = clf.predict(X_test_arr.reshape(X_test_arr.shape[0], -1))
        duration = time.time() - start
        results.append((dataset_name, "ROCKET", cohen_kappa_score(y_test_arr, y_pred), accuracy_score(y_test_arr, y_pred), duration))
        log_result("ROCKET", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] ROCKET failed on {dataset_name}: {e}")

    # --- Shapelet ---
    try:
        print(f"[INFO] Running Shapelet on {dataset_name}")
        clf = ShapeletTransformClassifier(random_state=SEED)
    
        start = time.time()
        clf.fit(X_train_arr.reshape(X_train_arr.shape[0], -1), y_train_arr)
        y_pred = clf.predict(X_test_arr.reshape(X_test_arr.shape[0], -1))
        duration = time.time() - start
        results.append((dataset_name, "Shapelet", cohen_kappa_score(y_test_arr, y_pred), accuracy_score(y_test_arr, y_pred), duration))
        log_result("ST", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] Shapelet failed on {dataset_name}: {e}")

    # --- DTW ---
    try:
        print(f"[INFO] Running DTW on {dataset_name}")
        clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
        start = time.time()
        clf.fit(X_train_arr.reshape(X_train_arr.shape[0], -1), y_train_arr)
        y_pred = clf.predict(X_test_arr.reshape(X_test_arr.shape[0], -1))
        duration = time.time() - start
        results.append((dataset_name, "DTW", cohen_kappa_score(y_test_arr, y_pred), accuracy_score(y_test_arr, y_pred), duration))
        log_result("1NN-DTW", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] DTW failed on {dataset_name}: {e}")

    # --- RSTSF ---
    try:
        print(f"[INFO] Running RSTSF on {dataset_name}")
        model = rstsf()
        start = time.time()
        model.fit(X_train_arr.reshape(X_train_arr.shape[0], -1), y_train_arr)
        y_pred = model.predict(X_test_arr.reshape(X_test_arr.shape[0], -1))
        duration = time.time() - start
        results.append((dataset_name, "RSTSF", cohen_kappa_score(y_test_arr, y_pred), accuracy_score(y_test_arr, y_pred), duration))
        log_result("RSTSF", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] RSTSF failed on {dataset_name}: {e}")

    # --- BOSS ---
    try:
        print(f"[INFO] Running BOSS on {dataset_name}")
        clf = BOSSEnsemble(random_state=SEED)
        start = time.time()
        clf.fit(X_train_arr.reshape(X_train_arr.shape[0], -1), y_train_arr)
        y_pred = clf.predict(X_test_arr.reshape(X_test_arr.shape[0], -1))
        duration = time.time() - start
        results.append((dataset_name, "BOSS", cohen_kappa_score(y_test_arr, y_pred), accuracy_score(y_test_arr, y_pred), duration))
        log_result("BOSS", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] BOSS failed on {dataset_name}: {e}")
    

    # --- DRCIF ---
    try:
        print(f"[INFO] Running DRCIF on {dataset_name}")
        clf = DrCIFClassifier(random_state=SEED)
        start = time.time()
        clf.fit(X_train_arr.reshape(X_train_arr.shape[0], -1), y_train_arr)
        y_pred = clf.predict(X_test_arr.reshape(X_test_arr.shape[0], -1))
        duration = time.time() - start
        results.append((dataset_name, "DRCIF", cohen_kappa_score(y_test_arr, y_pred), accuracy_score(y_test_arr, y_pred), duration))
        log_result("DRCIF", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] DRCIF failed on {dataset_name}: {e}")

    # --- HC2 ---
    try:
        print(f"[INFO] Running HC2 on {dataset_name}")
        
        clf = HIVECOTEV2(random_state=SEED)
        start = time.time()
        clf.fit(X_train_arr.reshape(X_train_arr.shape[0], -1), y_train_arr)
        y_pred = clf.predict(X_test_arr.reshape(X_test_arr.shape[0], -1))
        duration = time.time() - start
        results.append((dataset_name, "HC2", cohen_kappa_score(y_test_arr, y_pred), accuracy_score(y_test_arr, y_pred), duration))
        log_result("HC2", y_test_arr, y_pred, duration)
    except Exception as e:
        print(f"[ERROR] HC2 failed on {dataset_name}: {e}")

# === MAIN LOOP ===
for dataset in dataset_list:
    try:
        X_train, y_train, W_train, X_test, y_test, W_test = load_ucr_data(dataset)
        run_all_models(X_train, y_train, X_test, y_test, dataset, W_train=W_train, W_test=W_test)

    except Exception as e:
        print(f"[ERROR] Dataset {dataset} failed: {e}")

# === Save Step 1 Results ===
df_results = pd.DataFrame(results, columns=["dataset", "model", "kappa", "accuracy", "duration"])
df_results.to_csv("ucr_noised_results.csv", index=False)
print("Original noised data completed. Results saved to ucr_noised_results.csv")

