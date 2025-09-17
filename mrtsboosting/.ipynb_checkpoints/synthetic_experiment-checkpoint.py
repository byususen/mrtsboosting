#Synthetic Experiment (2025)

import random

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

#ALL 25+25 iteration
import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report, cohen_kappa_score
from pyts.classification import TimeSeriesForest
from sktime.datatypes import convert_to
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.interval_based import DrCIF as DrCIFClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sklearn.ensemble import RandomForestClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.dictionary_based import BOSSEnsemble
from sklearn.metrics import accuracy_score

def generate_synthetic_crop(n_sample, A_range, B_range, C_noise, D_range, label, n_days, cloud_prop, name_prefix):
    x = np.arange(5, n_days + 1, 5)
    vi, cloud, days, ids, labels = [], [], [], [], []

    for i in range(n_sample):
        idsamp = f'{name_prefix}_{i}'
        A = np.random.uniform(*A_range)
        B = np.random.uniform(*B_range)
        C = np.random.uniform(-10, 10) if C_noise == 'low' else np.random.uniform(-B, B)
        D = np.random.uniform(*D_range)

        for j in x:
            y = A * np.sin(2 * (np.pi / B) * (j - C)) + D + np.random.uniform(-0.05, 0.05)
            vi.append(y)
            cloud.append(np.random.uniform(0.9, 0.95))
            days.append(j)
            ids.append(idsamp)
            labels.append(label)

        days_cloud = np.random.choice(x, int(cloud_prop * len(x)), replace=False)
        for dc in days_cloud:
            for idx in range(len(days)):
                if days[idx] == dc and ids[idx] == idsamp:
                    vi[idx] = np.random.uniform(0.01, 0.1)
                    cloud[idx] = np.random.uniform(0.01, 0.1)

    return np.array(vi), np.array(cloud), np.array(days), np.array(ids), np.array(labels)

def generate_all_crops(n_sample, n_days, cloud_prop, q):
    crop_params = [
        ((0.25, 0.35), (30, 90), 'low', (0.5, 0.55), 0, 'crop0'),
        ((0.15, 0.25), (30, 90), 'low', (0.5, 0.55), 1, 'crop1'),
        ((0.05, 0.15), (30, 90), 'low', (0.5, 0.55), 2, 'crop2'),
        ((0.25, 0.35), (90, 180), 'low', (0.5, 0.55), 3, 'crop3'),
        ((0.15, 0.25), (90, 180), 'low', (0.5, 0.55), 4, 'crop4'),
        ((0.05, 0.15), (90, 180), 'low', (0.5, 0.55), 5, 'crop5'),
        ((0.25, 0.35), (180, 360), 'low', (0.5, 0.55), 6, 'crop6'),
        ((0.15, 0.25), (180, 360), 'low', (0.5, 0.55), 7, 'crop7'),
        ((0.05, 0.15), (180, 360), 'low', (0.5, 0.55), 8, 'crop8'),
    ]

    vi_all, cloud_all, days_all, id_all, label_all = [], [], [], [], []

    for A_range, B_range, C_noise, D_range, label, prefix in crop_params:
        vi, cloud, days, ids, labels = generate_synthetic_crop(
            n_sample=n_sample,
            A_range=A_range,
            B_range=B_range,
            C_noise='low' if q == 0 else 'high',
            D_range=D_range,
            label=label,
            n_days=n_days,
            cloud_prop=cloud_prop,
            name_prefix=prefix
        )
        vi_all.append(vi)
        cloud_all.append(cloud)
        days_all.append(days)
        id_all.append(ids)
        label_all.append(labels)

    # Forest - linear
    x = np.arange(5, n_days + 1, 5)
    vi_forest, cloud_forest, days_forest, id_forest, label_forest = [], [], [], [], []
    for i in range(n_sample):
        idsamp = f'forest_{i}'
        y_init = np.random.uniform(0.10, 0.90)
        for j in x:
            y = y_init + np.random.uniform(-0.05, 0.05)
            vi_forest.append(y)
            cloud_forest.append(np.random.uniform(0.9, 0.95))
            days_forest.append(j)
            id_forest.append(idsamp)
            label_forest.append(9)
        days_cloud = np.random.choice(x, int(cloud_prop * len(x)), replace=False)
        for dc in days_cloud:
            for idx in range(len(days_forest)):
                if days_forest[idx] == dc and id_forest[idx] == idsamp:
                    vi_forest[idx] = np.random.uniform(0.01, 0.1)
                    cloud_forest[idx] = np.random.uniform(0.01, 0.1)

    # Random trend
    vi_random, cloud_random, days_random, id_random, label_random = [], [], [], [], []
    trend_strength = np.random.uniform(-0.0004, 0.0004)
    y_init = np.random.uniform(0.5, 0.9) if trend_strength < 0 else np.random.uniform(0.1, 0.5)
    for i in range(n_sample):
        idsamp = f'random_{i}'
        for j in x:
            y = y_init + trend_strength * j + np.random.uniform(-0.05, 0.05)
            vi_random.append(y)
            cloud_random.append(np.random.uniform(0.9, 0.95))
            days_random.append(j)
            id_random.append(idsamp)
            label_random.append(10)
        days_cloud = np.random.choice(x, int(cloud_prop * len(x)), replace=False)
        for dc in days_cloud:
            for idx in range(len(days_random)):
                if days_random[idx] == dc and id_random[idx] == idsamp:
                    vi_random[idx] = np.random.uniform(0.01, 0.1)
                    cloud_random[idx] = np.random.uniform(0.01, 0.1)

    # Append forest and trend
    vi_all.extend([np.array(vi_forest), np.array(vi_random)])
    cloud_all.extend([np.array(cloud_forest), np.array(cloud_random)])
    days_all.extend([np.array(days_forest), np.array(days_random)])
    id_all.extend([np.array(id_forest), np.array(id_random)])
    label_all.extend([np.array(label_forest), np.array(label_random)])

    return (
        np.concatenate(vi_all),
        np.concatenate(cloud_all),
        np.concatenate(days_all),
        np.concatenate(id_all),
        np.concatenate(label_all)
    )

def split_train_test_ids(id_all, label_all, prop_test=0.3, random_state=SEED):
    rng = np.random.default_rng(random_state)
    class_ids = {label: np.unique(id_all[label_all == label]) for label in np.unique(label_all)}
    id_test_u = np.concatenate([rng.choice(ids, int(prop_test * len(ids)), replace=False) for ids in class_ids.values()])
    id_train_u = np.setdiff1d(id_all, id_test_u)
    return id_train_u, id_test_u


def sine_func(x, A, B, C, D):
    y = A * np.sin(2*(np.pi/B)*(x-C)) + D
    return y

# --- Scenario settings ---
scenarios = {
    'low_cloud': 0.1,
    'moderate_cloud': 0.3,
    'high_cloud': 0.5
}


start_iter = 1  # or 25 for the second part
n_iter = 5     # number of iterations in each part

n_sample = 20
n_days = 500
prop_test = 0.3

results_by_scenario = {}

for scenario_name, prop_cloud in scenarios.items():
    print(f"\n=== Running Scenario: {scenario_name} ===")
    results_by_shift = {
        'df_mrtsboost': {}, 'df_tsf': {}, 'df_rocket': {}, 'df_dtw': {},
        'df_rstsf': {}, 'df_st': {}, 'df_drcif': {}, 'df_hc2': {}, 'df_boss': {},
        'kappa_mrtsboost': {}, 'kappa_tsf': {}, 'kappa_rocket': {},
        'kappa_dtw': {}, 'kappa_rstsf': {}, 'kappa_st': {}, 'kappa_drcif': {}, 'kappa_hc2': {}, 'kappa_boss': {},
        'accuracy_mrtsboost': {}, 'accuracy_tsf': {}, 'accuracy_rocket': {},
        'accuracy_dtw': {}, 'accuracy_rstsf': {}, 'accuracy_st': {}, 'accuracy_drcif': {}, 'accuracy_hc2': {}, 'accuracy_boss': {},
        'duration_df': {}
    }


    for q in [0, 1]:
        # Initialize accumulators for this q value
        eval_dfs = {m: pd.DataFrame() for m in ['mrtsboost', 'tsf', 'rocket', 'dtw', 'rstsf', 'st', 'drcif', 'hc2', 'boss']}
        kappa_scores = {m: [] for m in eval_dfs.keys()}
        durations = {m: [] for m in eval_dfs.keys()}
        accuracy_scores = {m: [] for m in eval_dfs.keys()}
        duration_df = pd.DataFrame(index=np.arange(n_iter), columns=[f'duration_{m}' for m in eval_dfs.keys()])

        for r in range(start_iter, start_iter + n_iter):
            print(f"[Scenario: {scenario_name} | Q: {q} | Iteration: {r}]")
            vi_all, cloud_all, days_all, id_all, label_all = generate_all_crops(n_sample, n_days, prop_cloud, q)
            id_train_u, id_test_u = split_train_test_ids(id_all, label_all, prop_test, random_state=SEED)
            indices_train = np.where(np.isin(id_all, id_train_u))[0]
            indices_test = np.where(np.isin(id_all, id_test_u))[0]

            def select_by_indices(arr, idx): return np.delete(arr, idx)

            id_train = select_by_indices(id_all, indices_test)
            vi_train = select_by_indices(vi_all, indices_test)
            cloud_train = select_by_indices(cloud_all, indices_test)
            days_train = select_by_indices(days_all, indices_test)
            label_train = select_by_indices(label_all, indices_test)

            id_test = select_by_indices(id_all, indices_train)
            vi_test = select_by_indices(vi_all, indices_train)
            cloud_test = select_by_indices(cloud_all, indices_train)
            days_test = select_by_indices(days_all, indices_train)
            label_test = select_by_indices(label_all, indices_train)

            unique_ids, unique_indices = np.unique(id_test, return_index=True)
            label_test_u = label_test[unique_indices]

            # Format data for multivariate MRTSBoost
            x_data_dict_train = {'vi': {'id': id_train, 'time': days_train, 'value': vi_train, 'weight': cloud_train}}
            x_data_dict_test = {'vi': {'id': id_test, 'time': days_test, 'value': vi_test, 'weight': cloud_test}}
            y_data_dict_train = {'id': id_train, 'label': label_train}
            y_data_dict_test = {'id': id_test, 'label': label_test}

            def evaluate_model(name, model, X_train, y_train, X_test, y_test):
                start = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
                # Special handling for MRTSBoost
                if name == 'mrtsboost':
                    df_y = pd.DataFrame({'id': y_test['id'], 'label': y_test['label']})
                    y_test_array = df_y.groupby('id').max()['label'].values
                else:
                    y_test_array = y_test
            
                duration = time.time() - start
                kappa = cohen_kappa_score(y_test_array, y_pred)
                acc = accuracy_score(y_test_array, y_pred)
                report = classification_report(y_test_array, y_pred, output_dict=True)
            
                # Reshape report to long format
                df_rows = []
                for label, metrics in report.items():
                    if label in ['accuracy', 'macro avg', 'weighted avg']:
                        continue  # Optional: skip these if focusing on per-class
                    for metric_name in ['precision', 'recall', 'f1-score']:
                        df_rows.append({
                            'model': name,
                            'iteration': r,
                            'q': q,
                            'scenario': scenario_name,
                            'class': label,
                            'metric': metric_name,
                            'value': metrics[metric_name]
                        })
            
                df = pd.DataFrame(df_rows)
                return df, kappa, acc, duration

            # Run all models
            models = {
                'mrtsboost': MRTSBoostingClassifier(n_window=50, window_min=30, min_period=10, n_jobs=-1, random_state=SEED),
                'tsf': TimeSeriesForest(n_jobs=-1, random_state=SEED),
                'rocket': RocketClassifier(n_jobs=-1, random_state=SEED),
                'drcif': DrCIFClassifier(n_jobs=-1, random_state=SEED),
                'dtw': KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw"),  # no random_state needed
                'rstsf': rstsf(),
                'hc2': HIVECOTEV2(n_jobs=-1, random_state=SEED),
                'boss': BOSSEnsemble(n_jobs=-1, random_state=SEED),
                'st': ShapeletTransformClassifier(n_jobs=-1, random_state=SEED)
            }


            for model_name, model in models.items():
                if model_name == 'mrtsboost':
                    X_train = model.preprocess_x_data_dict(x_data_dict_train)
                    X_test = model.preprocess_x_data_dict(x_data_dict_test)
                    y_train = y_data_dict_train
                    y_test = y_data_dict_test
                elif model_name in ['tsf', 'rocket', 'hc2', 'drcif', 'rstsf', 'st', 'dtw']:
                    df_train = pd.DataFrame({'id': id_train, 'days': days_train, 'vi': vi_train})
                    df_test = pd.DataFrame({'id': id_test, 'days': days_test, 'vi': vi_test})
                    df_train.set_index(['id', 'days'], inplace=True)
                    df_test.set_index(['id', 'days'], inplace=True)
                    X_train = convert_to(df_train, to_type="numpy3D").reshape(len(np.unique(id_train)), -1)
                    X_test = convert_to(df_test, to_type="numpy3D").reshape(len(np.unique(id_test)), -1)
                    y_train = pd.DataFrame({'id': id_train, 'label': label_train}).groupby('id').max()['label'].values
                    y_test = pd.DataFrame({'id': id_test, 'label': label_test}).groupby('id').max()['label'].values
            
                df_eval, kappa, acc, dur = evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
                
                # 🖨️ Log the results per model
                print(f"[{scenario_name} | Q={q} | Iter {r:02d}] {model_name.upper()} → "
                      f"Kappa: {kappa:.3f} | Acc: {acc:.3f} | Duration: {dur:.2f}s")
            
                eval_dfs[model_name] = pd.concat([eval_dfs[model_name], df_eval], ignore_index=True)
                kappa_scores[model_name].append(kappa)
                accuracy_scores[model_name].append(acc)
                durations[model_name].append(dur)
                duration_df.loc[r, f'duration_{model_name}'] = dur

        # Save per q value
        for model in eval_dfs:
            results_by_shift[f'df_{model}'][q] = eval_dfs[model]
            results_by_shift[f'kappa_{model}'][q] = kappa_scores[model]
            results_by_shift[f'accuracy_{model}'][q] = accuracy_scores[model]  # <-- Add this line

        results_by_shift['duration_df'][q] = duration_df

    # Combine all
    eval_all = pd.concat([results_by_shift[f'df_{m}'][q] for m in eval_dfs for q in [0, 1]], ignore_index=True)
    kappa_all = pd.DataFrame([
        {'model': m, 'iteration': i, 'kappa': k, 'q': q, 'scenario': scenario_name}
        for m in eval_dfs
        for q in [0, 1]
        for i, k in enumerate(results_by_shift[f'kappa_{m}'][q])
    ])
    accuracy_all = pd.DataFrame([
        {'model': m, 'iteration': i, 'accuracy': a, 'q': q, 'scenario': scenario_name}
        for m in eval_dfs
        for q in [0, 1]
        for i, a in enumerate(results_by_shift[f'accuracy_{m}'][q])
    ])
    metrics_all = pd.merge(kappa_all, accuracy_all, on=['model', 'iteration', 'q', 'scenario'])

    durations_all = pd.concat([
        results_by_shift['duration_df'][q].assign(q=q, scenario=scenario_name)
        for q in [0, 1]
    ], ignore_index=True)

    results_by_scenario[scenario_name] = {
        'eval': eval_all,
        'kappa': kappa_all,
        'accuracy': accuracy_all,
        'combined_metrics': metrics_all,
        'duration': durations_all
    }

# Export per scenario
for scenario_name, result_dict in results_by_scenario.items():
    suffix = f"{scenario_name}_part_{start_iter}"
    result_dict['eval'].to_csv(f"eval_all_scenarios_new_{suffix}.csv", index=False)
    result_dict['combined_metrics'].to_csv(f"combined_metrics_all_scenarios_new_{suffix}.csv", index=False)
    result_dict['duration'].to_csv(f"duration_all_scenarios_new_{suffix}.csv", index=False)
