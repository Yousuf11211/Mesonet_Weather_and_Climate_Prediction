import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import shap

# --- CONFIG ---
input_folder = 'New_Random_Forest'          # Folder with CSVs
output_folder = 'SHAP_Feature_Plots_Full'   # Output folder
targets = ['VT20', 'VT90']
all_vars = [
    'TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD',
    'WDSD', 'WSSD', 'SM02', 'SM04', 'ST02', 'ST04', 'VT05', 'VT20', 'VT90',
    'VR05', 'VR20', 'VR90'
]
batch_size = 1000

os.makedirs(output_folder, exist_ok=True)

def compute_shap_for_target(X, y, site_name, target):
    print(f"  → Training RandomForest for {target} on {len(X)} rows...")
    imp = SimpleImputer(strategy='mean')
    X_imp = imp.fit_transform(X)

    model = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_imp, y)

    explainer = shap.TreeExplainer(model)

    shap_sum = np.zeros(X.shape[1])
    total_rows = len(X)

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_X = X_imp[start:end]
        print(f"    → SHAP calculating rows {start} to {end}...")

        shap_values = explainer.shap_values(batch_X)
        shap_sum += np.abs(shap_values).sum(axis=0)

    shap_mean = shap_sum / total_rows
    sorted_idx = np.argsort(shap_mean)[::-1]
    sorted_features = np.array(X.columns)[sorted_idx]
    sorted_importance = shap_mean[sorted_idx]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_features)), sorted_importance, color='skyblue')
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
    plt.title(f"{site_name} – SHAP Feature Importance for {target}")
    plt.tight_layout()

    plot_path = os.path.join(output_folder, f"{site_name}_{target}_shap_full.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"  ✔ SHAP plot saved: {plot_path}")

# --- Main Loop ---
for file in os.listdir(input_folder):
    if not file.endswith('.csv'):
        continue

    file_path = os.path.join(input_folder, file)
    site_name = os.path.splitext(file)[0]

    print(f"\n[SHAP] Processing {file}...")

    df = pd.read_csv(file_path, parse_dates=['UTCTimestampCollected'])
    df = df[all_vars].dropna()

    if df.empty:
        print(f"  ❌ Skipped (no complete rows after dropna).")
        continue

    for target in targets:
        if target not in df.columns:
            print(f"  ❌ Target {target} missing in data.")
            continue

        X = df.drop(columns=targets, errors='ignore')
        y = df[target]

        compute_shap_for_target(X, y, site_name, target)

print("\n✅ SHAP Full-Dataset Batch Processing Completed.")
