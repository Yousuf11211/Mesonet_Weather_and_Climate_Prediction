import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# --- CONFIG ---
input_folder = 'New_Random_Forest'  # Folder with filled CSVs
output_folder = 'Fixed_Feature_Plots'
os.makedirs(output_folder, exist_ok=True)

targets = ['VT20', 'VT90']
all_vars = [
    'TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD',
    'WDSD', 'WSSD', 'SM02', 'SM04', 'ST02', 'ST04', 'VT05', 'VT20', 'VT90',
    'VR05', 'VR20', 'VR90'
]

# --- Main Loop ---
for file in os.listdir(input_folder):
    if not file.endswith('.csv'):
        continue

    file_path = os.path.join(input_folder, file)
    site_name = os.path.splitext(file)[0]
    print(f"\n[Processing] {file}")

    # Load data
    df = pd.read_csv(file_path, parse_dates=['UTCTimestampCollected'])
    df = df[all_vars].dropna()

    if df.empty:
        print(f"  Skipping {file}, no complete rows")
        continue

    imp = SimpleImputer(strategy='mean')

    importance_dict = {}

    for target in targets:
        print(f"  Training RandomForest for target: {target}")

        # Exclude BOTH target variables (VT20 and VT90) from feature inputs for both cases
        features = [col for col in all_vars if col not in targets]

        X = df[features]
        y = df[target]

        X_imp = imp.fit_transform(X)

        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_imp, y)

        importances = model.feature_importances_
        importance_dict[target] = importances

    # --- Plot Combined ---
    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, importance_dict['VT20'], width, label='VT20')
    ax.bar(x + width / 2, importance_dict['VT90'], width, label='VT90')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance for VT20 & VT90 – {site_name}")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_folder, f"{site_name}_feature_importance.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"  ✔ Saved plot: {plot_path}")

print("\n✅ All feature importance plots generated (RandomForest, no data leakage, both targets excluded from inputs for each run).")
