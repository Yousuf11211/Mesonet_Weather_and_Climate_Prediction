import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# --- Configuration ---
input_folder = 'RandomForest_Regression'  # Folder with completed CSVs
output_folder = 'Final_Importance_Plots'
os.makedirs(output_folder, exist_ok=True)

# --- Columns to analyze ---
targets = ['VT20', 'VT90']
base_features = ['TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD',
                 'WDIR', 'WSPD', 'WDSD', 'WSSD',
                 'SM02', 'SM04', 'ST02', 'ST04',
                 'VT05', 'VR05', 'VR20', 'VR90']

# --- Function to plot and save feature importance ---
def save_feature_importance_plot(feature_names, importances, title, save_path):
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[::-1], sorted_importances[::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Process each CSV file ---
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file in csv_files:
    file_path = os.path.join(input_folder, file)
    print(f"\n📂 Analyzing: {file}")
    df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
    df.set_index("UTCTimestampCollected", inplace=True)

    # Add time features
    df['hour'] = df.index.hour
    df['doy'] = df.index.dayofyear

    for target in targets:
        if target not in df.columns:
            print(f"⚠️ {target} missing in {file}, skipping...")
            continue

        # Final feature set (remove target and NaN-heavy cols)
        features = [f for f in base_features if f in df.columns and f != target]
        features += ['hour', 'doy']

        X = df[features].copy()
        y = df[target].copy()

        # Skip if target has missing
        if y.isna().sum() > 0:
            print(f"⚠️ {target} still has missing values, skipping...")
            continue

        # Impute X
        imp = SimpleImputer(strategy='mean')
        X_imp = imp.fit_transform(X)

        model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        model.fit(X_imp, y)

        importances = model.feature_importances_

        # Save plot
        site = file.replace('.csv', '')
        plot_path = os.path.join(output_folder, f"{site}_{target}_final_importance.png")
        save_feature_importance_plot(features, importances, f"{site} – {target} Feature Importance", plot_path)
        print(f"✅ Saved plot: {plot_path}")
