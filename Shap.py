import os
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURATION ---
root_folder = 'Random_Forest'
target_features = ['VT90_TAIR_diff', 'VT90_VT20_diff']

# --- UTILITY FUNCTION ---
def generate_shap_summary_plots(csv_path, save_dir, target):
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Add difference columns if not present
    if 'VT90_TAIR_diff' not in df.columns and 'VT90' in df.columns and 'TAIR' in df.columns:
        print("Adding column: VT90_TAIR_diff")
        df['VT90_TAIR_diff'] = df['VT90'] - df['TAIR']
    if 'VT90_VT20_diff' not in df.columns and 'VT90' in df.columns and 'VT20' in df.columns:
        print("Adding column: VT90_VT20_diff")
        df['VT90_VT20_diff'] = df['VT90'] - df['VT20']

    if target not in df.columns:
        print(f"Target column '{target}' not found in {csv_path}. Skipping this target.")
        return

    print(f"Dropping rows with NA in target column '{target}'")
    df = df.dropna(subset=[target])
    print(f"Remaining rows after dropping NA in target: {len(df)}")

    X = df.drop(columns=[target])
    y = df[target]

    # Drop non-numeric columns and columns with any NA values
    print("Selecting numeric columns and dropping columns with NA in features")
    X = X.select_dtypes(include=['number']).dropna(axis=1)
    print(f"Features considered for model training: {list(X.columns)}")

    if len(X.columns) < 2:
        print(f"Not enough numeric features to compute SHAP for target '{target}' in {csv_path}. Need at least 2.")
        return

    print(f"Training RandomForestRegressor model for target '{target}'")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("Model training completed")

    print("Initializing SHAP explainer")
    explainer = shap.Explainer(model, X)

    sample_size = min(len(X), 200)
    print(f"Sampling {sample_size} rows from data for SHAP computation")
    X_sample = X.sample(sample_size, random_state=42)

    print("Calculating SHAP values")
    shap_values = explainer(X_sample)
    print("SHAP values calculation done")

    print("Generating SHAP summary plot")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
    plt.title(f"SHAP Summary Plot for {target}")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{target}_shap_beeswarm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot to: {save_path}")


# --- MAIN SCRIPT ---
print(f"Starting SHAP summary plot generation in root folder: '{root_folder}'")
for site_folder in os.listdir(root_folder):
    site_path = os.path.join(root_folder, site_folder)
    if not os.path.isdir(site_path):
        print(f"Skipping non-directory item: {site_path}")
        continue

    csv_path = os.path.join(site_path, "ELST.csv")
    if not os.path.exists(csv_path):
        print(f"ELST.csv not found in {site_path}, skipping this site")
        continue

    print(f"\nProcessing site folder: {site_folder}")
    shap_save_dir = os.path.join(site_path, "shap_summary")

    for target in target_features:
        print(f"Processing target feature: {target}")
        generate_shap_summary_plots(csv_path, shap_save_dir, target)

print("\nAll SHAP summary plots have been generated.")
