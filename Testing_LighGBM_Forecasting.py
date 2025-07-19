import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

# Config
model_types = ['LightGBM', 'RandomForest', 'XGBoost']
sites = ['ELST', 'LXGN']
target_vars = ['VT20', 'VT90', 'VT20_VT90_multi']
base_model_dir = 'trained_models'
base_data_dir = 'Random_Forest'
output_report_dir = 'Forecasting_Comparison_Reports'
drop_cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']
os.makedirs(output_report_dir, exist_ok=True)

print("Starting model evaluation...\n")

for model_type in model_types:
    print(f"--- MODEL TYPE: {model_type} ---")
    for site in sites:
        # Compose path to model folder for this site & type
        model_site_dir = os.path.join(base_model_dir, model_type, site)
        if not os.path.exists(model_site_dir):
            print(f"  Model directory not found for {site} under {model_type}, skipping.")
            continue

        # Compose path to CSV data file
        data_folder = f"{site}_no_long_gaps"
        csv_path = os.path.join(base_data_dir, data_folder, f"{site}.csv")
        if not os.path.exists(csv_path):
            print(f"  Data file missing for {site} at {csv_path}, skipping.")
            continue

        # Load the CSV data
        df = pd.read_csv(csv_path)
        print(f"  Loaded data for {site}, {len(df)} rows.")

        errors = {}

        # Sort model files to process in consistent order
        for model_file in sorted(os.listdir(model_site_dir)):
            if not model_file.endswith('.pkl'):
                continue
            model_path = os.path.join(model_site_dir, model_file)
            model_name = model_file.replace('_model.pkl', '')

            try:
                model = joblib.load(model_path)
            except Exception as e:
                print(f"    Failed to load model {model_file}: {e}")
                continue

            try:
                if model_name == 'VT20':
                    if 'VT20' not in df.columns:
                        print(f"    VT20 column missing in data, skipping model {model_name}")
                        continue
                    # Drop only VT20 + metadata columns
                    features = df.drop(columns=drop_cols + ['VT20'], errors='ignore')
                    target = df['VT20']

                elif model_name == 'VT90':
                    if 'VT90' not in df.columns:
                        print(f"    VT90 column missing in data, skipping model {model_name}")
                        continue
                    # Drop only VT90 + metadata columns
                    features = df.drop(columns=drop_cols + ['VT90'], errors='ignore')
                    target = df['VT90']

                elif model_name == 'VT20_VT90_multi':
                    if 'VT20' not in df.columns or 'VT90' not in df.columns:
                        print(f"    VT20 or VT90 columns missing for multi-output model, skipping {model_name}")
                        continue
                    # For multi-output, drop only metadata columns
                    features = df.drop(columns=drop_cols, errors='ignore')
                    target = df[['VT20', 'VT90']]

                else:
                    print(f"    Unknown model name {model_name}, skipping.")
                    continue

                if hasattr(model, 'feature_name_'):
                    model_features = model.feature_name_
                elif hasattr(model, 'estimators_'):
                    # For MultiOutputRegressor, get feature names from base estimator
                    base_estimator = model.estimators_[0]
                    if hasattr(base_estimator, 'feature_name_'):
                        model_features = base_estimator.feature_name_
                    else:
                        model_features = None
                else:
                    model_features = None

                if model_features is not None:
                    missing_feats = set(model_features) - set(features.columns)
                    extra_feats = set(features.columns) - set(model_features)
                    if missing_feats:
                        print(f"    WARNING: Missing features for model {model_name}: {missing_feats}")
                    if extra_feats:
                        print(f"    WARNING: Extra features in data for model {model_name}: {extra_feats}")

                    # Reorder columns exactly to match training order, ignore missing columns for safety
                    features = features.reindex(columns=model_features, fill_value=0)

                preds = model.predict(features)

                if model_name == 'VT20_VT90_multi':
                    mae_vt20 = mean_absolute_error(target['VT20'], preds[:, 0])
                    mae_vt90 = mean_absolute_error(target['VT90'], preds[:, 1])
                    errors[model_name] = (mae_vt20, mae_vt90)
                else:
                    mae = mean_absolute_error(target, preds)
                    errors[model_name] = mae

            except Exception as e:
                print(f"    Prediction failed for model {model_name}: {e}")

        if not errors:
            print(f"  No models evaluated for site {site}.")
            continue

        # Save report
        report_path = os.path.join(output_report_dir, f"{model_type}_{site}_forecast_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Forecasting Evaluation Report for site: {site}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write("-" * 50 + "\n")

            for model_n, err in errors.items():
                if model_n == 'VT20_VT90_multi':
                    f.write(f"{model_n} Model MAE:\n")
                    f.write(f"  VT20: {err[0]:.4f}\n")
                    f.write(f"  VT90: {err[1]:.4f}\n")
                else:
                    f.write(f"{model_n} Model MAE: {err:.4f}\n")
            f.write("-" * 50 + "\n")

        print(f"  Report saved to: {report_path}\n")

print("Evaluation completed.")
