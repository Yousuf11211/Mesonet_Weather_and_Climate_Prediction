import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

# --- CONFIG ---
model_types = ['LightGBM', 'RandomForest', 'XGBoost']
sites = ['ELST', 'LXGN']
drop_cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']
base_model_dir = 'trained_models'
base_data_dir = 'Random_Forest'
output_report_dir = 'Forecasting_Comparison_Reports'
os.makedirs(output_report_dir, exist_ok=True)

print("Starting evaluation for all available models...\n")

for model_type in model_types:
    for site in sites:
        site_model_path = os.path.join(base_model_dir, model_type, site)
        data_path = os.path.join(base_data_dir, f"{site}_no_long_gaps", f"{site}.csv")

        if not os.path.exists(site_model_path):
            print(f"Model path not found: {site_model_path}")
            continue
        if not os.path.exists(data_path):
            print(f"Data CSV not found: {data_path}")
            continue

        print(f"Processing models for site: {site}, model type: {model_type}")
        df_original = pd.read_csv(data_path)
        report_lines = [f"Evaluation Report for {site} – {model_type}"]

        for model_file in sorted(os.listdir(site_model_path)):
            if not model_file.endswith('.pkl'):
                continue

            # Keep full model filename without .pkl extension
            model_target = model_file.replace(".pkl", "")

            model_path = os.path.join(site_model_path, model_file)

            try:
                model = joblib.load(model_path)
            except Exception as e:
                print(f"  Failed to load model {model_file}: {e}")
                continue

            # Reload fresh data for each model
            df = df_original.copy()
            print(f"CSV Loaded: {data_path}")

            # Determine target columns based on model filename substring
            if 'VT20_VT90' in model_target:
                target_cols = ['VT20', 'VT90']
            elif 'VT20' in model_target:
                target_cols = ['VT20']
            elif 'VT90' in model_target:
                target_cols = ['VT90']
            else:
                print(f"  Unknown target type in model filename: {model_file}. Skipping.")
                continue

            cols_to_drop = drop_cols + target_cols
            print(f"Columns Dropped: {cols_to_drop}")

            features = df.drop(columns=cols_to_drop, errors='ignore')
            targets = df[target_cols]

            print(f"Model Used: {model_target} ({model_type})")

            # Reorder features if model supports it
            model_features = None
            if hasattr(model, 'feature_name_'):
                model_features = model.feature_name_
            elif hasattr(model, 'estimators_'):
                base_estimator = model.estimators_[0]
                if hasattr(base_estimator, 'feature_name_'):
                    model_features = base_estimator.feature_name_

            if model_features:
                features = features.reindex(columns=model_features, fill_value=0)

            try:
                preds = model.predict(features)

                if len(target_cols) == 2:  # Multi-output model
                    mae1 = mean_absolute_error(targets[target_cols[0]], preds[:, 0])
                    mae2 = mean_absolute_error(targets[target_cols[1]], preds[:, 1])
                    report_lines.append(f"{model_target}: MAE {target_cols[0]} = {mae1:.4f}, MAE {target_cols[1]} = {mae2:.4f}")
                    print(f"Prediction completed for model: {model_target} (SUCCESS)")
                else:
                    mae = mean_absolute_error(targets[target_cols[0]], preds)
                    report_lines.append(f"{model_target}: MAE = {mae:.4f}")
                    print(f"Prediction completed for model: {model_target} (SUCCESS)")
            except Exception as e:
                report_lines.append(f"{model_target}: Prediction FAILED - {e}")
                print(f"Prediction failed for model: {model_target} (FAILED) - {e}")

            # Clear memory
            del df, features, targets
            print("Data cleared, ready to reload CSV for next model.\n")

        # Save report file for this site and model type
        report_path = os.path.join(output_report_dir, f"{site}_{model_type}_forecast_eval.txt")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Report saved to: {report_path}\n")

print("Evaluation completed for all models.")
