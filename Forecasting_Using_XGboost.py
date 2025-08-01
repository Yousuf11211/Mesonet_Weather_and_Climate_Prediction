import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib

# --- CONFIG ---
base_dir = 'Random_Forest'
output_dir = 'Forecasting_using_xgboost'
model_output_base = 'trained_models/XGBoost'
target_variables = ['VT20', 'VT90']
drop_cols = ['NetSiteAbbrev', 'County']
sample_size = 500  # for plotting
train_ratio = 0.8

os.makedirs(output_dir, exist_ok=True)

print("Starting XGBoost training and forecasting for all sites...\n")

for site_folder in os.listdir(base_dir):
    site_path = os.path.join(base_dir, site_folder)

    if not os.path.isdir(site_path):
        continue

    for file in os.listdir(site_path):
        if not file.endswith('.csv'):
            continue

        csv_path = os.path.join(site_path, file)
        site_name = file.replace('.csv', '')
        print(f"Processing site: {site_name}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        if 'UTCTimestampCollected' not in df.columns:
            print("  Timestamp column missing. Skipping time-aware split.")
            continue

        # Drop metadata columns and NA rows
        df = df.drop(columns=drop_cols, errors='ignore').dropna()
        df = df.sort_values(by='UTCTimestampCollected')

        df_features = df.drop(columns=['UTCTimestampCollected'], errors='ignore')
        timestamps = pd.to_datetime(df['UTCTimestampCollected'])

        report_lines = [
            f"Forecasting Report for {site_name}",
            f"Timestamp: {datetime.now()}",
            "-" * 50
        ]

        site_model_dir = os.path.join(model_output_base, site_name)
        os.makedirs(site_model_dir, exist_ok=True)

        # Train single-output models for VT20 and VT90
        for target in target_variables:
            print(f"  Forecasting {target} (single-output)...")

            if target not in df_features.columns:
                print(f"    {target} not found. Skipping.")
                continue

            X = df_features.drop(columns=[target])
            y = df_features[target]

            split_index = int(len(X) * train_ratio)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            timestamps_test = timestamps.iloc[split_index:]

            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=1
            )
            model.fit(X_train, y_train)

            # Save model with site name prefix
            model_path = os.path.join(site_model_dir, f"{site_name}_{target}_model.pkl")
            joblib.dump(model, model_path)
            print(f"    Model saved to: {model_path}")

            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            accuracy_pct = r2 * 100

            print(f"    MAE: {mae:.3f}")
            print(f"    R² Score: {r2:.3f}")
            print(f"    Approx. Accuracy: {accuracy_pct:.2f}%")

            report_lines.extend([
                f"Target Variable: {target}",
                f"MAE: {mae:.4f}",
                f"R² Score: {r2:.4f}",
                f"Approx. Accuracy: {accuracy_pct:.2f}%",
                "Features used:",
                *[f"  - {col}" for col in X.columns],
                "-" * 50,
            ])

            # Plot forecast results
            n_points = len(timestamps_test)
            if n_points > sample_size:
                idx = np.linspace(0, n_points - 1, sample_size).astype(int)
                ts_plot = timestamps_test.iloc[idx]
                actual_plot = y_test.iloc[idx]
                pred_plot = predictions[idx]
            else:
                ts_plot = timestamps_test
                actual_plot = y_test
                pred_plot = predictions

            plt.figure(figsize=(10, 5))
            plt.plot(ts_plot, actual_plot, label='Actual', color='blue', linewidth=2)
            plt.plot(ts_plot, pred_plot, label='Predicted', color='red', linestyle='--')
            plt.title(f'{target} Prediction – {site_name}')
            plt.xlabel('Time')
            plt.ylabel(target)
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f"{site_name}_{target}_forecast_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"    Plot saved to: {plot_path}")

        # Train multi-output model for VT20 & VT90 together
        print("  Forecasting VT20 & VT90 together (multi-output)...")

        if all(t in df_features.columns for t in target_variables):
            X_multi = df_features.drop(columns=target_variables)
            y_multi = df_features[target_variables]

            split_index = int(len(X_multi) * train_ratio)
            X_train_multi, X_test_multi = X_multi.iloc[:split_index], X_multi.iloc[split_index:]
            y_train_multi, y_test_multi = y_multi.iloc[:split_index], y_multi.iloc[split_index:]
            timestamps_test_multi = timestamps.iloc[split_index:]

            base_xgb = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=1
            )
            mo_model = MultiOutputRegressor(base_xgb)
            mo_model.fit(X_train_multi, y_train_multi)

            mo_model_path = os.path.join(site_model_dir, f"{site_name}_VT20_VT90_multi_model.pkl")
            joblib.dump(mo_model, mo_model_path)
            print(f"    Multi-output model saved to: {mo_model_path}")

            mo_preds = mo_model.predict(X_test_multi)
            for i, target in enumerate(target_variables):
                mae = mean_absolute_error(y_test_multi[target], mo_preds[:, i])
                r2 = r2_score(y_test_multi[target], mo_preds[:, i])
                accuracy_pct = r2 * 100

                report_lines.extend([
                    f"[Multi] Target Variable: {target}",
                    f"MAE: {mae:.4f}",
                    f"R² Score: {r2:.4f}",
                    f"Approx. Accuracy: {accuracy_pct:.2f}%",
                    "-" * 50,
                ])

                # Plot multi-output results per target
                n_points = len(timestamps_test_multi)
                if n_points > sample_size:
                    idx = np.linspace(0, n_points - 1, sample_size).astype(int)
                    ts_plot = timestamps_test_multi.iloc[idx]
                    actual_plot = y_test_multi[target].iloc[idx]
                    pred_plot = mo_preds[idx, i]
                else:
                    ts_plot = timestamps_test_multi
                    actual_plot = y_test_multi[target]
                    pred_plot = mo_preds[:, i]

                plt.figure(figsize=(10, 5))
                plt.plot(ts_plot, actual_plot, label='Actual', color='blue', linewidth=2)
                plt.plot(ts_plot, pred_plot, label='Predicted', color='red', linestyle='--')
                plt.title(f'{target} Prediction (Multi-output) – {site_name}')
                plt.xlabel('Time')
                plt.ylabel(target)
                plt.legend()
                plt.tight_layout()

                plot_path = os.path.join(output_dir, f"{site_name}_multi_{target}_forecast_plot.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"    Multi-output plot saved to: {plot_path}")

        else:
            print("    One or both target variables missing; skipping multi-output model.")

        # Save report
        site_report_path = os.path.join(output_dir, f"{site_name}_forecast_report.txt")
        with open(site_report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"  Report saved to: {site_report_path}\n")

print("All site forecasts and models completed using XGBoost.")
