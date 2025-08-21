import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

base_dir = 'Random_Forest'
output_dir = 'Forecasting_using_lightgbm'
model_output_base = 'trained_models/LightGBM'
target_variables = ['VT20', 'VT90']
drop_cols = ['NetSiteAbbrev', 'County']
sample_size = 500  # for plotting
train_ratio = 0.8

os.makedirs(output_dir, exist_ok=True)

print("Starting LightGBM training and forecasting for all sites...\n")

for site_folder in os.listdir(base_dir):
    site_path = os.path.join(base_dir, site_folder)

    if os.path.isdir(site_path):
        for file in os.listdir(site_path):
            if file.endswith('.csv'):
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

                df = df.drop(columns=drop_cols, errors='ignore')
                df = df.dropna()
                df = df.sort_values(by='UTCTimestampCollected')

                df_features = df.drop(columns=['UTCTimestampCollected'], errors='ignore')
                timestamps = pd.to_datetime(df['UTCTimestampCollected'])
                split_index = int(len(df_features) * train_ratio)
                timestamps_test = timestamps.iloc[split_index:]

                site_model_dir = os.path.join(model_output_base, site_name)
                os.makedirs(site_model_dir, exist_ok=True)

                report_lines = []
                report_lines.append(f"Forecasting Report for {site_name}")
                report_lines.append(f"Timestamp: {datetime.now()}")
                report_lines.append("-" * 50)

                #INDIVIDUAL MODELS
                for target in target_variables:
                    print(f"  Forecasting {target} (single-output)...")

                    if target not in df_features.columns:
                        print(f"    {target} not found. Skipping.")
                        continue

                    X = df_features.drop(columns=[target])
                    y = df_features[target]

                    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

                    model = LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=-1,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)

                    # Save model
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

                    report_lines.append(f"Target Variable: {target}")
                    report_lines.append(f"MAE: {mae:.4f}")
                    report_lines.append(f"R² Score: {r2:.4f}")
                    report_lines.append(f"Approx. Accuracy: {accuracy_pct:.2f}%")
                    report_lines.append("Features used:")
                    report_lines.extend([f"  - {col}" for col in X.columns])
                    report_lines.append("-" * 50)

                    # Plotting
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
#MULTI-OUTPUT MODEL
                print(f"  Forecasting {target_variables} together (multi-output)...")
                if all(t in df_features.columns for t in target_variables):
                    X_multi = df_features.drop(columns=target_variables)
                    y_multi = df_features[target_variables]

                    X_train_mo, X_test_mo = X_multi.iloc[:split_index], X_multi.iloc[split_index:]
                    y_train_mo, y_test_mo = y_multi.iloc[:split_index], y_multi.iloc[split_index:]

                    base_model = LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=-1,
                        random_state=42,
                        n_jobs=-1
                    )
                    multi_model = MultiOutputRegressor(base_model)
                    multi_model.fit(X_train_mo, y_train_mo)

                    # Save multi-output model with site name prefix
                    model_path = os.path.join(site_model_dir, f"{site_name}_VT20_VT90_multi_model.pkl")
                    joblib.dump(multi_model, model_path)
                    print(f"    Multi-output model saved to: {model_path}")

                    predictions = multi_model.predict(X_test_mo)

                    # Save evaluation for multi-output model
                    for i, target in enumerate(target_variables):
                        mae = mean_absolute_error(y_test_mo[target], predictions[:, i])
                        r2 = r2_score(y_test_mo[target], predictions[:, i])
                        accuracy_pct = r2 * 100

                        print(f"    [Multi-output] {target} MAE: {mae:.3f}")
                        print(f"    [Multi-output] {target} R² Score: {r2:.3f}")
                        print(f"    [Multi-output] {target} Approx. Accuracy: {accuracy_pct:.2f}%")

                        report_lines.append(f"[Multi-output] Target Variable: {target}")
                        report_lines.append(f"MAE: {mae:.4f}")
                        report_lines.append(f"R² Score: {r2:.4f}")
                        report_lines.append(f"Approx. Accuracy: {accuracy_pct:.2f}%")
                        report_lines.append("-" * 50)

                    # (Optional) Add multi-output plots here if you want

                else:
                    print("    One or both target variables missing – skipping multi-output.")

                site_report_path = os.path.join(output_dir, f"{site_name}_forecast_report.txt")
                with open(site_report_path, 'w') as f:
                    f.write('\n'.join(report_lines))
                print(f"  Report saved to: {site_report_path}\n")

print("All site forecasts and models completed using LightGBM.")
