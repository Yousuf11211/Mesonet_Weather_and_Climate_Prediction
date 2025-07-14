import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

base_dir = 'Random_Forest'
output_dir = 'Forecasting_using_random_forest'
model_output_base = 'trained_models/RandomForest'
target_variables = ['VT20', 'VT90']
drop_cols = ['NetSiteAbbrev', 'County']
sample_size = 500
train_ratio  = 0.80
# -------------------------------------------------------------

os.makedirs(output_dir,       exist_ok=True)

print("Starting forecasting and training for all sites...\n")

for site_folder in os.listdir(base_dir):
    site_path = os.path.join(base_dir, site_folder)
    if not os.path.isdir(site_path):
        continue

    for file in os.listdir(site_path):
        if not file.endswith('.csv'):
            continue

        csv_path  = os.path.join(site_path, file)
        site_name = file.replace('.csv', '')
        print(f"Processing site: {site_name}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Failed to read {file}: {e}")
            continue

        if 'UTCTimestampCollected' not in df.columns:
            print("  Timestamp column missing – skipping site.")
            continue

        df = df.drop(columns=drop_cols, errors='ignore').dropna()
        df = df.sort_values('UTCTimestampCollected')

        timestamps = pd.to_datetime(df['UTCTimestampCollected'])
        df_features = df.drop(columns=['UTCTimestampCollected'], errors='ignore')

        # Output dirs
        site_model_dir = os.path.join(model_output_base, site_name)
        os.makedirs(site_model_dir, exist_ok=True)

        report_lines = [
            f"Forecasting Report for {site_name}",
            f"Timestamp: {datetime.now()}",
            "-" * 50,
        ]

        split_idx = int(len(df_features) * train_ratio)
        ts_test   = timestamps.iloc[split_idx:]        # for all plots later

        for target in target_variables:
            print(f"  Forecasting {target} (single‑output)…")
            if target not in df_features.columns:
                print(f"    {target} missing – skipped.")
                continue

            X = df_features.drop(columns=[target])
            y = df_features[target]

            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # save model
            m_path = os.path.join(site_model_dir, f"{target}_model.pkl")
            joblib.dump(model, m_path)
            print(f"    Model saved → {m_path}")

            # evaluate
            preds = model.predict(X_test)
            mae   = mean_absolute_error(y_test, preds)
            r2    = r2_score(y_test, preds)

            report_lines += [
                f"Target Variable: {target}",
                f"MAE: {mae:.4f}",
                f"R² Score: {r2:.4f}",
                f"Approx. Accuracy: {r2*100:.2f}%",
                "Features used:",
                *[f"  - {col}" for col in X.columns],
                "-" * 50,
            ]

            n_pts = len(ts_test)
            idx   = (np.linspace(0, n_pts-1, sample_size).astype(int)
                     if n_pts > sample_size else np.arange(n_pts))
            plt.figure(figsize=(10,5))
            plt.plot(ts_test.iloc[idx], y_test.iloc[idx],
                     label='Actual',  color='blue', linewidth=2)
            plt.plot(ts_test.iloc[idx], preds[idx],
                     label='Predicted', color='red',  linestyle='--')
            plt.title(f'{target} Prediction – {site_name}')
            plt.xlabel('Time'); plt.ylabel(target); plt.legend(); plt.tight_layout()

            p_path = os.path.join(output_dir,
                                  f"{site_name}_{target}_forecast_plot.png")
            plt.savefig(p_path); plt.close()
            print(f"    Plot saved   → {p_path}")

        print("  Forecasting VT20 & VT90 together (multi‑output)…")
        if all(t in df_features.columns for t in target_variables):
            X_multi = df_features.drop(columns=target_variables)
            y_multi = df_features[target_variables]           # DataFrame with both cols

            X_tr, X_te = X_multi.iloc[:split_idx], X_multi.iloc[split_idx:]
            y_tr, y_te = y_multi.iloc[:split_idx], y_multi.iloc[split_idx:]

            base_rf = RandomForestRegressor(n_estimators=150, random_state=42)
            mo_model = MultiOutputRegressor(base_rf)
            mo_model.fit(X_tr, y_tr)

            mo_path = os.path.join(site_model_dir,
                                   "VT20_VT90_multi_model.pkl")
            joblib.dump(mo_model, mo_path)
            print(f"    Multi‑output model saved → {mo_path}")

            mo_preds = mo_model.predict(X_te)          # ndarray shape (n,2)
            for i, tgt in enumerate(target_variables):
                mae = mean_absolute_error(y_te[tgt], mo_preds[:, i])
                r2  = r2_score(y_te[tgt], mo_preds[:, i])
                report_lines += [
                    f"[Multi] Target Variable: {tgt}",
                    f"MAE: {mae:.4f}",
                    f"R² Score: {r2:.4f}",
                    f"Approx. Accuracy: {r2*100:.2f}%",
                    "-" * 50,
                ]
        else:
            print("    One of the targets missing – multi‑output skipped.")

        rep_path = os.path.join(output_dir, f"{site_name}_forecast_report.txt")
        with open(rep_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"  Report saved   → {rep_path}\n")

print("All site forecasts and models completed.")
