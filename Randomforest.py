import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ------------- CONFIG -------------
input_folder = 'Gap_Deleted_CSVs'
output_root = 'Random_Forest'
os.makedirs(output_root, exist_ok=True)

original_column_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected',
    'TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD',
    'WDIR', 'WSPD', 'WDSD', 'WSSD',
    'SM02', 'SM04', 'ST02', 'ST04',
    'VT05', 'VT20', 'VT90',
    'VR05', 'VR20', 'VR90'
]

non_numeric_cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']

# ------------- FUNCTION TO FILL MISSING AND PLOT IMPORTANCE -------------
def fill_missing_values(df, filename, site_folder):
    filled_stats = {}
    per_variable_summary = {}

    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]

    if 'UTCTimestampCollected' in df.columns:
        try:
            df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
            df['Hour'] = df['UTCTimestampCollected'].dt.hour
            df['Month'] = df['UTCTimestampCollected'].dt.month
            df['DayOfYear'] = df['UTCTimestampCollected'].dt.dayofyear
        except Exception as e:
            print(f"Timestamp conversion failed: {e}")

    drop_cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']
    meta_data = df[drop_cols].copy() if all(col in df.columns for col in drop_cols) else pd.DataFrame()
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    for target_col in numeric_cols:
        missing_count = df[target_col].isnull().sum()
        total_count = len(df)

        if missing_count == 0:
            filled_stats[target_col] = 0
            per_variable_summary[target_col + '_R2'] = 'NA'
            per_variable_summary[target_col + '_Top3'] = 'NA'
            continue

        feature_cols = [col for col in df.columns if col != target_col]

        train_df = df[df[target_col].notnull()]
        predict_df = df[df[target_col].isnull()]

        if train_df.empty or predict_df.empty:
            filled_stats[target_col] = 0
            per_variable_summary[target_col + '_R2'] = 'NA'
            per_variable_summary[target_col + '_Top3'] = 'NA'
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_pred = predict_df[feature_cols]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)

        y_pred = model.predict(X_pred)
        df.loc[df[target_col].isnull(), target_col] = y_pred

        importances = model.feature_importances_
        importance_percent = 100 * importances / importances.sum()

        sorted_idx = np.argsort(importance_percent)[::-1]
        sorted_features = [feature_cols[i] for i in sorted_idx]
        sorted_importance = [importance_percent[i] for i in sorted_idx]

        top_features = list(zip(sorted_features, np.round(sorted_importance, 2)))[:3]
        top_features_str = "; ".join([f"{f}: {imp}%" for f, imp in top_features])

        per_variable_summary[target_col + '_R2'] = round(train_score, 3)
        per_variable_summary[target_col + '_Top3'] = top_features_str

        # Plot and save
        plt.figure(figsize=(12, 6))
        plt.bar(sorted_features, sorted_importance)
        plt.ylabel('Importance (%)')
        plt.title(f'Feature Importance for {target_col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = os.path.join(site_folder, f"{target_col}.png")
        plt.savefig(plot_path)
        plt.close()

        filled_percent = (missing_count / total_count) * 100
        filled_stats[target_col] = round(filled_percent, 2)

    # Reattach metadata
    if not meta_data.empty:
        df = pd.concat([meta_data.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    # Reorder columns
    df = df[[col for col in original_column_order if col in df.columns]]

    # Merge summaries
    summary_row = {**filled_stats, **per_variable_summary}
    return df, summary_row

# ------------- MAIN SCRIPT -------------
print("Starting batch processing of CSV files...")
summary_stats = []

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        print(f"\nProcessing file: {filename}")

        try:
            df = pd.read_csv(file_path)
            site_name = os.path.splitext(filename)[0]
            site_folder = os.path.join(output_root, site_name)
            os.makedirs(site_folder, exist_ok=True)

            print(f"Loaded {filename}: {len(df)} rows")

            filled_df, stats = fill_missing_values(df, filename, site_folder)

            output_csv_path = os.path.join(site_folder, "ELST.csv")
            filled_df.to_csv(output_csv_path, index=False)
            print(f"Saved filled CSV to {output_csv_path}")

            stats['filename'] = filename
            summary_stats.append(stats)

            # Save report.txt for the site
            report_path = os.path.join(site_folder, "report.txt")
            with open(report_path, "w") as f:
                f.write(f"Report for {filename}\n\n")
                for key, value in stats.items():
                    if key != 'filename':
                        f.write(f"{key}: {value}\n")
            print(f"Saved report to {report_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# ------------- SAVE OVERALL SUMMARY -------------
if summary_stats:
    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = os.path.join(output_root, 'filling_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary report saved: {summary_csv_path}")
else:
    print("\nNo files were processed successfully.")

print("All tasks completed.")
