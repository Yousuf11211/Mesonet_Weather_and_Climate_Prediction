import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend suitable for scripts
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import shap


# --- FEATURE IMPORTANCE FUNCTION ---
def calculateFeatureImportance(df, target_col, output_dir, excluded_features=None):
    print(f"[FeatureImportance] Starting feature importance calculation for target: {target_col}")
    if excluded_features is None:
        excluded_features = ['VT90_TAIR_diff', 'VT90_VT20_diff', 'VT90', 'VT20', 'TAIR',
                             'SM04', 'ST04', 'UTCTimestampCollected', 'NetSiteAbbrev', 'County']

    print(f"[FeatureImportance] Excluding features: {excluded_features}")

    # Prepare data
    X = df.drop(columns=[target_col] + [f for f in excluded_features if f in df.columns])
    y = df[target_col]
    print(f"[FeatureImportance] Data prepared: {X.shape[0]} samples, {X.shape[1]} features")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[FeatureImportance] Split data into train ({X_train.shape[0]}) and test ({X_test.shape[0]})")

    # --- Decision Tree ---
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_imp = pd.Series(dt.feature_importances_, index=X.columns)
    print("[FeatureImportance] Decision Tree model trained")

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    rf_imp = pd.Series(rf_perm.importances_mean, index=X.columns)
    print("[FeatureImportance] Random Forest model trained and permutation importance calculated")

    # --- SHAP ---
    print("[FeatureImportance] Computing SHAP values (this might take some time)...")
    explainer = shap.PermutationExplainer(rf.predict, X_test, n_jobs=-1)
    X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer(X_test_sample)
    print("[FeatureImportance] SHAP values computed")

    # --- Plot Decision Tree ---
    dt_plot = dt_imp.nlargest(10)
    plt.figure(figsize=(10, 6))
    dt_plot.plot.barh()
    plt.title(f"{target_col} - Decision Tree Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    dt_plot_path = os.path.join(output_dir, f"{target_col}_dt.png")
    plt.savefig(dt_plot_path)
    plt.close()
    print(f"[FeatureImportance] Decision Tree importance plot saved to {dt_plot_path}")

    # --- Plot Random Forest ---
    rf_plot = rf_imp.nlargest(10)
    plt.figure(figsize=(10, 6))
    rf_plot.plot.barh()
    plt.title(f"{target_col} - Random Forest Importance")
    plt.xlabel("Mean Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    rf_plot_path = os.path.join(output_dir, f"{target_col}_rf.png")
    plt.savefig(rf_plot_path)
    plt.close()
    print(f"[FeatureImportance] Random Forest importance plot saved to {rf_plot_path}")

    # --- SHAP Summary Plot ---
    shap_plot_path = os.path.join(output_dir, f"{target_col}_shap.png")
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(shap_plot_path)
    plt.close()
    print(f"[FeatureImportance] SHAP summary plot saved to {shap_plot_path}")

    # --- Save R² and RMSE ---
    dt_r2 = r2_score(y_test, y_pred_dt)
    rf_r2 = r2_score(y_test, y_pred_rf)
    dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print(f"[FeatureImportance] Decision Tree R2: {dt_r2:.4f}, RMSE: {dt_rmse:.4f}")
    print(f"[FeatureImportance] Random Forest R2: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")

    return {
        'target': target_col,
        'DecisionTree_R2': round(dt_r2, 4),
        'DecisionTree_RMSE': round(dt_rmse, 4),
        'RandomForest_R2': round(rf_r2, 4),
        'RandomForest_RMSE': round(rf_rmse, 4)
    }


# --- MISSING VALUE FILLING FUNCTION ---
original_column_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected',
    'TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD',
    'WDIR', 'WSPD', 'WDSD', 'WSSD',
    'SM02', 'SM04', 'ST02', 'ST04',
    'VT05', 'VT20', 'VT90',
    'VR05', 'VR20', 'VR90'
]

non_numeric_cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']


def fill_missing_values(df, filename, site_folder):
    print(f"[FillMissing] Starting missing value fill for {filename}")
    filled_stats = {}
    per_variable_summary = {}

    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
    print(f"[FillMissing] Numeric columns to process: {numeric_cols}")

    if 'UTCTimestampCollected' in df.columns:
        try:
            df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
            df['Hour'] = df['UTCTimestampCollected'].dt.hour
            df['Month'] = df['UTCTimestampCollected'].dt.month
            df['DayOfYear'] = df['UTCTimestampCollected'].dt.dayofyear
            print("[FillMissing] Extracted Hour, Month, DayOfYear from UTCTimestampCollected")
        except Exception as e:
            print(f"[FillMissing] Timestamp conversion failed: {e}")

    drop_cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']
    meta_data = df[drop_cols].copy() if all(col in df.columns for col in drop_cols) else pd.DataFrame()
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Create folder for individual feature importance plots
    individual_fi_folder = os.path.join(site_folder, "individual_feature_importance")
    os.makedirs(individual_fi_folder, exist_ok=True)
    print(f"[FillMissing] Created folder for individual feature importance plots: {individual_fi_folder}")

    for target_col in numeric_cols:
        missing_count = df[target_col].isnull().sum()
        total_count = len(df)
        print(f"[FillMissing] Processing {target_col}: missing values = {missing_count} / {total_count}")

        if missing_count == 0:
            print(f"[FillMissing] No missing values for {target_col}, skipping filling.")
            filled_stats[target_col] = 0
            per_variable_summary[target_col + '_R2'] = 'NA'
            per_variable_summary[target_col + '_Top3'] = 'NA'
            continue

        feature_cols = [col for col in df.columns if col != target_col]
        train_df = df[df[target_col].notnull()]
        predict_df = df[df[target_col].isnull()]

        if train_df.empty or predict_df.empty:
            print(f"[FillMissing] Not enough data to train or predict for {target_col}, skipping.")
            filled_stats[target_col] = 0
            per_variable_summary[target_col + '_R2'] = 'NA'
            per_variable_summary[target_col + '_Top3'] = 'NA'
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_pred = predict_df[feature_cols]

        print(f"[FillMissing] Training Random Forest model for {target_col}...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print(f"[FillMissing] Model trained.")

        train_score = model.score(X_train, y_train)
        print(f"[FillMissing] Training R² score for {target_col}: {train_score:.4f}")

        y_pred = model.predict(X_pred)
        df.loc[df[target_col].isnull(), target_col] = y_pred
        print(f"[FillMissing] Missing values filled for {target_col}")

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

        plot_path = os.path.join(individual_fi_folder, f"{target_col}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[FillMissing] Feature importance plot saved for {target_col} at {plot_path}")

        filled_percent = (missing_count / total_count) * 100
        filled_stats[target_col] = round(filled_percent, 2)

    # Reattach metadata
    if not meta_data.empty:
        df = pd.concat([meta_data.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
        print("[FillMissing] Reattached metadata columns")

    # Reorder columns
    df = df[[col for col in original_column_order if col in df.columns]]
    print("[FillMissing] Reordered columns to original order")

    # Merge summaries
    summary_row = {**filled_stats, **per_variable_summary}
    print(f"[FillMissing] Completed filling missing values for {filename}")

    return df, summary_row


# --- MAIN SCRIPT ---
input_folder = 'Gap_Deleted_CSVs'
output_root = 'Random_Forest'
os.makedirs(output_root, exist_ok=True)

summary_stats = []

for filename in os.listdir(input_folder):
    if not filename.endswith('.csv'):
        print(f"[Main] Skipping non-CSV file: {filename}")
        continue

    file_path = os.path.join(input_folder, filename)
    site_name = os.path.splitext(filename)[0]
    site_folder = os.path.join(output_root, site_name)
    os.makedirs(site_folder, exist_ok=True)

    print(f"\n[Main] Processing file: {filename}")

    try:
        df = pd.read_csv(file_path)
        print(f"[Main] Loaded {filename}: {len(df)} rows")

        # Fill missing values
        filled_df, stats = fill_missing_values(df, filename, site_folder)

        # Save filled CSV
        output_csv_path = os.path.join(site_folder, "ELST.csv")
        filled_df.to_csv(output_csv_path, index=False)
        print(f"[Main] Saved filled CSV to {output_csv_path}")

        # Save filling report
        report_path = os.path.join(site_folder, "report.txt")
        with open(report_path, "w") as f:
            f.write(f"Report for {filename}\n\n")
            for key, value in stats.items():
                if key != 'filename':
                    f.write(f"{key}: {value}\n")
        print(f"[Main] Saved report to {report_path}")

        # Create folder for difference target feature importance
        diff_fi_folder = os.path.join(site_folder, "difference_feature_importance")
        os.makedirs(diff_fi_folder, exist_ok=True)
        print(f"[Main] Created folder for difference feature importance: {diff_fi_folder}")

        # --- Run feature importance on the filled dataframe for difference targets ---
        for target in ['VT90_TAIR_diff', 'VT90_VT20_diff']:
            # Add difference features if not present
            if 'VT90_TAIR_diff' not in filled_df.columns:
                filled_df['VT90_TAIR_diff'] = filled_df['VT90'] - filled_df['TAIR']
                print("[Main] Added 'VT90_TAIR_diff' column")
            if 'VT90_VT20_diff' not in filled_df.columns:
                filled_df['VT90_VT20_diff'] = filled_df['VT90'] - filled_df['VT20']
                print("[Main] Added 'VT90_VT20_diff' column")

            print(f"[Main] Calculating feature importance for: {target}")
            fi_results = calculateFeatureImportance(filled_df, target, diff_fi_folder)

            # Append the feature importance metrics to the stats report
            with open(report_path, "a") as f:
                f.write(f"\nFeature Importance Results for {target}:\n")
                for k, v in fi_results.items():
                    f.write(f"{k}: {v}\n")

            # Also save a CSV summary for feature importance metrics per target
            fi_summary_csv = os.path.join(diff_fi_folder, f"{target}_feature_importance_summary.csv")
            pd.DataFrame([fi_results]).to_csv(fi_summary_csv, index=False)
            print(f"[Main] Saved feature importance summary to {fi_summary_csv}")

        # Add filename to stats for overall summary CSV
        stats['filename'] = filename
        summary_stats.append(stats)

    except Exception as e:
        print(f"[Main] Error processing {filename}: {e}")

# --- Save overall summary CSV ---
if summary_stats:
    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = os.path.join(output_root, 'filling_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[Main] Summary report saved: {summary_csv_path}")
else:
    print("\n[Main] No files were processed successfully.")

print("[Main] All tasks completed.")
