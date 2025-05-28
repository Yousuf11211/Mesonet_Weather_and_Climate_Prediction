import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from datetime import datetime

# --- Folders ---
input_folder = 'Dummyfull_data'
output_folder = 'RandomForest_Regression'
os.makedirs(output_folder, exist_ok=True)

# --- Desired column order ---
desired_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected', 'TAIR', 'DWPT', 'PRCP',
    'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD', 'VT05',
    'SM02', 'SM04', 'VT20', 'VT90', 'VR05', 'VR20', 'VR90'
]

# --- Get all CSV files ---
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file in csv_files:
    print(f"\n🔍 Processing file: {file}")
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])

    # Ensure consistent datetime index
    df["UTCTimestampCollected"] = pd.to_datetime(df["UTCTimestampCollected"])
    df.set_index("UTCTimestampCollected", inplace=True)
    df.sort_index(inplace=True)
    total_rows = len(df)

    # Get site name from data
    site_name = df["NetSiteAbbrev"].dropna().unique()
    site_name = site_name[0] if len(site_name) > 0 else file.replace('.csv', '')

    changes_log = []
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Impute all columns with missing data
    missing_counts = df[numeric_cols].isna().sum()
    missing_columns = missing_counts[missing_counts > 0]

    print(f"\n📊 Missing summary (Total rows: {total_rows}):")
    if missing_columns.empty:
        print("✅ No missing values found.")
        continue

    for col, count in missing_columns.items():
        print(f"{col:<20} -----> {count}")

    for col_to_process in missing_columns.index:
        print(f"\n🔧 Processing column: {col_to_process}")
        target = col_to_process
        features = [c for c in numeric_cols if c != target]

        df_rf = df[features + [target]].copy()
        train_df = df_rf.dropna()
        test_df = df_rf[df_rf[target].isna()].drop(columns=[target])

        if train_df.empty or test_df.empty:
            print(f"⚠️ Skipping {target}: Not enough data.")
            continue

        X_train = train_df[features]
        y_train = train_df[target]

        imp = SimpleImputer(strategy='mean')
        X_train_imp = imp.fit_transform(X_train)
        X_test_imp = imp.transform(test_df)

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_imp, y_train)
        y_pred = model.predict(X_test_imp)

        for idx, ts in enumerate(test_df.index):
            pred_value = round(y_pred[idx], 4)
            df.loc[ts, target] = pred_value
            changes_log.append(f"{target} filled at {ts} | previous: NaN → new: {pred_value}")

        # Add feature importance summary
        importance = model.feature_importances_
        feature_info = "\n".join([f"    {feat}: {round(imp, 4)}" for feat, imp in zip(features, importance)])
        changes_log.append(f"\n📈 Feature importance for {target}:\n{feature_info}")

        print(f"✅ Filled {len(y_pred)} values in '{target}' using Random Forest.")

    # Save updated file and log
    df_reset = df.reset_index()

    # Reorder columns
    if all(col in df_reset.columns for col in desired_order):
        df_reset = df_reset[desired_order]
    else:
        missing_cols = [col for col in desired_order if col not in df_reset.columns]
        print(f"⚠️ Missing columns in file: {missing_cols}. Will skip reordering.")

    output_csv = os.path.join(output_folder, f"{site_name}.csv")
    df_reset.to_csv(output_csv, index=False, float_format="%.4f")
    print(f"✅ CSV saved: {output_csv}")

    log_file = os.path.join(output_folder, f"{site_name}.txt")
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"Random Forest interpolation log for {site_name}\nGenerated: {datetime.now()}\n\n")
        for entry in changes_log:
            log.write(entry + "\n")
    print(f"📝 Log saved: {log_file}")
