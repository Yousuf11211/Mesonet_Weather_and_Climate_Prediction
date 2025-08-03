import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor

# --- CONFIG ---
input_folder = 'Raw_Data'
output_folder = 'Cleaned_Data'
report_folder = 'Reports'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

delete_gap_threshold = timedelta(hours=6)


def find_gaps(series, index):
    """Find consecutive NaN gaps."""
    gaps = []
    is_nan = series.isna()
    start = None
    for i in range(len(series)):
        if is_nan.iloc[i] and start is None:
            start = i
        elif not is_nan.iloc[i] and start is not None:
            gaps.append((start, i - 1))
            start = None
    if start is not None:
        gaps.append((start, len(series) - 1))
    return gaps


def fill_missing_with_rf(df, target_col):
    """Fill missing values of a column using Random Forest trained on other features."""
    filled_count = 0
    try:
        features = df.drop(columns=[target_col])
        mask = df[target_col].notna()
        if mask.sum() == 0 or mask.sum() == len(df):
            return df[target_col], filled_count  # Nothing to train or predict

        X_train = features[mask]
        y_train = df.loc[mask, target_col]
        X_pred = features[~mask]

        if len(X_train) < 10:
            return df[target_col], filled_count  # Not enough data to train

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_pred)
        df.loc[~mask, target_col] = predictions
        filled_count = len(predictions)
    except Exception as e:
        print(f"⚠️ Error filling {target_col}: {e}")
    return df[target_col], filled_count


input_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file in input_files:
    file_path = os.path.join(input_folder, file)
    print(f"\n🔷 Processing {file}...")

    # --- Read CSV safely ---
    df = pd.read_csv(file_path, dtype=str, low_memory=False, skiprows=[1])
    df['UTCTimestampCollected'] = pd.to_datetime(
        df['UTCTimestampCollected'],
        errors='coerce',
        infer_datetime_format=True
    )
    df = df.dropna(subset=['UTCTimestampCollected'])
    df.sort_values('UTCTimestampCollected', inplace=True)

    # Identify all weather variable columns dynamically
    static_cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']
    all_vars = [c for c in df.columns if c not in static_cols]
    correct_order = static_cols + all_vars

    # Convert weather columns to numeric
    for col in all_vars:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 1️⃣ Add missing 5-min timestamps ---
    station_id_val = df['NetSiteAbbrev'].dropna().iloc[0]
    county_val = df['County'].dropna().iloc[0]

    full_range = pd.date_range(df['UTCTimestampCollected'].min(),
                               df['UTCTimestampCollected'].max(),
                               freq='5min')
    df = df.set_index('UTCTimestampCollected').reindex(full_range).reset_index()
    df.rename(columns={'index': 'UTCTimestampCollected'}, inplace=True)

    df['NetSiteAbbrev'].fillna(station_id_val, inplace=True)
    df['County'].fillna(county_val, inplace=True)

    total_missing_5min = len(full_range) - df['NetSiteAbbrev'].count()

    # --- 2️⃣ Remove gaps > 6 hours ---
    df.set_index('UTCTimestampCollected', inplace=True)
    deleted_rows = set()
    gap_logs = []
    before_rows = len(df)

    for col in all_vars:
        if col not in df.columns:
            continue
        gaps = find_gaps(df[col], df.index)
        for start, end in gaps:
            duration = df.index[end] - df.index[start]
            if duration > delete_gap_threshold:
                deleted_rows.update(df.index[start:end + 1])
                gap_logs.append(
                    f"Column: {col} | Gap Start: {df.index[start]} | Gap End: {df.index[end]} | Duration: {duration}\n")

    df = df.drop(index=list(deleted_rows))
    after_rows = len(df)
    df.reset_index(inplace=True)

    # --- 3️⃣ Fill remaining gaps with Random Forest ---
    missing_counts_before = df[all_vars].isna().sum().to_dict()
    rf_filled_counts = {}
    features_for_rf = df[all_vars].copy()

    for col in all_vars:
        features_for_rf[col], filled_count = fill_missing_with_rf(features_for_rf.copy(), col)
        rf_filled_counts[col] = filled_count

    df[all_vars] = features_for_rf[all_vars]

    # --- 4️⃣ Save cleaned CSV ---
    station_id = station_id_val
    df = df[correct_order]
    output_path = os.path.join(output_folder, f"{station_id}.csv")
    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"✅ Saved cleaned CSV: {output_path}")

    # --- 5️⃣ Save report ---
    report_lines = []
    report_lines.append(f"Station ID: {station_id}\n")
    report_lines.append(f"Total rows (original): {len(full_range)}\n")
    report_lines.append(f"Missing 5-min timestamps added: {total_missing_5min}\n")
    report_lines.append(f"Number of 6-hour gaps removed: {len(gap_logs)}\n")
    report_lines.extend(gap_logs)
    report_lines.append("\nMissing values per attribute before filling:\n")
    for col, count in missing_counts_before.items():
        report_lines.append(f"{col}: {count}\n")
    report_lines.append("\nValues filled by Random Forest per attribute:\n")
    for col, count in rf_filled_counts.items():
        pct = (count / max(1, missing_counts_before[col])) * 100 if col in missing_counts_before else 0
        report_lines.append(f"{col}: {count} ({pct:.1f}%)\n")
    report_lines.append(f"\nRows before removing 6-hour gaps: {before_rows}\n")
    report_lines.append(f"Rows after removing 6-hour gaps: {after_rows}\n")

    report_path = os.path.join(report_folder, f"{station_id}_report.txt")
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    print(f"📝 Saved report: {report_path}")

print("\n✅ All files processed successfully.")
