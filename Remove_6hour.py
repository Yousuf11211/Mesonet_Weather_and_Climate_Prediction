import os
import pandas as pd
from datetime import timedelta

# --- CONFIG ---
input_folder = 'filled_timestamps'          # Folder containing original CSVs
output_folder = 'Gap_Deleted_CSVs'          # Output folder for cleaned CSVs
os.makedirs(output_folder, exist_ok=True)

delete_gap_threshold = timedelta(hours=6)

all_vars = [
    'TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD',
    'WDSD', 'WSSD', 'SM02', 'SM04', 'ST02', 'ST04', 'VT05', 'VT20', 'VT90',
    'VR05', 'VR20', 'VR90'
]
correct_order = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected'] + all_vars

def find_gaps(series, index):
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

input_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file in input_files:
    file_path = os.path.join(input_folder, file)
    file_stem = os.path.splitext(file)[0]
    print(f"\n🔷 Checking {file} for >6hr gaps...")

    df = pd.read_csv(file_path, parse_dates=['UTCTimestampCollected'])
    df.set_index('UTCTimestampCollected', inplace=True)
    df.sort_index(inplace=True)

    deleted_rows = set()
    log_lines = []

    for col in all_vars:
        if col not in df.columns:
            continue

        gaps = find_gaps(df[col], df.index)
        for start, end in gaps:
            duration = df.index[end] - df.index[start]
            if duration > delete_gap_threshold:
                gap_times = df.index[start:end+1]
                deleted_rows.update(gap_times)
                log_lines.append(f"Column: {col} | Gap Start: {df.index[start]} | Gap End: {df.index[end]} | Duration: {duration}\n")

    df_cleaned = df.drop(index=list(deleted_rows))

    # Save cleaned CSV
    df_cleaned.reset_index(inplace=True)
    df_cleaned = df_cleaned[correct_order]
    output_path = os.path.join(output_folder, f"{file_stem}_no_long_gaps.csv")
    df_cleaned.to_csv(output_path, index=False, float_format="%.4f")

    # Save log
    log_path = os.path.join(output_folder, f"{file_stem}_gap_delete_log.txt")
    with open(log_path, 'w') as f:
        f.writelines(log_lines)

    print(f"✅ Saved cleaned CSV: {output_path}")
    print(f"📝 Saved gap log: {log_path}")

print("\n✅ All files processed. Large gaps (>6hr) deleted and saved.")
