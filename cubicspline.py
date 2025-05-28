import pandas as pd
import os
from scipy.interpolate import CubicSpline
from datetime import datetime

# --- Folders ---
input_folder = 'Dummyfull_data'
output_folder = 'Spline_interpolation'
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

    df["UTCTimestampCollected"] = pd.to_datetime(df["UTCTimestampCollected"])
    df.set_index("UTCTimestampCollected", inplace=True)
    df.sort_index(inplace=True)
    total_rows = len(df)

    # Get site name
    site_name = df["NetSiteAbbrev"].dropna().unique()
    site_name = site_name[0] if len(site_name) > 0 else file.replace('.csv', '')

    changes_log = []
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    time_seconds = (df.index - df.index[0]).total_seconds()

    for col in numeric_cols:
        if df[col].isna().sum() == 0:
            continue

        print(f"🔧 Interpolating: {col}")
        x = time_seconds[~df[col].isna()]
        y = df[col].dropna()

        if len(y) < 3:
            print(f"⚠️ Skipping {col}: Not enough points for cubic spline.")
            continue

        cs = CubicSpline(x, y)

        x_missing = time_seconds[df[col].isna()]
        y_pred = cs(x_missing)

        df.loc[df[col].isna(), col] = y_pred
        for ts, pred_value in zip(df[df[col].isna()].index, y_pred):
            changes_log.append(f"{col} filled at {ts} | previous: NaN → new: {round(pred_value, 4)}")

        print(f"✅ Filled {len(y_pred)} missing values in {col}.")

    # Save updated CSV
    df_reset = df.reset_index()

    # Reorder columns if possible
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
        log.write(f"Cubic Spline interpolation log for {site_name}\nGenerated: {datetime.now()}\n\n")
        for entry in changes_log:
            log.write(entry + "\n")
    print(f"📝 Log saved: {log_file}")
