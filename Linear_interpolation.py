import pandas as pd
import os

input_folder = 'Testing'
output_folder = 'Linear_interpolation'
os.makedirs(output_folder, exist_ok=True)

desired_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected', 'TAIR', 'DWPT', 'PRCP',
    'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD','WDSD','WSSD',
    'SM02', 'SM04','ST02', 'ST04','VT05', 'VT20', 'VT90', 'VR05', 'VR20', 'VR90'
]

csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file in csv_files:
    print(f"\nProcessing file: {file}")
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
    df.set_index("UTCTimestampCollected", inplace=True)
    total_rows = len(df)

    site_id = df["NetSiteAbbrev"].dropna().unique()
    site_name = site_id[0] if len(site_id) > 0 else file.replace('.csv', '')
    changes_log = []

    while True:
        missing_counts = df.isna().sum()
        missing_columns = missing_counts[missing_counts > 0]

        print(f"\nMissing value summary (Total rows: {total_rows}):")
        if missing_columns.empty:
            print("No missing values left.")
            break

        for col, count in missing_columns.items():
            print(f"{col:<20} -----> {count}")

        # Automatically select the first missing column
        col_to_process = missing_columns.index[0]

        # Interpolate only missing values and round to 4 decimals
        missing_before = df[df[col_to_process].isna()]
        interpolated = df[col_to_process].interpolate(method='linear', limit_direction='both')

        for ts in missing_before.index:
            interpolated_value = interpolated.loc[ts]
            if pd.notna(interpolated_value):
                df.at[ts, col_to_process] = round(interpolated_value, 4)

        missing_after = df[df[col_to_process].isna()]
        filled = set(missing_before.index) - set(missing_after.index)

        for ts in sorted(filled):
            prev_value = missing_before.loc[ts, col_to_process]
            new_value = df.loc[ts, col_to_process]
            changes_log.append(f"{col_to_process} filled at {ts} | previous: {prev_value} → new: {new_value}")

        print(f"Filled {len(filled)} missing values in '{col_to_process}'.")

    # Save updated CSV and log
    df_reset = df.reset_index()

    # Reorder columns if possible
    if all(col in df_reset.columns for col in desired_order):
        df_reset = df_reset[desired_order]
    else:
        missing_cols = [col for col in desired_order if col not in df_reset.columns]
        print(f"Missing columns: {missing_cols}. Skipping reordering.")

    output_csv = os.path.join(output_folder, f"{site_name}.csv")
    df_reset.to_csv(output_csv, index=False)
    print(f"CSV saved: {output_csv}")

    log_file = os.path.join(output_folder, f"{site_name}.txt")
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"Interpolation log for {site_name}\n\n")
        for entry in changes_log:
            log.write(entry + "\n")
    print(f"Log saved: {log_file}")
