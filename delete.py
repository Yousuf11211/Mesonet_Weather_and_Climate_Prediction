import pandas as pd
import os

# Folder and file setup
input_folder = 'Original_data'
output_folder = 'updated_site_reports'
filtered_folder = 'filtered_data'
target_sites = ['ELST.csv', 'LXGN.csv']
os.makedirs(output_folder, exist_ok=True)
os.makedirs(filtered_folder, exist_ok=True)

cutoff = pd.Timestamp('2025-01-01')

# Desired column order
desired_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected', 'TAIR', 'DWPT', 'PRCP',
    'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD','WDSD','WSSD',
    'SM02', 'SM04','ST02', 'ST04','VT05', 'VT20', 'VT90', 'VR05', 'VR20', 'VR90'
]


for filename in target_sites:
    path = os.path.join(input_folder, filename)
    site_name = os.path.splitext(filename)[0]
    report_lines = []

    try:
        # Count raw data rows (excluding header + units row)
        raw_total_rows = sum(1 for _ in open(path)) - 2

        # Load and clean data
        df = pd.read_csv(path, header=0, skiprows=[1], low_memory=False)
        df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
        df = df.dropna(subset=['UTCTimestampCollected'])
        df = df.sort_values('UTCTimestampCollected')
        df.set_index('UTCTimestampCollected', inplace=True)

        # --- Logic: TAIR + VT90 + SM02 + PRES ---
        required_cols = ['TAIR', 'VT90', 'SM02', 'PRES']
        valid_mask = df[required_cols].notna().all(axis=1)
        start_time = df.index[valid_mask].min()

        expected_index = pd.date_range(start=start_time, end=df.index.max(), freq='5min')
        missing_ts = expected_index.difference(df.index)
        rows_expected = len(expected_index)
        rows_present = rows_expected - len(missing_ts)

        # --- Console Output ---
        print(f"\n📍 Site: {site_name}")
        print(f"Total Rows in CSV file: {raw_total_rows}")
        print(f"Start (TAIR + VT90 + SM02 + PRES): {start_time}")
        print(f"Expected 5-min timestamps: {rows_expected}")
        print(f"Missing timestamps: {len(missing_ts)}")
        print(f"Actual rows present: {rows_present}")

        # # --- Text Report Output ---
        # report_lines.append(f"Site: {site_name}")
        # report_lines.append(f"Total Rows in CSV file: {raw_total_rows}")
        # report_lines.append("")
        # report_lines.append(f"Start (TAIR + VT90 + SM02 + PRES): {start_time}")
        # report_lines.append(f"Expected 5-min timestamps: {rows_expected}")
        # report_lines.append(f"Missing timestamps from {start_time} to {cutoff}: {len(missing_ts)}")
        # report_lines.append(f"Actual rows present: {rows_present}")
        # if len(missing_ts) > 0:
        #     report_lines.append("List of Missing Timestamps:")
        #     report_lines.extend(ts.strftime('%Y-%m-%d %H:%M:%S') for ts in missing_ts)
        #
        # report_path = os.path.join(output_folder, f"{site_name}_timestamp_report.txt")
        # with open(report_path, 'w') as f:
        #     for line in report_lines:
        #         f.write(line + '\n')
        #
        # print(f"✅ Report saved: {report_path}")

        # --- Prompt for deletion of early rows ---
        to_delete = df[df.index < start_time]
        print(f"\n❓ {len(to_delete)} rows occur before {start_time} and will be deleted.")
        user_input = input("Do you want to delete these rows and save a new CSV? (y/n): ").strip().lower()

        if user_input == 'y':
            filtered_df = df.loc[start_time:].copy()
            filtered_df.reset_index(inplace=True)

            # Reorder columns if possible
            if all(col in filtered_df.columns for col in desired_order):
                filtered_df = filtered_df[desired_order]
            else:
                missing_cols = [col for col in desired_order if col not in filtered_df.columns]
                print(f"⚠ Warning: Missing columns {missing_cols}. CSV will not be reordered.")

            output_csv_path = os.path.join(filtered_folder, f"updated_{filename}")
            filtered_df.to_csv(output_csv_path, index=False)
            print(f"New filtered CSV saved: {output_csv_path}")
        else:
            print("Skipped saving filtered CSV.")

        del df

    except Exception as e:
        print(f"Failed to process {site_name}: {e}")
