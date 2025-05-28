import pandas as pd
import os

# Folder and file setup
input_folder = 'b'
output_folder = 'cheoutput'
target_sites = ['ELST.csv', 'LXGN.csv']
os.makedirs(output_folder, exist_ok=True)

cutoff = pd.Timestamp('2025-01-01')

for filename in target_sites:
    path = os.path.join(input_folder, filename)
    site_name = os.path.splitext(filename)[0]
    report_lines = []

    try:
        # Step 0: Count raw lines (excluding header + units)
        raw_total_rows = sum(1 for _ in open(path)) - 2

        # Read and clean
        df = pd.read_csv(path, header=0, skiprows=[1], low_memory=False)
        df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
        df = df.dropna(subset=['UTCTimestampCollected'])
        df = df.sort_values('UTCTimestampCollected')
        df.set_index('UTCTimestampCollected', inplace=True)

        # --- Logic: TAIR + VT90 + SM02 + PRES ---
        required_vars = ['TAIR', 'VT90', 'SM02', 'PRES']
        valid_mask = df[required_vars].notna().all(axis=1)
        start_time = df.index[valid_mask].min()

        expected_index = pd.date_range(start=start_time, end=cutoff, freq='5min')
        missing_ts = expected_index.difference(df.index)
        rows_expected = len(expected_index)
        rows_present = rows_expected - len(missing_ts)

        # --- Console Output ---
        print(f"\nSite: {site_name}")
        print(f"Total Rows in CSV file: {raw_total_rows}")
        print(f"Start (TAIR + VT90 + SM02 + PRES): {start_time}")
        print(f"Expected 5-min timestamps: {rows_expected}")
        print(f"Missing timestamps: {len(missing_ts)}")
        print(f"Actual rows present: {rows_present}")

        # report_lines.append(f"Start (TAIR + VT90 + SM02 + PRES): {start_time}")
        # report_lines.append(f"Expected 5-min timestamps: {rows_expected}")
        # report_lines.append(f"Missing timestamps: {len(missing_ts)}")
        # report_lines.append(f"Actual rows present: {rows_present}")
        # if len(missing_ts) > 0:
        #     report_lines.append("List of Missing Timestamps:")
        #     for ts in missing_ts:
        #         report_lines.append(ts.strftime('%Y-%m-%d %H:%M:%S'))


        # report_path = os.path.join(output_folder, f"{site_name}_timestamp_report.txt")
        # with open(report_path, 'w') as f:
        #     for line in report_lines:
        #         f.write(line + '\n')

        # print(f"Report saved: {report_path}")

        del df

    except Exception as e:
        print(f"Failed to process {site_name}: {e}")
