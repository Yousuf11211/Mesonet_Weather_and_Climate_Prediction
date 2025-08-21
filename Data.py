import pandas as pd
import glob
import os

# Show all columns when printing
pd.set_option('display.max_columns', None)

# Get all CSV files from folder 'a'
file_paths = glob.glob('b/*.csv')
print(f"Found {len(file_paths)} CSV files.\n")

# Prepare output folder
os.makedirs("site_reports", exist_ok=True)

# Use 5-minute interval (future-proof)
expected_freq = '5min'

# Track all features for comparison
feature_sets = []

# Process each file
for file_path in file_paths:
    site_name = os.path.splitext(os.path.basename(file_path))[0]
    report_lines = []
    try:
        # Read and parse file
        df = pd.read_csv(file_path, header=0, skiprows=[1], low_memory=False)
        df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
        df = df.dropna(subset=['UTCTimestampCollected'])

        # Find duplicate timestamps (but don't remove)
        duplicate_rows = df[df.duplicated(subset='UTCTimestampCollected', keep=False)]

        # Sort by time and set index
        df = df.sort_values('UTCTimestampCollected')
        df.set_index('UTCTimestampCollected', inplace=True)

        # Expected full time index (5-minute spacing)
        start_time = df.index.min()
        end_time = df.index.max()
        expected_index = pd.date_range(start=start_time, end=end_time, freq=expected_freq)
        missing_timestamps = expected_index.difference(df.index)

        # Track columns (features)
        features = set(df.columns)
        feature_sets.append(features)
        print(f"Site: {site_name}")
        print(f"Start: {start_time}")
        print(f"End: {end_time}")
        print(f"Rows: {len(df)}")
        print(f"Duplicate Timestamps: {len(duplicate_rows)}");
        # # Build text report
        # report_lines.append(f"Site: {site_name}")
        # report_lines.append(f"Start: {start_time}")
        # report_lines.append(f"End: {end_time}")
        # report_lines.append(f"Rows: {len(df)}")
        # report_lines.append(f"Duplicate Timestamps: {len(duplicate_rows)}")
        # if not duplicate_rows.empty:
        #     report_lines.append("\nList of Duplicate Timestamps:")
        #     report_lines.extend(duplicate_rows.index.strftime('%Y-%m-%d %H:%M:%S').tolist())

        # report_lines.append(f"\nExpected Rows (5-min interval): {len(expected_index)}")
        # report_lines.append(f"Missing Timestamps: {len(missing_timestamps)}")
        # report_lines.append(f"Missing %: {round(len(missing_timestamps) / len(expected_index) * 100, 2)}")
        # if len(missing_timestamps) > 0:
        #     report_lines.append("\nList of Missing Timestamps:")
        #     report_lines.extend(missing_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist())

        # # Write to site report file
        # report_path = os.path.join("site_reports", f"{site_name}_report.txt")
        # with open(report_path, 'w') as f:
        #     for line in report_lines:
        #         f.write(line + '\n')

        # print(f"Report saved: {report_path}")

    except Exception as e:
        print(f"Failed to process {site_name}: {e}")

