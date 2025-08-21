import pandas as pd
import os

# Folders
input_folder = 'filtered_data'
output_folder = 'Added_Missing_Timestamps'
log_folder = output_folder
os.makedirs(output_folder, exist_ok=True)

# Files to process
target_sites = ['updated_ELST.csv', 'updated_LXGN.csv']

# Desired column order
desired_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected', 'TAIR', 'DWPT', 'PRCP',
    'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD','WDSD','WSSD',
    'SM02', 'SM04','ST02', 'ST04','VT05', 'VT20', 'VT90', 'VR05', 'VR20', 'VR90'
]

# Function to process each site
for filename in target_sites:
    try:
        site_path = os.path.join(input_folder, filename)
        site_name = filename.replace("updated_", "").replace(".csv", "")

        print(f"\nProcessing {site_name}...")

        # Load the cleaned CSV
        df = pd.read_csv(site_path, parse_dates=["UTCTimestampCollected"])
        df = df.sort_values("UTCTimestampCollected")
        df.set_index("UTCTimestampCollected", inplace=True)

        # Summary before filling
        rows_before = len(df)
        start_time = df.index.min()
        end_time = df.index.max()

        # Create complete timestamp range
        expected_index = pd.date_range(start=start_time, end=end_time, freq='5min')
        rows_expected = len(expected_index)

        # Fill only real gaps (limit 6 forward steps) BEFORE inserting timestamps
        # df[["SM02", "SM04"]] = df[["SM02", "SM04"]].ffill(limit=6)

        # Now reindex to insert missing timestamps (they remain NaN)
        df_full = df.reindex(expected_index)
        df_full.index.name = "UTCTimestampCollected"

        # Reset index to bring timestamp back as a column
        df_full = df_full.reset_index()

        # Set consistent metadata values for all rows
        if "ELST" in filename:
            df_full["NetSiteAbbrev"] = "ELST"
            df_full["County"] = "Madison"
        elif "LXGN" in filename:
            df_full["NetSiteAbbrev"] = "LXGN"
            df_full["County"] = "Choctaw"  # replace with actual if different

        # Reorder columns
        if all(col in df_full.columns for col in desired_order):
            df_full = df_full[desired_order]
        else:
            missing = [col for col in desired_order if col not in df_full.columns]
            print(f"Missing columns in {filename}: {missing}. Columns will not be reordered.")

        # Save the completed CSV
        output_csv_path = os.path.join(output_folder, f"{site_name}_complete.csv")
        df_full.to_csv(output_csv_path, index=False)
        print(f"Saved completed CSV: {output_csv_path}")

        # Prepare the log text
        rows_added = len(expected_index.difference(df.index))
        rows_after = len(df_full)

        report_lines = [
            f"Site: {site_name}",
            f"Start time: {start_time}",
            f"End time: {end_time}",
            f"Original rows: {rows_before}",
            f"Expected rows with 5-min interval: {rows_expected}",
            f"Missing rows inserted: {rows_added}",
            f"Final row count after insertion: {rows_after}"
        ]

        # Save log file
        log_path = os.path.join(log_folder, f"{site_name}_missing_report.txt")
        with open(log_path, 'w') as log_file:
            for line in report_lines:
                log_file.write(line + '\n')
        print(f"Log saved: {log_path}")

        # Print summary to terminal
        print("\n".join(report_lines))

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
