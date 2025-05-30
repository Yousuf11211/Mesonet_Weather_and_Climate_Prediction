import pandas as pd
import os

# --- CONFIG ---
input_folder = "filtered_data"
output_folder = "filled_timestamps"
os.makedirs(output_folder, exist_ok=True)

# --- Desired column order ---
desired_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected', 'TAIR', 'DWPT', 'PRCP',
    'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD', 'WDSD', 'WSSD',
    'SM02', 'SM04', 'ST02', 'ST04', 'VT05', 'VT20', 'VT90', 'VR05', 'VR20', 'VR90'
]

# --- Process all CSV files ---
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

for file in csv_files:
    file_path = os.path.join(input_folder, file)
    print(f"\n🔍 Processing {file}...")

    try:
        df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
        df = df.sort_values("UTCTimestampCollected")
        df.set_index("UTCTimestampCollected", inplace=True)

        start_time = df.index.min()
        end_time = df.index.max()
        full_index = pd.date_range(start=start_time, end=end_time, freq="5min")

        missing_times = full_index.difference(df.index)

        print(f"🕒 Total expected timestamps: {len(full_index)}")
        print(f"⛔ Missing 5-min timestamps: {len(missing_times)}")

        if len(missing_times) == 0:
            print("✅ No missing timestamps. Skipping.")
            continue

        confirm = input("Do you want to insert missing rows with empty values? (y/n): ").strip().lower()
        if confirm != "y":
            print("❌ Skipping this file.")
            continue

        # Get site info
        net_site = df["NetSiteAbbrev"].iloc[0]
        county = df["County"].iloc[0]

        # Create new rows
        new_rows = pd.DataFrame({
            "NetSiteAbbrev": net_site,
            "County": county
        }, index=missing_times)
        new_rows.index.name = "UTCTimestampCollected"

        # Fill other columns with NaN
        for col in df.columns:
            if col not in new_rows.columns:
                new_rows[col] = pd.NA

        # Match original columns
        new_rows = new_rows[df.columns]

        # Combine and reset index
        combined_df = pd.concat([df, new_rows])
        combined_df = combined_df.sort_index().reset_index()

        # Reorder to desired column layout
        for col in desired_order:
            if col not in combined_df.columns:
                combined_df[col] = pd.NA
        combined_df = combined_df[desired_order]

        # Save CSV using NetSiteAbbrev
        output_csv = os.path.join(output_folder, f"{net_site}.csv")
        combined_df.to_csv(output_csv, index=False)
        print(f"💾 Saved updated CSV: {output_csv}")

        # Save log of inserted times
        log_file = os.path.join(output_folder, f"{net_site}_missing_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"Missing timestamps inserted for {net_site}:\n")
            for ts in missing_times:
                f.write(ts.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        print(f"📝 Log saved: {log_file}")

    except Exception as e:
        print(f"❌ Failed to process {file}: {e}")
