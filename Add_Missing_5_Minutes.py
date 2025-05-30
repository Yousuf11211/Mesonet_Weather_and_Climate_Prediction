import pandas as pd
import os

# --- CONFIG ---
input_folder = "filtered_data"  # Your folder containing site CSVs
output_folder = "filled_timestamps"
os.makedirs(output_folder, exist_ok=True)

# --- Process all CSV files in the folder ---
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

for file in csv_files:
    file_path = os.path.join(input_folder, file)
    print(f"\n🔍 Processing {file}...")

    try:
        df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
        df = df.sort_values("UTCTimestampCollected")
        df.set_index("UTCTimestampCollected", inplace=True)

        # Determine timestamp range
        start_time = df.index.min()
        end_time = df.index.max()
        full_index = pd.date_range(start=start_time, end=end_time, freq="5min")

        missing_times = full_index.difference(df.index)

        print(f"🕒 Total expected timestamps: {len(full_index)}")
        print(f"⛔ Missing 5-min timestamps: {len(missing_times)}")

        if len(missing_times) == 0:
            print("✅ No missing timestamps. Skipping.")
            continue

        # Ask for confirmation
        confirm = input("Do you want to insert missing rows with empty values? (y/n): ").strip().lower()
        if confirm != "y":
            print("❌ Skipping this file.")
            continue

        # Prepare rows to add
        net_site = df["NetSiteAbbrev"].iloc[0]
        county = df["County"].iloc[0]

        new_rows = pd.DataFrame({
            "NetSiteAbbrev": net_site,
            "County": county
        }, index=missing_times)
        new_rows.index.name = "UTCTimestampCollected"

        # Fill any missing columns with NaN to match input structure
        for col in df.columns:
            if col not in new_rows.columns:
                new_rows[col] = pd.NA

        # Match original column order
        new_rows = new_rows[df.columns]

        # Combine, sort, and reset index
        combined_df = pd.concat([df, new_rows])
        combined_df = combined_df.sort_index().reset_index()

        # Save updated CSV
        site_name = os.path.splitext(file)[0]
        new_csv_path = os.path.join(output_folder, f"{site_name}_with_missing.csv")
        combined_df.to_csv(new_csv_path, index=False)
        print(f"💾 Saved updated CSV: {new_csv_path}")

        # Save TXT log
        log_path = os.path.join(output_folder, f"{site_name}_missing_log.txt")
        with open(log_path, 'w') as f:
            f.write(f"Missing timestamps inserted for {site_name}:\n")
            for ts in missing_times:
                f.write(ts.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        print(f"📝 Log saved: {log_path}")

    except Exception as e:
        print(f"❌ Failed to process {file}: {e}")
