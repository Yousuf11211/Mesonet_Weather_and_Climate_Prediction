import pandas as pd
import os

# --- Configuration ---
input_folder = 'filtered_data'
output_folder = 'timestamp_only_filled'
os.makedirs(output_folder, exist_ok=True)

# Files to process
target_files = ['updated_ELST.csv', 'updated_LXGN.csv']

# County lookup (adjust if needed)
site_metadata = {
    "ELST": "Madison",
    "LXGN": "Choctaw"
}

for filename in target_files:
    try:
        site_path = os.path.join(input_folder, filename)
        site_code = filename.replace("updated_", "").replace(".csv", "")
        county = site_metadata.get(site_code, "Unknown")

        print(f"\n🔄 Processing {site_code}...")

        # Load file
        df = pd.read_csv(site_path, parse_dates=["UTCTimestampCollected"])
        df = df.sort_values("UTCTimestampCollected")
        df.set_index("UTCTimestampCollected", inplace=True)

        # Create complete 5-minute time range
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
        df_full = df.reindex(full_index)
        df_full.index.name = "UTCTimestampCollected"

        # Reset index to make timestamp a column
        df_full = df_full.reset_index()

        # Fill only metadata
        df_full["NetSiteAbbrev"] = site_code
        df_full["County"] = county

        # Reorder so metadata appears first
        cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected'] + [col for col in df_full.columns if col not in ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']]
        df_full = df_full[cols]

        # Save output
        output_path = os.path.join(output_folder, f"{site_code}_timestamp_filled.csv")
        df_full.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
