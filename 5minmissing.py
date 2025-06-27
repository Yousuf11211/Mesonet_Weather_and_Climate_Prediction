import pandas as pd
import os

# --- CONFIGURATION ---
input_folder = 'filled_timestamps'
target_file = 'ELST.csv'  # Change as needed
file_path = os.path.join(input_folder, target_file)

# --- Load data ---
df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
df = df.dropna(subset=["UTCTimestampCollected"])
df = df.sort_values("UTCTimestampCollected")
df.set_index("UTCTimestampCollected", inplace=True)

# --- Generate full expected 5-min timestamp range ---
start_time = df.index.min()
end_time = df.index.max()
expected_index = pd.date_range(start=start_time, end=end_time, freq="5min")

# --- Find missing timestamps ---
missing = expected_index.difference(df.index)

# --- Print summary ---
print(f"\n🕒 Checking for missing timestamps in: {target_file}")
print(f"Start time: {start_time}")
print(f"End time  : {end_time}")
print(f"Expected timestamps (5-min): {len(expected_index)}")
print(f"Actual timestamps          : {len(df)}")
print(f"Missing timestamps         : {len(missing)}")

if len(missing) > 0:
    print("\nFirst 10 missing timestamps:")
    for ts in missing[:10]:
        print(f" - {ts}")

# --- Save to file ---
output_folder = "missing_timestamp_logs"
os.makedirs(output_folder, exist_ok=True)
site_name = target_file.replace(".csv", "")
output_path = os.path.join(output_folder, f"{site_name}_missing_timestamps.txt")

with open(output_path, 'w') as f:
    f.write(f"Missing 5-min timestamps in {target_file}\n\n")
    for ts in missing:
        f.write(ts.strftime("%Y-%m-%d %H:%M:%S") + "\n")

print(f"\n📄 Saved missing timestamp list to: {output_path}")
