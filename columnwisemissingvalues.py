import pandas as pd
import os

# --- CONFIGURATION ---
input_folder = 'filled_timestamps'
target_file = 'ELST.csv'  # Change as needed
file_path = os.path.join(input_folder, target_file)

# --- Load Data ---
df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
total_rows = len(df)

# --- Soil columns observed every 30 mins only ---
soil_cols = ['SM02', 'SM04', 'ST02', 'ST04']
soil_df = df[df['UTCTimestampCollected'].dt.minute.isin([0, 30])]

# --- Function to find longest missing streak and its dates ---
def find_longest_missing_streak(series, timestamps):
    is_nan = series.isna()
    max_len = 0
    current_len = 0
    start_idx = None
    end_idx = None
    best_start = None
    best_end = None

    for i in range(len(series)):
        if is_nan.iloc[i]:
            if current_len == 0:
                start_idx = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                best_start = start_idx
                best_end = i - 1
            current_len = 0

    # Final check at end
    if current_len > max_len:
        max_len = current_len
        best_start = start_idx
        best_end = len(series) - 1

    if best_start is not None and best_end is not None:
        start_time = timestamps.iloc[best_start]
        end_time = timestamps.iloc[best_end]
    else:
        start_time = end_time = None

    return max_len, start_time, end_time

# --- Analyze Missing Values ---
missing_summary = {}
for col in df.columns:
    if col == 'UTCTimestampCollected':
        continue

    if col in soil_cols:
        target_df = soil_df
    else:
        target_df = df

    series = target_df[col]
    timestamps = target_df['UTCTimestampCollected']

    total_missing = series.isna().sum()
    max_streak, streak_start, streak_end = find_longest_missing_streak(series, timestamps)

    if total_missing > 0:
        missing_summary[col] = {
            "Missing Count": total_missing,
            "Max Consecutive Missing": max_streak,
            "Streak Start": streak_start,
            "Streak End": streak_end
        }

# --- Print Summary ---
print(f"\n📊 Missing value summary for {target_file}")
print(f"Total rows in full CSV: {total_rows}")
print(f"Soil rows used (30-min filter): {len(soil_df)}\n")

if not missing_summary:
    print("✅ No missing values found.")
else:
    print(f"{'Column':<20} {'Missing Count':<15} {'Max Consecutive Missing':<25} {'Streak Start':<20} {'Streak End'}")
    print("-" * 100)
    for col, stats in missing_summary.items():
        print(f"{col:<20} {stats['Missing Count']:<15} {stats['Max Consecutive Missing']:<25} {str(stats['Streak Start'])[:19]:<20} {str(stats['Streak End'])[:19]}")
