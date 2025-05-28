import pandas as pd
import os

# --- CONFIGURATION ---
#input_folder = "timestamp_only_filled"
#target_file = "ELST_timestamp_filled.csv"

# input_folder = "filtered_data"
# target_file = "updated_ELST.csv"

input_folder = "DUMMYFULL_data"
target_file = "ELST_complete.csv"

file_path = os.path.join(input_folder, target_file)
# --- Load data ---
df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
df.set_index("UTCTimestampCollected", inplace=True)

# --- Print total rows ---
total_rows = len(df)
print(f"\n📄 Total rows in {target_file}: {total_rows}")

# --- Function to analyze a column ---
def analyze_column(series, label):
    series = series.copy()

    changes = (series != series.shift(1)).sum()
    null_count = series.isna().sum()
    zero_count = (series == 0).sum()

    print(f"\n📊 Analysis for {label}")
    print(f"  🔁 Value changes: {changes}")
    print(f"  ❌ Null entries  : {null_count}")
    print(f"  0️⃣  Zero values  : {zero_count}")

# --- Run for both SM02 and SM04 ---
analyze_column(df["SM02"], "SM02")
analyze_column(df["SM04"], "SM04")