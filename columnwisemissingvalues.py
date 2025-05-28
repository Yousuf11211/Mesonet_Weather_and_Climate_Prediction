import pandas as pd
import os

# --- CONFIGURATION ---
input_folder = 'Linear_interpolation'
target_file = 'ELST.csv'  # Change as needed
file_path = os.path.join(input_folder, target_file)
# --- Load data ---
df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
total_rows = len(df)

# --- Count missing values ---
missing_counts = df.isna().sum()
missing_columns = missing_counts[missing_counts > 0]

# --- Print result ---
print(f"\n📊 Missing value summary for {target_file}")
print(f"Total rows in CSV: {total_rows}\n")

if missing_columns.empty:
    print("✅ No missing values found in any column.")
else:
    for col, count in missing_columns.items():
        print(f"{col:<20} -----> {count}")