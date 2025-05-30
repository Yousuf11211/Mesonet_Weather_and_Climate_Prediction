import pandas as pd
import os

# --- CONFIGURATION ---
input_folder = 'filtered_data'
target_file = 'ELST.csv'  # Change as needed
file_path = os.path.join(input_folder, target_file)

# --- Load data ---
df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
total_rows = len(df)

# --- Soil columns observed every 30 mins only ---
soil_cols = ['SM02', 'SM04', 'ST02', 'ST04']

# Get rows where minutes = 00 or 30
soil_df = df[df['UTCTimestampCollected'].dt.minute.isin([0, 30])]

# Count missing values
missing_counts = {}
for col in df.columns:
    if col in soil_cols:
        missing_counts[col] = soil_df[col].isna().sum()
    elif col != 'UTCTimestampCollected':  # Exclude timestamp column
        missing_counts[col] = df[col].isna().sum()

# Filter to only those columns that actually have missing values
missing_columns = {col: count for col, count in missing_counts.items() if count > 0}

# --- Print result ---
print(f"\n📊 Missing value summary for {target_file}")
print(f"Total rows in CSV: {total_rows}\n")

if not missing_columns:
    print("✅ No missing values found.")
else:
    for col, count in missing_columns.items():
        print(f"{col:<20} -----> {count}")
