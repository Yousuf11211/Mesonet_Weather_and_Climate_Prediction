import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
input_folder = "timestamp_only_filled"
target_file = "ELST_timestamp_filled.csv"
file_path = os.path.join(input_folder, target_file)

# --- Load data ---
df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
df.set_index("UTCTimestampCollected", inplace=True)

# --- Select variable ---
variable = "SM04"  # Change to SM02 if needed
color = "green"

# --- Drop NaNs and detect change points only ---
series_full = df[variable].dropna()
change_points = series_full[series_full != series_full.shift(1)]

# --- Get all available years ---
years = sorted(set(change_points.index.year))

for year in years:
    # Filter change points for this year
    series_year = change_points[change_points.index.year == year]

    if len(series_year) < 2:
        print(f"⚠️ Skipping {year} (not enough data)")
        continue

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(series_year.index, series_year.values, marker='o', linestyle='-', color=color, label=f"{variable} ({year})")
    plt.title(f"{variable} Change-Only Plot for {year}")
    plt.xlabel("Time")
    plt.ylabel("Soil Moisture")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
