from herbie import Herbie  # <-- Correct import for HRRR downloader
import pygrib
import pandas as pd
from datetime import datetime
from pathlib import Path

# ---------------- CONFIG ----------------
date_to_download = datetime(2025, 1, 29)  # Change date here
output_csv = Path(__file__).parent / f"HRRR_Surface_AllVars_{date_to_download.strftime('%Y%m%d')}.csv"

# Create empty list to collect data dictionaries
all_data = []

# Loop through 24 hours of the day
for hour in range(24):
    run_time = date_to_download.replace(hour=hour)
    print(f"Downloading HRRR surface for {run_time}...")

    try:
        H = Herbie(
            run_time.strftime("%Y-%m-%d %H:%M"),
            model="hrrr",
            product="sfc",
            fxx=0,  # Analysis data
            save_dir=Path(__file__).parent / "HRRR_Downloads"
        )

        # Download the GRIB2 file if not exists
        grib_path = H.get_localFilePath()
        if not grib_path.exists():
            H.download()

        # Read all variables from the GRIB2 file
        grbs = pygrib.open(str(grib_path))
        data_dict = {"datetime": run_time}

        # Iterate over all messages (variables) in the grib file
        for grb in grbs:
            short_name = grb.shortName
            level = grb.level

            # Create a unique column name for each variable+level
            col_name = f"{short_name}_{level}"

            # Calculate mean value over grid points (you can change this to other aggregates)
            values_mean = grb.values.mean()
            data_dict[col_name] = values_mean

        grbs.close()
        all_data.append(data_dict)

    except Exception as e:
        print(f"Error downloading/converting {run_time}: {e}")

# Combine all hourly data into a DataFrame
df = pd.DataFrame(all_data)

# Save the DataFrame to CSV in the script directory
df.to_csv(output_csv, index=False)
print(f"\n✅ Download complete. CSV saved at:\n{output_csv}")
