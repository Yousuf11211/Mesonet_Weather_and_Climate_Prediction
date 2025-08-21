import xarray as xr
import os

# List of levels you want CSVs for
levels = [
    'atmosphere', 'cloudTop', 'surface', 'heightAboveGround', 'isothermal', 'isobaricInhPa',
    'pressureFromGroundLayer', 'sigmaLayer', 'meanSea', 'heightAboveGroundLayer',
    'sigma', 'atmosphereSingleLayer', 'depthBelowLand'
]

# Paths for your two GRIB files (update these)
grib_files = [
    r"E:\Herbie\HRRR_Default\hrrr\20200101\hrrr.t00z.wrfsfcf00.grib2",
    r"E:\Herbie\HRRR_Default\hrrr\20200102\hrrr.t00z.wrfsfcf00.grib2"
]

output_base_dir = r"E:\Herbie\HRRR_Default\CSV_Converted"
os.makedirs(output_base_dir, exist_ok=True)

for grib_file in grib_files:
    day_str = os.path.basename(os.path.dirname(grib_file))  # e.g., "20200101"
    output_dir = os.path.join(output_base_dir, day_str)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {grib_file} ...")
    for level in levels:
        try:
            ds = xr.open_dataset(grib_file, engine='cfgrib', filter_by_keys={'typeOfLevel': level})
            df = ds.to_dataframe().reset_index()
            output_path = os.path.join(output_dir, f"{level}.csv")
            df.to_csv(output_path, index=False)
            print(f"  Saved CSV for level '{level}' at {output_path}")
        except Exception as e:
            print(f"  Could not process level '{level}': {e}")

print("Done converting 2 days of GRIB files to CSV.")
