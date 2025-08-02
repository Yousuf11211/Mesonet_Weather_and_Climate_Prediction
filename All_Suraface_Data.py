from herbie import Herbie
import pygrib
import pandas as pd
from datetime import datetime
from pathlib import Path

date_to_download = datetime(2025, 1, 29)  # Change date here
output_csv = Path(__file__).parent / f"HRRR_Surface_5min_AllVars_{date_to_download.strftime('%Y%m%d')}.csv"

forecast_steps = list(range(0, 60, 5))  # every 5 minutes (0,5,10,...55)

all_times_data = []  # collect all data here

for hour in range(24):
    run_time = date_to_download.replace(hour=hour, minute=0)
    print(f"Processing run time: {run_time}")

    for fxx_min in forecast_steps:
        try:
            H = Herbie(
                run_time.strftime("%Y-%m-%d %H:%M"),
                model="hrrr",
                product="sfc",
                fxx=fxx_min,
                save_dir=Path(__file__).parent / "HRRR_Downloads"
            )

            grib_path = H.get_localFilePath()
            if not grib_path.exists():
                H.download()

            grbs = pygrib.open(str(grib_path))

            dfs = []
            lat_lon_df = None

            for grb in grbs:
                short_name = grb.shortName
                level = grb.level
                col_name = f"{short_name}_{level}"

                lats, lons = grb.latlons()
                values = grb.values

                # Create lat/lon DataFrame once
                if lat_lon_df is None:
                    lat_lon_df = pd.DataFrame({
                        "latitude": lats.flatten(),
                        "longitude": lons.flatten()
                    })

                # Create DataFrame for this variable's values (flattened)
                var_df = pd.DataFrame(values.flatten(), columns=[col_name])

                dfs.append(var_df)

            grbs.close()

            # Concatenate all variable columns horizontally
            vars_df = pd.concat(dfs, axis=1)

            # Combine lat/lon and vars
            full_df = pd.concat([lat_lon_df, vars_df], axis=1)
            full_df['datetime'] = run_time + pd.Timedelta(minutes=fxx_min)

            all_times_data.append(full_df)

        except Exception as e:
            print(f"Error processing {run_time} fxx={fxx_min}: {e}")

# Combine all times vertically
final_df = pd.concat(all_times_data, ignore_index=True)

# Save to one big CSV
final_df.to_csv(output_csv, index=False)

print(f"\n✅ Download complete. All data saved in one CSV at:\n{output_csv}")
