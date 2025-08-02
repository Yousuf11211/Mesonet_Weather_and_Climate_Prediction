from herbie import Herbie
import pygrib
import pandas as pd
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ---------------- CONFIG ----------------
date_to_download = datetime(2025, 1, 29)
output_dir = Path(__file__).parent
temp_dir = output_dir / "HRRR_Downloads"
temp_dir.mkdir(exist_ok=True)

hourly_csv = output_dir / f"HRRR_Surface_Hourly_{date_to_download.strftime('%Y%m%d')}.csv"
interp_csv = output_dir / f"HRRR_Surface_5min_Interpolated_{date_to_download.strftime('%Y%m%d')}.csv"

# ---------------- FUNCTION ----------------
def download_hour(hour):
    run_time = date_to_download.replace(hour=hour)
    print(f"Downloading HRRR data for {run_time}...")

    try:
        H = Herbie(
            run_time.strftime("%Y-%m-%d %H:%M"),
            model="hrrr",
            product="sfc",
            fxx=0,
            save_dir=temp_dir
        )

        grib_path = H.get_localFilePath()
        if not grib_path or not Path(grib_path).exists():
            print(f"File missing → downloading now...")
            H.download()
            grib_path = H.get_localFilePath()

        if not grib_path or not Path(grib_path).exists():
            print(f"No file for hour {hour}")
            return None

        grbs = pygrib.open(str(grib_path))
        data = {"datetime": run_time}

        for grb in grbs:
            short_name = grb.shortName
            human_name = grb.name.replace(" ", "_")
            level = grb.level

            tech_col = f"{short_name}_{level}"
            human_col = f"{human_name}_{level}"

            mean_val = grb.values.mean()
            data[tech_col] = mean_val
            data[human_col] = mean_val

        grbs.close()
        return pd.DataFrame([data])

    except Exception as e:
        print(f"Error downloading hour {hour}: {e}")
        return None

# ---------------- MAIN ----------------
if __name__ == "__main__":
    hours = list(range(24))

    with Pool(cpu_count()) as pool:
        results = pool.map(download_hour, hours)

    valid_results = [r for r in results if r is not None and not r.empty]

    if valid_results:
        df = pd.concat(valid_results, ignore_index=True)
        df.sort_values("datetime", inplace=True)
        df.to_csv(hourly_csv, index=False)
        print(f"Hourly CSV saved: {hourly_csv}")

        df.set_index("datetime", inplace=True)
        df_5min = df.resample("5T").interpolate(method="linear")
        df_5min.to_csv(interp_csv)
        print(f"Interpolated 5-min CSV saved: {interp_csv}")
    else:
        print("No data downloaded successfully.")
