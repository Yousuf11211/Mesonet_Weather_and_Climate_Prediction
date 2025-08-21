#needs conda environment
from herbie import Herbie
import pygrib
import pandas as pd
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count

date_to_download = datetime(2025, 1, 29)
output_dir = Path(__file__).parent
temp_dir = output_dir / "HRRR_Downloads"
temp_dir.mkdir(exist_ok=True)

hourly_csv = output_dir / f"HRRR_Surface_Hourly_{date_to_download.strftime('%Y%m%d')}.csv"
interp_csv = output_dir / f"HRRR_Surface_5min_Interpolated_{date_to_download.strftime('%Y%m%d')}.csv"

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
        human_names = {"datetime": "Datetime"}
        tech_names = {"datetime": "datetime"}

        for grb in grbs:
            short_name = grb.shortName
            human_name = grb.name.replace(" ", "_")
            level = grb.level

            col = f"{short_name}_{level}"

            mean_val = grb.values.mean()
            data[col] = mean_val
            human_names[col] = f"{human_name}_{level}"
            tech_names[col] = col

        grbs.close()
        return pd.DataFrame([data]), human_names, tech_names

    except Exception as e:
        print(f"Error downloading hour {hour}: {e}")
        return None

if __name__ == "__main__":
    hours = list(range(24))

    with Pool(cpu_count()) as pool:
        results = pool.map(download_hour, hours)

    valid_results = [(r[0], r[1], r[2]) for r in results if r is not None]

    if valid_results:
        dfs = [r[0] for r in valid_results]
        df = pd.concat(dfs, ignore_index=True)
        df.sort_values("datetime", inplace=True)

#gets human and tech names for easy understanding
        human_names = valid_results[0][1]
        tech_names = valid_results[0][2]

# Two headers
        with open(hourly_csv, "w", newline="") as f:
            f.write(",".join(human_names[col] for col in df.columns) + "\n")
            f.write(",".join(tech_names[col] for col in df.columns) + "\n")
            df.to_csv(f, index=False, header=False)

        print(f"Hourly CSV saved: {hourly_csv}")

        df.set_index("datetime", inplace=True)
        df_5min = df.resample("5T").interpolate(method="linear")
        with open(interp_csv, "w", newline="") as f:
            f.write(",".join(human_names[col] for col in df_5min.columns.insert(0, 'datetime')) + "\n")
            f.write(",".join(tech_names[col] for col in df_5min.columns.insert(0, 'datetime')) + "\n")
            df_5min.to_csv(f, index=True, header=False)

        print(f"Interpolated 5-min CSV saved: {interp_csv}")
    else:
        print("No data downloaded successfully.")
