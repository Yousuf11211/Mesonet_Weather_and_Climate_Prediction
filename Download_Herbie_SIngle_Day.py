import pandas as pd
from herbie import Herbie
from datetime import datetime, timedelta

herbie = Herbie(date=datetime(2025, 1, 29))

parameter_names = [
    "Visibility_surface",
    "WindGust_surface",
    "HGT:700 mb",
    "TMP:925 mb",
    "PRMSL_surface",
    "HGT:1000 mb",
    "PRES_surface",
    "TMP_surface",
    "MOIST_ground",
    "TMP:2 m",
    "POT:2 m",
    "SPFH:2 m",
    "DPT:2 m",
    "RH:2 m",
    "WIND:10 m",
    "FRICV_surface",
    "SHTFL_surface",
    "LHTFL_surface",
    "LFTX_surface",
    "LCDC_surface",
    "SWDOWN_surface",
    "LWDOWN_surface",
    "SWUP_surface",
    "LWUP_surface",
    "VIS:surface",
    "VIS_DIF_surface",
    "SRHE:3000 m",
    "USHR_surface",
    "IZP_surface",
    "HTFR_surface"
]

start_time = datetime(2025, 1, 29, 0, 0)
end_time = datetime(2025, 1, 29, 23, 55)

time_list = []
cur_time = start_time
while cur_time <= end_time:
    time_list.append(cur_time)
    cur_time += timedelta(minutes=5)

records = []
missing_counts = {pname: 0 for pname in parameter_names}
data_found = {pname: False for pname in parameter_names}

print(f"Downloading HRRR data for {len(time_list)} timestamps...")

for dt in time_list:
    row = {"timestamp": dt}
    for pname in parameter_names:
        value = None
        try:
            data = herbie.get(
                model='hrrr',
                variables=[pname],
                date=dt,
                spatial_points=[(38.0, -97.0)],
                verbose=False
            )
            if pname in data and data[pname]:
                value = data[pname][0]
        except Exception:
            pass

        if value is None:
            missing_counts[pname] += 1
        else:
            data_found[pname] = True

        row[pname] = value
    records.append(row)

# Save main data CSV
df = pd.DataFrame(records)
data_csv = "HRRR_Jan29_2025_named_params_5min.csv"
df.to_csv(data_csv, index=False)

# Write detailed TXT report
report_txt = "HRRR_Jan29_2025_download_report.txt"
total_timestamps = len(time_list)

with open(report_txt, "w") as f:
    f.write(f"HRRR Download Report for {start_time.date()}\n")
    f.write(f"Total timestamps (5-min intervals): {total_timestamps}\n\n")
    for pname in parameter_names:
        present = total_timestamps - missing_counts[pname]
        f.write(f"Parameter: {pname}\n")
        f.write(f"  Timestamps with data   : {present}\n")
        f.write(f"  Timestamps missing data: {missing_counts[pname]}\n")
        f.write(f"  Data found at least once: {'Yes' if data_found[pname] else 'No'}\n")
        f.write("-" * 40 + "\n")

print(f"Download complete. Data saved to '{data_csv}'.")
print(f"Report saved as text file '{report_txt}'.")
