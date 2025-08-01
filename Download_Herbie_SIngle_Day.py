import pandas as pd
from herbie import Herbie
from datetime import datetime, timedelta

herbie = Herbie()

# Your parameter IDs
param_ids = [
    "4", "8", "18", "28", "32", "40", "47", "59", "61", "66", "67", "68", "69", "70",
    "73", "89", "90", "94", "98", "108", "109", "110", "111", "112", "113", "114",
    "115", "116", "127", "128"
]

# Map IDs to parameter names (fill with actual HRRR variable names)
id_to_name = {
    "4": "Visibility_surface",
    "8": "WindGust_surface",
    "18": "HGT:700 mb",
    "28": "TMP:925 mb",
    "32": "PRMSL_surface",
    "40": "HGT:1000 mb",
    "47": "PRES_surface",
    "59": "TMP_surface",
    "61": "MOIST_ground",
    "66": "TMP:2 m",
    "67": "POT:2 m",
    "68": "SPFH:2 m",
    "69": "DPT:2 m",
    "70": "RH:2 m",
    "73": "WIND:10 m",
    "89": "FRICV_surface",
    "90": "SHTFL_surface",
    "94": "LHTFL_surface",
    "98": "LFTX_surface",
    "108": "LCDC_surface",
    "109": "SWDOWN_surface",
    "110": "LWDOWN_surface",
    "111": "SWUP_surface",
    "112": "LWUP_surface",
    "113": "VIS:surface",
    "114": "VIS_DIF_surface",
    "115": "SRHE:3000 m",
    "116": "USHR_surface",
    "127": "IZP_surface",
    "128": "HTFR_surface"
}

start_time = datetime(2025, 1, 29, 0, 0)
end_time = datetime(2025, 1, 29, 23, 55)

time_list = []
cur_time = start_time
while cur_time <= end_time:
    time_list.append(cur_time)
    cur_time += timedelta(minutes=5)

records = []

# Track missing counts per param
missing_counts = {pid: 0 for pid in param_ids}
# Track if any data found for param at all
data_found = {pid: False for pid in param_ids}

print(f"Downloading HRRR data for {len(time_list)} timestamps...")

for dt in time_list:
    row = {"timestamp": dt}
    for pid in param_ids:
        value = None
        # First try numeric ID string
        try:
            data = herbie.get(
                model='hrrr',
                variables=[pid],
                date=dt,
                spatial_points=[(38.0, -97.0)],
                verbose=False
            )
            if pid in data and data[pid]:
                value = data[pid][0]
            else:
                raise ValueError("No data for numeric ID")
        except Exception:
            # Try parameter name fallback
            param_name = id_to_name.get(pid)
            if param_name:
                try:
                    data2 = herbie.get(
                        model='hrrr',
                        variables=[param_name],
                        date=dt,
                        spatial_points=[(38.0, -97.0)],
                        verbose=False
                    )
                    if param_name in data2 and data2[param_name]:
                        value = data2[param_name][0]
                except Exception:
                    pass
        if value is None:
            missing_counts[pid] += 1
        else:
            data_found[pid] = True
        row[pid] = value
    records.append(row)

df = pd.DataFrame(records)
output_csv = "HRRR_Jan29_2025_fallback.csv"
df.to_csv(output_csv, index=False)

# Create detailed report
total_timestamps = len(time_list)
report_rows = []
for pid in param_ids:
    found = data_found[pid]
    missing = missing_counts[pid]
    present = total_timestamps - missing
    report_rows.append({
        "Parameter_ID": pid,
        "Parameter_Name": id_to_name.get(pid, "N/A"),
        "Total_Timestamps": total_timestamps,
        "Timestamps_With_Data": present,
        "Timestamps_Missing": missing,
        "Data_Found_At_Least_Once": found
    })

report_df = pd.DataFrame(report_rows)
report_csv = "HRRR_Jan29_2025_download_report.csv"
report_df.to_csv(report_csv, index=False)

print(f"Download complete. Data saved to '{output_csv}'.")
print(f"Report saved to '{report_csv}'.")
