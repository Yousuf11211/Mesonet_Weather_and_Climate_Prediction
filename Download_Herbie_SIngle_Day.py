import pandas as pd
from herbie import Herbie
from datetime import datetime

herbie = Herbie(date=datetime(2025, 1, 29))  # specify date here

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

print("Downloading HRRR data for specified parameters at 2025-01-29...")

row = {"timestamp": datetime.utcnow()}  # record download time
missing_params = []
data_found = {}

for pname in parameter_names:
    value = None
    try:
        data = herbie.get(
            model='hrrr',
            variables=[pname],
            spatial_points=[(38.0, -97.0)],
            verbose=False
        )
        if pname in data and data[pname]:
            value = data[pname][0]
    except Exception:
        pass

    if value is None:
        missing_params.append(pname)
        data_found[pname] = False
    else:
        data_found[pname] = True
    row[pname] = value

df = pd.DataFrame([row])
df.to_csv("HRRR_named_params_2025-01-29.csv", index=False)

with open("HRRR_named_params_2025-01-29_report.txt", "w") as f:
    f.write(f"HRRR Data Download Report for 2025-01-29\n")
    f.write(f"Download Timestamp (UTC): {row['timestamp']}\n")
    f.write(f"Total Parameters Requested: {len(parameter_names)}\n")
    f.write(f"Parameters Successfully Downloaded: {sum(data_found.values())}\n")
    f.write(f"Parameters Missing:\n")
    for p in missing_params:
        f.write(f" - {p}\n")

print("Download complete.")
print(f"Data saved to 'HRRR_named_params_2025-01-29.csv'")
print(f"Report saved to 'HRRR_named_params_2025-01-29_report.txt'")
