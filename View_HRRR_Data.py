from herbie import Herbie

# Load from a specific date/time and forecast hour
H = Herbie("2020-01-01 00:00", model="hrrr", product="sfc", fxx=0, save_dir="D:/Herbie/HRRR_Default")

# Load full dataset using xarray
ds = H.xarray()

# See basic info
print(ds)

# See list of available variables
print(ds.data_vars)

# For example, view 2-meter temperature
print(ds['TMP:2 m above ground'])

# Plot a quick map
ds['TMP:2 m above ground'].plot()
