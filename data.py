# import pandas as pd
# import glob
# import time
#
# pd.set_option('display.max_columns', None)
#
# file_paths = glob.glob('mesonetdata/*.csv')
# print(f"Found {len(file_paths)} CSV files.")
#
# all_dataframes = []
#
# start_all = time.time()
#
# for file_path in file_paths:
#     start = time.time()
#     try:
#         df = pd.read_csv(file_path, header=0, skiprows=[1], low_memory=False)
#         df.columns = df.columns.astype(str)
#         df.columns.name = None
#         df['SourceFile'] = file_path
#         all_dataframes.append(df)
#
#         end = time.time()
#         print(f"Read {file_path} in {end - start:.2f} seconds", flush=True)
#
#     except Exception as e:
#         print(f" Failed to read {file_path}: {e}", flush=True)
#
# end_all = time.time()
# total_duration = end_all - start_all
# print(f"\n Total time to read all files: {total_duration:.2f} seconds")
#
# if all_dataframes:
#     combined_df = pd.concat(all_dataframes, ignore_index=True)
#     combined_df['UTCTimestampCollected'] = pd.to_datetime(combined_df['UTCTimestampCollected'], errors='coerce')
#     combined_df = combined_df.dropna(subset=['UTCTimestampCollected'])
#     combined_df['Year'] = combined_df['UTCTimestampCollected'].dt.year
#
#     year_counts = combined_df['Year'].value_counts().sort_index()
#
#     print("\nRows per year:")
#     for year, count in year_counts.items():
#         print(f"{int(year)} → {count:,} rows")
#
#     print(combined_df.head())
#     print(f"\nTotal rows: {len(combined_df)}")
#     print(f"Total columns: {len(combined_df.columns)}")
# else:
#     print("No dataframes to combine.")


import pandas as pd
import glob
import time
from collections import defaultdict
import os

pd.set_option('display.max_columns', None)

file_paths = glob.glob('mesonetdata/*.csv')
print(f"Found {len(file_paths)} CSV files.\n")

year_counts = defaultdict(int)
year_sources = defaultdict(set)
all_dataframes = []

start_all = time.time()

for file_path in file_paths:
    start = time.time()
    try:
        df = pd.read_csv(file_path, header=0, skiprows=[1], low_memory=False)
        df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
        df = df.dropna(subset=['UTCTimestampCollected'])

        years = df['UTCTimestampCollected'].dt.year
        for year in years:
            year_counts[year] += 1
            year_sources[year].add(file_path)

        df = df.iloc[:, :19]

        all_dataframes.append(df)

        end = time.time()
        print(f"Read {file_path} in {end - start:.2f} seconds", flush=True)

    except Exception as e:
        print(f"Failed to read {file_path}: {e}", flush=True)

end_all = time.time()
print(f"\nTotal time: {end_all - start_all:.2f} seconds")

# Year summary
print("\nRows per year with contributing files:")
for year in sorted(year_counts.keys()):
    file_list = ', '.join(
        sorted(os.path.splitext(os.path.basename(f))[0] for f in year_sources[year])
    )
    print(f"{int(year)} → {year_counts[year]:,} rows (from: {file_list})")

if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print("\nCombined table preview (original 19 columns):")
    print(combined_df.head())
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
else:
    print("⚠No dataframes to combine.")
