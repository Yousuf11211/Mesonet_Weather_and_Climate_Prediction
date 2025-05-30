import pandas as pd
import os

# Folder containing your CSV files
input_folder = "Dummyfull_data"  # Change if your folder has a different name

# Soil columns of interest
soil_columns = ["SM02", "ST02"]

# Scan each CSV file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(filepath, parse_dates=["UTCTimestampCollected"], low_memory=False)

            print(f"\n{filename} Summary:")

            # Get full date range (regardless of soil)
            full_start = df["UTCTimestampCollected"].min()
            full_end = df["UTCTimestampCollected"].max()
            print(f"Without Soil Data: {full_start.date()} → {full_end.date()}")

            # Check for both SM02 and ST02
            if all(col in df.columns for col in soil_columns):
                soil_df = df[df[soil_columns].notna().all(axis=1)]

                if not soil_df.empty:
                    soil_start = soil_df["UTCTimestampCollected"].min()
                    soil_end = soil_df["UTCTimestampCollected"].max()
                    print(f"With Soil Data:    {soil_start.date()} → {soil_end.date()}")
                else:
                    print("With Soil Data:    No rows with both SM02 and ST02.")
            else:
                print("Missing SM02 and/or ST02 columns.")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
