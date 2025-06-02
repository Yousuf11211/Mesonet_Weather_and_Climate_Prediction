import pandas as pd
import os

input_folder = 'Original_data'
soil_folder = 'with_soil'
no_soil_folder = 'without_soil'
os.makedirs(soil_folder, exist_ok=True)
os.makedirs(no_soil_folder, exist_ok=True)

required_with_soil = ['TAIR', 'VT90', 'SM02', 'PRES']
required_without_soil = ['TAIR', 'VT90', 'PRES']

desired_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected', 'TAIR', 'DWPT', 'PRCP', 'PRES',
    'RELH', 'SRAD', 'WDIR', 'WSPD', 'WDSD', 'WSSD', 'SM02', 'SM04', 'ST02', 'ST04',
    'VT05', 'VT20', 'VT90', 'VR05', 'VR20', 'VR90'
]

for filename in os.listdir(input_folder):
    if not filename.endswith('.csv'):
        continue

    file_path = os.path.join(input_folder, filename)

    try:
        print(f"\n📁 Processing: {filename}")
        df = pd.read_csv(file_path, low_memory=False)

        # Remove 2nd-row human-readable header if detected
        second_row = df.iloc[0]
        if any(keyword in str(second_row.values).lower() for keyword in ['temperature', 'humidity', 'pressure']):
            print("⚠️ Found human-readable label row. Removing it.")
            df = df.iloc[1:].reset_index(drop=True)

        # Parse timestamps
        df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
        junk_rows = df[df['UTCTimestampCollected'].isnull()]
        if not junk_rows.empty:
            print(f"⚠️ Found {len(junk_rows)} junk rows with invalid timestamps:")
            print(junk_rows.to_string(index=False))

        df = df.dropna(subset=['UTCTimestampCollected'])

        # Drop duplicate header rows
        if 'NetSiteAbbrev' in df.columns:
            junk_headers = df[df['NetSiteAbbrev'] == "Station ID"]
            if not junk_headers.empty:
                print(f"⚠️ Found {len(junk_headers)} repeated header rows:")
                print(junk_headers.to_string(index=False))
            df = df[df['NetSiteAbbrev'] != "Station ID"]

        # Keep all remaining rows, including NaNs
        df = df.sort_values('UTCTimestampCollected')
        df.set_index('UTCTimestampCollected', inplace=True)

        site_name = df['NetSiteAbbrev'].dropna().iloc[0] if 'NetSiteAbbrev' in df.columns else filename.replace(".csv", "")

        # -------- WITH SOIL --------
        valid_soil_rows = df[required_with_soil].notna().all(axis=1)
        if valid_soil_rows.any():
            start_soil = df.index[valid_soil_rows].min()
            rows_before = df[df.index < start_soil]
            print(f"📌 [WITH SOIL] First complete data: {start_soil} | Rows to delete: {len(rows_before)}")
            choice = input("Delete rows before this for WITH SOIL? (y/n): ").strip().lower()
            if choice == 'y':
                filtered_df = df.loc[start_soil:].copy().reset_index()
                filtered_df = filtered_df[[col for col in desired_order if col in filtered_df.columns]]
                output_path = os.path.join(soil_folder, f"{site_name}.csv")
                filtered_df.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path}")
            else:
                print("⏭ Skipped saving for WITH SOIL.")
        else:
            print("⚠ No complete row found for all 4 variables in WITH SOIL case.")

        # -------- WITHOUT SOIL --------
        valid_nosoil_rows = df[required_without_soil].notna().all(axis=1)
        if valid_nosoil_rows.any():
            start_no_soil = df.index[valid_nosoil_rows].min()
            rows_before = df[df.index < start_no_soil]
            print(f"📌 [WITHOUT SOIL] First complete data: {start_no_soil} | Rows to delete: {len(rows_before)}")
            choice = input("Delete rows before this for WITHOUT SOIL? (y/n): ").strip().lower()
            if choice == 'y':
                filtered_df = df.loc[start_no_soil:].copy().reset_index()

                # Drop soil-related columns
                soil_columns = ['SM02', 'SM04', 'ST02', 'ST04']
                filtered_df = filtered_df.drop(columns=[col for col in soil_columns if col in filtered_df.columns])

                columns_to_keep = [col for col in desired_order if col in filtered_df.columns]
                filtered_df = filtered_df[columns_to_keep]

                output_path = os.path.join(no_soil_folder, f"{site_name}.csv")
                filtered_df.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path} (without soil columns)")
            else:
                print("⏭ Skipped saving for WITHOUT SOIL.")
        else:
            print("⚠ No complete row found for TAIR, VT90, PRES in WITHOUT SOIL case.")

    except Exception as e:
        print(f"❌ Failed to process {filename}. Reason: {type(e).__name__} - {e}")
