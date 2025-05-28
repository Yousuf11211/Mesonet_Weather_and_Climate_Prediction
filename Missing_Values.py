import pandas as pd
import os

input_folder = "b"
output_folder = "Missing_vales_CSV"
os.makedirs(output_folder, exist_ok=True)


csv_files =[f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    print(f"\n Processing file: {file}")

    df = pd.read_csv(file_path)
    total_rows = len(df)

    missing_df = df[df.isna().any(axis=1)]
    missing_rows = len(missing_df)

    output_file = os.path.join(output_folder, f"missing_{file}")
    missing_df.to_csv(output_file, index=False)

    print(f"Total rows: {total_rows}")
    print(f"Missing rows: {missing_rows}")
    print(f"Saved missing rows to: {output_file}")