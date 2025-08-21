import os
import re
import pandas as pd

site_name = "ELST"
linear_log_path = f'RandomForest_Regression/{site_name}.txt'
rf_log_path = f'RandomForest_interpolation/{site_name}.txt'
output_folder = f'Random_Forest_Difference_Logs/{site_name}'
os.makedirs(output_folder, exist_ok=True)

def parse_log(file_path):
    pattern = r"(\w+)\s+filled at\s+([\d\-:\s]+)\s+\|\s+previous:\s+\S+\s+→\s+new:\s+(\d+\.\d+)"
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                col, timestamp, value = match.groups()
                data.append((col, pd.to_datetime(timestamp.strip()), float(value)))
    return pd.DataFrame(data, columns=['column', 'timestamp', 'value'])

df_linear = parse_log(linear_log_path)
df_rf = parse_log(rf_log_path)

df_merged = pd.merge(
    df_linear,
    df_rf,
    on=['column', 'timestamp'],
    suffixes=('_linear', '_rf')
)

df_merged['diff'] = (df_merged['value_linear'] - df_merged['value_rf']).abs()

for component, group in df_merged.groupby('column'):
    file_path = os.path.join(output_folder, f"{component}_diff.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"Difference Report for {component} ({site_name})\n\n")
        for _, row in group.iterrows():
            f.write(f"{component} filled at {row['timestamp']} | Difference → {row['diff']:.4f}\n")
    print(f"Saved: {file_path}")
