import os
import re
import pandas as pd

# --- CONFIG ---
site_name = "ELST"
linear_log_path = f'Linear_interpolation/{site_name}.txt'
rf_log_path = f'RandomForest_interpolation/{site_name}.txt'
output_diff_path = f'Difference_logs/Difference_{site_name}.txt'
os.makedirs('Difference_logs', exist_ok=True)

# --- Parse logs ---
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

# --- Load logs ---
df_linear = parse_log(linear_log_path)
df_rf = parse_log(rf_log_path)

# --- Merge on timestamp + column ---
df_merged = pd.merge(
    df_linear,
    df_rf,
    on=['column', 'timestamp'],
    suffixes=('_linear', '_rf')
)

# --- Compute difference ---
df_merged['diff'] = (df_merged['value_linear'] - df_merged['value_rf']).abs()

# --- Save clean difference report ---
with open(output_diff_path, 'w', encoding='utf-8') as f:
    f.write(f"Difference Report for {site_name}\n\n")
    for _, row in df_merged.iterrows():
        f.write(f"{row['column']} filled at {row['timestamp']} | Difference → {row['diff']:.4f}\n")

print(f"✅ Saved difference-only report to: {output_diff_path}")
