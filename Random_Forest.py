import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# --- CONFIG ---
input_file = 'filled_timestamps/ELST.csv'
case1_output = 'output_case1.csv'
case2_output = 'output_case2.csv'
plot_file = 'Random_Forest/feature_importance.png'
gap_threshold = timedelta(days=30)

# --- Columns ---
all_vars = [
    'TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD',
    'WDSD', 'WSSD', 'SM02', 'SM04', 'ST02', 'ST04', 'VT05', 'VT20', 'VT90',
    'VR05', 'VR20', 'VR90'
]
targets = ['VT20', 'VT90']
correct_order = [
    'NetSiteAbbrev', 'County', 'UTCTimestampCollected'
] + all_vars

# --- Load ---
df = pd.read_csv(input_file, parse_dates=['UTCTimestampCollected'])
df.set_index('UTCTimestampCollected', inplace=True)
df.sort_index(inplace=True)

# --- Utilities ---
def find_gaps(series):
    gaps = []
    is_nan = series.isna()
    start = None
    for i in range(len(series)):
        if is_nan.iloc[i] and start is None:
            start = i
        elif not is_nan.iloc[i] and start is not None:
            gaps.append((start, i - 1))
            start = None
    if start is not None:
        gaps.append((start, len(series) - 1))
    return gaps

# --- Model Filling ---
def fill_gaps(df_in, allow_large=True):
    df = df_in.copy()
    for target in targets:
        print(f"[FILLING] {target} using RandomForest (allow_large={allow_large})")
        features = [col for col in all_vars if col != target]
        complete_data = df[features + [target]].dropna()
        if complete_data.empty:
            continue
        X_train = complete_data[features]
        y_train = complete_data[target]
        imp = SimpleImputer(strategy='mean')
        X_train_imp = imp.fit_transform(X_train)
        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_train_imp, y_train)

        gaps = find_gaps(df[target])
        for start, end in gaps:
            duration = df.index[end] - df.index[start]
            if not allow_large and duration > gap_threshold:
                continue
            X_test = df.iloc[start:end+1][features]
            X_test_imp = imp.transform(X_test)
            preds = model.predict(X_test_imp)
            df.iloc[start:end+1, df.columns.get_loc(target)] = preds
    return df

# --- Feature Importance ---
def plot_feature_importance(df):
    importance_dict = {}
    for target in targets:
        features = [col for col in all_vars if col != target]
        data = df[features + [target]].dropna()
        if data.empty:
            continue
        X = data[features]
        y = data[target]
        imp = SimpleImputer(strategy='mean')
        X_imp = imp.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_imp, y)
        importances = model.feature_importances_
        importance_dict[target] = importances

    x = np.arange(len(features))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, importance_dict['VT20'], width, label='VT20')
    ax.bar(x + width/2, importance_dict['VT90'], width, label='VT90')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance for VT20 & VT90")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

# --- Run ---
df_case1 = fill_gaps(df, allow_large=True)
df_case2 = fill_gaps(df, allow_large=False)

df_case1.reset_index(inplace=True)
df_case2.reset_index(inplace=True)

df_case1 = df_case1[correct_order]
df_case2 = df_case2[correct_order]

df_case1.to_csv(case1_output, index=False, float_format="%.4f")
df_case2.to_csv(case2_output, index=False, float_format="%.4f")

plot_feature_importance(df_case1)

print("✅ Finished filling. Outputs saved:")
print(f"  - {case1_output}")
print(f"  - {case2_output}")
print(f"  - {plot_file}")
