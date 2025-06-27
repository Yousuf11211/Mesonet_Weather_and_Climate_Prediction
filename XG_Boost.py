import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

# --- CONFIG ---
input_folder = 'filled_timestamps'
output_folder = 'xgboost'
os.makedirs(output_folder, exist_ok=True)

gap_threshold = timedelta(days=30)

all_vars = [
    'TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD',
    'WDSD', 'WSSD', 'SM02', 'SM04', 'ST02', 'ST04', 'VT05', 'VT20', 'VT90',
    'VR05', 'VR20', 'VR90'
]
targets = ['VT20', 'VT90']
correct_order = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected'] + all_vars

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

def fill_gaps(df_in, allow_large=True):
    df = df_in.copy()
    for target in targets:
        print(f"[FILLING] {target} using XGBoost (allow_large={allow_large})")
        features = [col for col in all_vars if col != target]
        complete_data = df[features + [target]].dropna()
        if complete_data.empty:
            continue
        X_train = complete_data[features]
        y_train = complete_data[target]
        imp = SimpleImputer(strategy='mean')
        X_train_imp = imp.fit_transform(X_train)
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                             tree_method='hist', verbosity=0, random_state=42)
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

def plot_feature_importance(df, site):
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
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                             tree_method='hist', verbosity=0, random_state=42)
        model.fit(X_imp, y)
        importance_dict[target] = model.feature_importances_

    if len(importance_dict) == 2:
        x = np.arange(len(features))
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width/2, importance_dict['VT20'], width, label='VT20')
        ax.bar(x + width/2, importance_dict['VT90'], width, label='VT90')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel("Importance")
        ax.set_title(f"XGBoost Feature Importance – {site}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{site}_feature_importance.png"))
        plt.close()

# --- MAIN LOOP ---
for file in os.listdir(input_folder):
    if not file.endswith('.csv'):
        continue

    print(f"\n📁 Processing: {file}")
    df = pd.read_csv(os.path.join(input_folder, file), parse_dates=['UTCTimestampCollected'])
    site = df['NetSiteAbbrev'].dropna().unique()[0]
    df.set_index('UTCTimestampCollected', inplace=True)
    df.sort_index(inplace=True)

    df_case1 = fill_gaps(df, allow_large=True)
    df_case2 = fill_gaps(df, allow_large=False)

    df_case1.reset_index(inplace=True)
    df_case2.reset_index(inplace=True)

    df_case1 = df_case1[correct_order]
    df_case2 = df_case2[correct_order]

    out1 = os.path.join(output_folder, f"{site}_case1.csv")
    out2 = os.path.join(output_folder, f"{site}_case2.csv")

    df_case1.to_csv(out1, index=False, float_format="%.4f")
    df_case2.to_csv(out2, index=False, float_format="%.4f")

    plot_feature_importance(df_case1, site)

    print(f"✅ Saved: {out1}")
    print(f"✅ Saved: {out2}")
    print(f"📊 Saved: {site}_feature_importance.png")
