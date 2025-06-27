import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

# --- CONFIG ---
input_folder = 'filled_timestamps'  # CHANGE THIS if needed
output_folder = 'decision_tree'
gap_threshold = timedelta(days=30)
target_cols = ['VT20', 'VT90']
meta_cols = ['NetSiteAbbrev', 'County', 'UTCTimestampCollected']
all_columns = meta_cols + [
    'TAIR', 'DWPT', 'PRCP', 'PRES', 'RELH', 'SRAD', 'WDIR', 'WSPD', 'WDSD', 'WSSD',
    'SM02', 'SM04', 'ST02', 'ST04', 'VT05', 'VT20', 'VT90', 'VR05', 'VR20', 'VR90'
]

# --- HELPERS ---
def find_gaps(series, timestamps):
    gaps = []
    start = None
    for i in range(len(series)):
        if pd.isna(series.iloc[i]):
            if start is None:
                start = i
        else:
            if start is not None:
                gaps.append((start, i - 1, timestamps[i - 1] - timestamps[start]))
                start = None
    if start is not None:
        gaps.append((start, len(series) - 1, timestamps[len(series) - 1] - timestamps[start]))
    return gaps

def plot_feature_importance(importances, features, out_path):
    x = np.arange(len(features))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (target, importance) in enumerate(importances.items()):
        ax.bar(x + i * width, importance, width, label=target)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_ylabel("Normalized Importance")
    ax.set_title("Feature Importance for VT20 and VT90 (Decision Tree)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --- MAIN ---
os.makedirs(output_folder, exist_ok=True)
input_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file in input_files:
    df = pd.read_csv(os.path.join(input_folder, file), parse_dates=['UTCTimestampCollected'])
    df.sort_values('UTCTimestampCollected', inplace=True)
    site = df['NetSiteAbbrev'].iloc[0]
    df_full = df.copy()
    features = [col for col in all_columns if col not in meta_cols + target_cols]
    feature_importance = {}

    for case_name, leave_large_empty in zip(['case1', 'case2'], [False, True]):
        df_case = df_full.copy()
        for target in target_cols:
            temp_df = df_case[features + [target]].dropna()
            if temp_df.empty:
                continue
            X_train = temp_df[features]
            y_train = temp_df[target]
            imp = SimpleImputer(strategy='mean')
            X_train_imp = imp.fit_transform(X_train)
            model = DecisionTreeRegressor(max_depth=10, random_state=42)
            model.fit(X_train_imp, y_train)

            if not leave_large_empty:
                feature_importance[target] = model.feature_importances_ / np.sum(model.feature_importances_)

            gaps = find_gaps(df_case[target], df_case['UTCTimestampCollected'])
            for start, end, duration in gaps:
                if leave_large_empty and duration > gap_threshold:
                    continue
                X_gap = df_case.iloc[start:end + 1][features]
                X_gap_imp = imp.transform(X_gap)
                preds = model.predict(X_gap_imp)
                df_case.loc[start:end, target] = preds

        df_case = df_case[all_columns]
        df_case.to_csv(os.path.join(output_folder, f"{site}_{case_name}.csv"),
                       index=False, float_format="%.4f")

    # Save feature importance only for case1
    plot_feature_importance(feature_importance, features,
                            os.path.join(output_folder, f"{site}_feature_importance.png"))
