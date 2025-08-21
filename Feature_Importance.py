import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import shap
import numpy as np


def calculateFeatureImportance(df, target_col, output_dir, excluded_features=None):
    if excluded_features is None:
        excluded_features = ['VT90_TAIR_diff', 'VT90_VT20_diff', 'VT90', 'VT20', 'TAIR',
                             'SM04', 'ST04', 'UTCTimestampCollected', 'NetSiteAbbrev', 'County']

    # Prepare data
    X = df.drop(columns=[target_col] + [f for f in excluded_features if f in df.columns])
    y = df[target_col]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

#Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_imp = pd.Series(dt.feature_importances_, index=X.columns)

#Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    rf_imp = pd.Series(rf_perm.importances_mean, index=X.columns)

#SHAP
    explainer = shap.PermutationExplainer(rf.predict, X_test, n_jobs=-1)
    X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer(X_test_sample)

#Plot Decision Tree
    dt_plot = dt_imp.nlargest(10)
    plt.figure(figsize=(10, 6))
    dt_plot.plot.barh()
    plt.title(f"{target_col} - Decision Tree Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_col}_dt.png"))
    plt.close()

#Plot Random Forest
    rf_plot = rf_imp.nlargest(10)
    plt.figure(figsize=(10, 6))
    rf_plot.plot.barh()
    plt.title(f"{target_col} - Random Forest Importance")
    plt.xlabel("Mean Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_col}_rf.png"))
    plt.close()

#SHAP Summary Plot
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", max_display=10,
                      show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target_col}_shap.png"))
    plt.close()
#Save R² and RMSE
    dt_r2 = r2_score(y_test, y_pred_dt)
    rf_r2 = r2_score(y_test, y_pred_rf)
    dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    return {
        'target': target_col,
        'DecisionTree_R2': round(dt_r2, 4),
        'DecisionTree_RMSE': round(dt_rmse, 4),
        'RandomForest_R2': round(rf_r2, 4),
        'RandomForest_RMSE': round(rf_rmse, 4)
    }



input_folder = 'Processed Data'
output_folder = 'Feature_Importance_Results'
os.makedirs(output_folder, exist_ok=True)

summary_rows = []

for filename in os.listdir(input_folder):
    if not filename.endswith('.csv'):
        continue

    site_name = os.path.splitext(filename)[0]  # e.g., GRDR_interpolated
    print(f"\nProcessing site: {site_name}")
    site_output_folder = os.path.join(output_folder, site_name)
    os.makedirs(site_output_folder, exist_ok=True)

    file_path = os.path.join(input_folder, filename)
    try:
        df = pd.read_csv(file_path)
        df['VT90_TAIR_diff'] = df['VT90'] - df['TAIR']
        df['VT90_VT20_diff'] = df['VT90'] - df['VT20']

        for target in ['VT90_TAIR_diff', 'VT90_VT20_diff']:
            print(f"Calculating importance for: {target}")
            result = calculateFeatureImportance(df, target, site_output_folder)
            summary_rows.append({'Site': site_name, **result})

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

#Save Summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)
print("\nAll sites processed. Summary saved.")