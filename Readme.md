# 🌦️ Weather Forecasting with Machine Learning

This project applies machine learning models (Random Forest, LightGBM, XGBoost) to forecast thermal inversion strength using Kentucky Mesonet weather station data. It covers data preprocessing, gap filling, model training, evaluation, forecasting, and visualization.

The goal is to build reliable predictive models for VT20 and VT90 temperature differences (thermal inversion indicators), and to assess forecasting accuracy across multiple sites
## 🚗 Features

- **Data Preprocessing**  
    - Cleans raw Mesonet datasets.
    - Handles missing values and fills gaps.
  
- **Model Training**  
     - Supports Random Forest, XGBoost, and LightGBM.
     - Trains single-target models (VT20, VT90) and multi-output models (VT20 & VT90).
  
- **Model Evaluation**  
     - Computes MAE (Mean Absolute Error) and difference metrics.
     - Generates evaluation reports across all sites and models.

- **Automated Forecasting**  
    - Loads trained models from disk.
    - Applies them to new site data for predictions.
  
- **Weather Data Integration**  
     - Downloads HRRR reanalysis data via Herbie
     - Converts outputs and raw data into clean CSVs.
  
- **Results & Reporting**  
    - Produces site-level evaluation reports.
    - Supports visualization of forecast vs actual values.


---
