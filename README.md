# 🏠 Household Energy Forecasting – MSc AI Project

> A comprehensive comparative analysis of ARIMA, Random Forest, LSTM, and XGBoost for household energy consumption forecasting using the UCI Appliances Energy Prediction dataset.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project implements a complete time-series forecasting pipeline for household energy consumption. Four models are compared:

- **ARIMA** – Classical statistical baseline
- **Random Forest** – Tree-based ensemble
- **LSTM** – Deep recurrent neural network
- **XGBoost** – Gradient-boosted trees

**Key innovation:** Integration of weather variables (9 features) + engineered temporal features (cyclical encoding, lags, rolling statistics) to improve short-term (1-hour ahead) energy forecasts.

---

## 📊 Dataset

**Source:** UCI Appliances Energy Prediction Dataset  
**Original:** 19,735 observations at 10-minute intervals (Jan–May 2016)  
**Processed:** 3,122 hourly observations with 47 features  

### Features
- **Target:** `Appliances` – energy use (Wh)
- **Indoor Sensors:** Temperature & humidity from 9 rooms
- **Weather Station:** T_out, RH_out, Press_mm_hg, Windspeed, Visibility, Tdewpoint
- **Engineered:** Cyclical time, lags (1h–168h), rolling statistics, weather differentials

---

# 📁 Repository Structure

MLF_Final_Appliances_Energy_Forecasting/

├── data/
│   ├── raw/
│   │   └── energydata_complete.csv
│   ├── processed/
│   │   └── preprocessed_hourly.csv
│   └── results/
│       └── model_results.csv
│
├── figures/
│   ├── fig1_target_distribution.png
│   ├── fig2_timeseries_raw.png
│   ├── fig3_temporal_patterns.png
│   ├── fig4_correlation_heatmap.png
│   ├── fig5_weather_vs_energy.png
│   ├── fig6_hourly_boxplot.png
│   ├── fig7_train_test_split.png
│   ├── fig8_acf_pacf.png
│   ├── fig9_arima_predictions.png
│   ├── fig10_rf_feature_importance.png
│   ├── fig11_rf_predictions.png
│   ├── fig12_lstm_training_history.png
│   ├── fig13_lstm_predictions.png
│   ├── fig14_xgb_feature_importance.png
│   ├── fig15_xgb_predictions.png
│   ├── fig16_model_comparison_bars.png
│   ├── fig17_all_models_comparison.png
│   └── fig18_scatter_all_models.pdf
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AugustineTamba/MLF_Final_Appliances_Energy_Forecasting.git
   cd MLF_Final_Appliances_Energy_Forecasting

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

**Key dependencies:**
pandas, numpy
matplotlib, seaborn
scikit-learn
statsmodels
tensorflow
xgboost
Run the notebook

3.**Open notebooks/energy_forecasting.ipynb**
   Run all cells sequentially

## 🔄 Methodology
**Data Preprocessing**
1. Drop noise variables (rv1, rv2)
2. Resample 10-min → hourly (mean aggregation)
3. Create 47 features:
  - Cyclical encoding (hour_sin, hour_cos, dow_sin, dow_cos)
  - Lags (1h, 2h, 3h, 6h, 12h, 24h, 168h)
  - Rolling statistics (6h mean/std, 24h mean, 7d mean)
  - Weather differentials (temp_diff, humidity_diff)

**Model Training**
1. Chronological split: 80% train (Jan–Apr 2016), 20% test (May 2016)
2. Scaler: StandardScaler (Standardisation)
3. Metrics: MAE, RMSE, R²

**Visualisations**
1. EDA: Distribution, time series, temporal patterns, correlation heatmap, weather relationships
2. Model: Feature importance, predictions vs actual, scatter plots, bar charts
