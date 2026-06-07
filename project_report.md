# Serverless MLOps: 3-Day Air Quality Index (AQI) Predictor
## Comprehensive Final Project Report

---

### 1. Executive Summary & Problem Statement
Air pollution remains a critical global health crisis. Static index reporting tells citizens the air quality *right now*, but fails to provide the foresight required to plan outdoor activities, protect vulnerable demographics, or initiate local policy interventions. 

This project delivers a **Serverless MLOps system** that predicts the AQI for the upcoming three days (24-hour, 48-hour, and 72-hour horizons) using a multi-output regression setup. By tracking ambient air pollutants and atmospheric conditions, the system captures complex temporal patterns. The architecture runs completely serverless and free of infrastructure costs, leveraging Github Actions for CI/CD automation, a Local Parquet database for the feature store and model registry, FastAPI for model serving, and Streamlit for the UI dashboard.

---

### 2. System Architecture & The 4-Pipeline Loop
The application is structured into four independent, decoupled pipelines. This decoupled nature ensures that ingestion, training, serving, and UI rendering can scale and fail independently.

1. **Backfill Pipeline (Historical Data)**
   Downloads historical weather and air quality data from the Open-Meteo API. It engineers temporal features and shifts the target columns (`aqi_next_24h`, `aqi_next_48h`, `aqi_next_72h`), saving the initial dataset into a local Parquet database (`data/local_feature_store/`).
2. **Feature Pipeline (Hourly Webhook Trigger)**
   Executed hourly on GitHub Actions compute runners. To bypass the notorious lag and unpredictability of GitHub's internal cron scheduler, execution is strictly triggered by an external webhook API service (`cron-job.org`). The pipeline fetches current pollutant statistics from the AQICN API, formats the data, appends the new row into the Parquet database, and automatically commits the updated database back to the GitHub repository.
3. **Training Pipeline (Daily Webhook Trigger)**
   Executed daily on GitHub Actions compute runners, also triggered externally via webhook to guarantee execution time. It pulls the updated Parquet database, performs chronological time-series splitting, trains candidate regression models (RF, XGBoost, Ridge, LightGBM, LSTM), and scores them. The best performing model artifacts and SHAP feature importance plots are saved to the `outputs/` folder and automatically committed back to the GitHub repository.
4. **Serving & Dashboard Pipeline (FastAPI + Streamlit)**
   * **FastAPI Backend**: Exposes the `/predict` endpoint, which loads the latest trained models from the local registry, fetches live features, and serves the 3-day AQI forecast.
   * **Streamlit Dashboard**: A beautiful, user-friendly frontend allowing users to inspect current air quality, explore predictions, read explainability plots, and interact with the data in real-time.

**[INSERT SCREENSHOT HERE: GitHub Actions Tab showing the automated "Scheduled" and "Successful" workflow runs]**

---

### 3. Data Acquisition & Feature Engineering
A predictive model is only as good as its features. The dataset combines meteorological features and ambient pollutants:

* **Ambient Pollutants (AQICN API)**: `aqi` (overall index), `pm25` (fine particulate matter), `pm10` (coarse particulate matter), `o3` (ozone), `no2` (nitrogen dioxide), `so2` (sulfur dioxide), and `co` (carbon monoxide).
* **Meteorological Conditions (OpenWeather API / Open-Meteo)**: `temperature`, `humidity`, `wind_speed`, `wind_direction` (decomposed into sin/cos components), `pressure`, `cloud_cover`, and `visibility`.
* **Engineered Temporal Features**: 
  * Cyclic time features: Sin/cos conversions of `hour`, `day_of_week`, and `month` to allow the models to learn diurnal, weekly, and seasonal cycles smoothly.
  * Lags & Rates of Change: 1-hour lag of AQI (`aqi_lag_1h`), 1-hour lag of PM2.5 (`pm25_lag_1h`), AQI rate of change over the last hour (`aqi_change_rate`), and a 3-hour rolling average (`rolling_aqi_mean_3h`) to capture momentum and short-term trends.
* **Target Configurations**: For each row at timestamp $t$, target variables are the future ground-truth AQI values: $y_{t+24}$, $y_{t+48}$, and $y_{t+72}$.

---

### 4. Model Training & Validation Protocol
To prevent temporal data leakage (predicting the past using future data), we avoid random K-Fold cross-validation. Instead, we use a **Chronological Time-Series Split**:
* **Training Set**: 70% of chronological history.
* **Validation Set**: 15% of chronological history.
* **Test Set**: 15% of chronological history.

We train and compare five different models to handle multi-step forecasting:
1. **Multi-Output Ridge Regression**: A linear model with $L_2$ regularization acting as our baseline.
2. **Multi-Output Random Forest Regressor**: An ensemble bagger that captures non-linear splits.
3. **Multi-Output XGBoost Regressor**: An ensemble gradient booster optimized for structured tabular features.
4. **Multi-Output LightGBM Regressor**: A fast, leaf-wise tree growth gradient booster.
5. **LSTM (Long Short-Term Memory) Neural Network**: A recurrent network built in TensorFlow to model sequential relationships.

The training pipeline automatically tracks the Root Mean Squared Error (RMSE) on the validation set across all models and saves the best-performing model to the Local Model Registry.

**[INSERT SCREENSHOT HERE: One of the model prediction vs actual plots from the `outputs/plots/` folder, for example `predictions_RandomForest_24h.png`]**

---

### 5. Model Explainability (SHAP Analysis)
To ensure the model is not a "black box," we integrate SHAP (SHapley Additive exPlanations) to explain individual forecasts. 

* **Lag Features Dominate**: `aqi_lag_1h` and `rolling_aqi_mean_3h` have the highest SHAP values, showing that air quality exhibits high inertia—meaning the air quality in the next 24 hours is heavily dependent on the air quality right now.
* **Meteorological Drivers**: For the 48-hour and 72-hour horizons, weather variables like `temperature` (higher temperatures often increase thermal convection, dispersing pollutants) and `wind_speed` (high winds dilute particle concentration) become significantly more prominent.
* **Diurnal Cycles**: The engineered hour components (`hour_sin` / `hour_cos`) capture rush-hour spikes (morning and evening traffic emissions).

**[INSERT SCREENSHOT HERE: The SHAP summary plot located at `outputs/plots/shap_summary_plot.png`]**

---

### 6. Serving and User Interface (Dashboard)
The predictions are served to users via an interactive Streamlit application. The dashboard consumes the FastAPI backend data and presents:
- Dynamic color-coded cards for current AQI (Good, Moderate, Unhealthy, Hazardous).
- Real-time measurements of all 6 primary pollutants.
- 24-hour, 48-hour, and 72-hour future AQI forecasts with integrated health warnings.

**[INSERT SCREENSHOT HERE: A full screenshot of the Streamlit Dashboard UI running in your browser]**
