# 🌫️ Serverless ML System: 3-Day Air Quality Index (AQI) Predictor

[![Feature Pipeline (Hourly)](https://github.com/Ateeb30/AQI_Predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/Ateeb30/AQI_Predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline (Daily)](https://github.com/Ateeb30/AQI_Predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/Ateeb30/AQI_Predictor/actions/workflows/training_pipeline.yml)

A production-grade, serverless Machine Learning system that predicts Air Quality Index (AQI) forecasts for the next 24, 48, and 72 hours for the city of Karachi. Built using a decoupled, 4-pipeline architecture, it leverages a Local Parquet Feature Store and Model Registry, GitHub Actions for orchestration, FastAPI for the model serving backend, and Streamlit for a user-facing dashboard.

---

## 📖 Comprehensive Project Report

### 1. Executive Summary & Problem Statement
Air pollution remains a critical global health crisis. Static index reporting tells citizens the air quality *right now*, but fails to provide the foresight required to plan outdoor activities, protect vulnerable demographics, or initiate local policy interventions. 

This project delivers a **Serverless MLOps system** that predicts the AQI for the upcoming three days (24-hour, 48-hour, and 72-hour horizons) using a multi-output regression setup. By tracking ambient air pollutants and atmospheric conditions, the system captures complex temporal patterns. The architecture runs completely serverless and free of infrastructure costs, leveraging Github Actions for pipeline runners, local Parquet for storage/registry, and Streamlit for serving the UI.

---

### 2. System Architecture & The 4-Pipeline Loop
The application is structured into four independent, decoupled pipelines communicating via a Local Feature Store and Model Registry hosted directly in the GitHub repository. This decoupled nature ensures that ingestion, training, serving, and UI rendering can scale and fail independently.

* **1. Backfill Pipeline (Historical Data)**
  Downloads historical weather and air quality data from the Open-Meteo API. It engineers temporal features and shifts the target columns (`aqi_next_24h`, `aqi_next_48h`, `aqi_next_72h`), saving the initial dataset into a local Parquet database (`data/local_feature_store/`).

* **2. Feature Pipeline (Hourly Cron)**
  Executed automatically every hour via GitHub Actions. It fetches current pollutant statistics from the AQICN API (Karachi station) and current weather attributes from OpenWeather. It formats the data, engineers lags, appends the new row into the Parquet database, and automatically commits the updated database back to the GitHub repository.

* **3. Training Pipeline (Daily Cron)**
  Executed automatically every day via GitHub Actions. It pulls the updated Parquet database, performs chronological time-series splitting, trains candidate regression models (RF, XGBoost, Ridge, LightGBM, LSTM), and scores them. The best performing model artifacts and SHAP feature importance plots are saved to the `outputs/` folder and automatically committed back to the GitHub repository.

* **4. Serving & Dashboard Pipeline (FastAPI + Streamlit)**
  * **FastAPI Backend**: Exposes the `/predict` endpoint, which loads the latest trained models from the local registry, fetches live features, and serves the 3-day AQI forecast.
  * **Streamlit Dashboard**: A beautiful, user-friendly frontend allowing users to inspect current air quality, explore predictions, read explainability plots, and interact with the data in real-time.

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

The training pipeline automatically tracks the Root Mean Squared Error (RMSE) on the validation set across all models and saves the best-performing model to the Hopsworks Model Registry.

---

### 5. Model Explainability (SHAP Analysis)
To ensure the model is not a "black box," we integrate SHAP (SHapley Additive exPlanations) to explain individual forecasts. 

* **Lag Features Dominate**: `aqi_lag_1h` and `rolling_aqi_mean_3h` have the highest SHAP values, showing that air quality exhibits high inertia—meaning the air quality in the next 24 hours is heavily dependent on the air quality right now.
* **Meteorological Drivers**: For the 48-hour and 72-hour horizons, weather variables like `temperature` (higher temperatures often increase thermal convection, dispersing pollutants) and `wind_speed` (high winds dilute particle concentration) become significantly more prominent.
* **Diurnal Cycles**: The engineered hour components (`hour_sin` / `hour_cos`) capture rush-hour spikes (morning and evening traffic emissions).

---

### 6. Results and Model Performance

The evaluation metrics achieved by candidate models on the hold-out test set are structured below:

| Target Horizon | Model Type | Root Mean Squared Error (RMSE) | Mean Absolute Error (MAE) | R² Score |
|---|---|---|---|---|
| **24-Hour Forecast** | XGBoost / RF | *[RMSE Placeholder]* | *[MAE Placeholder]* | *[R² Placeholder]* |
| **48-Hour Forecast** | XGBoost / RF | *[RMSE Placeholder]* | *[MAE Placeholder]* | *[R² Placeholder]* |
| **72-Hour Forecast** | XGBoost / RF | *[RMSE Placeholder]* | *[MAE Placeholder]* | *[R² Placeholder]* |

*Note: These placeholder metrics are updated dynamically in Hopsworks whenever the training pipeline executes on new data.*

---

## 🛠️ Tech Stack Table

| Component | Technology | Description |
|---|---|---|
| **Feature Store & Registry** | [Hopsworks](https://www.hopsworks.ai/) | Single source of truth for features and model registry. |
| **Data Ingestion** | Open-Meteo API, AQICN (WAQI) API, OpenWeather API | Provides historical, hourly live AQI, and weather forecasts. |
| **Model Frameworks** | Scikit-Learn, XGBoost, TensorFlow | Regressors optimized for multi-step AQI forecasting (24h/48h/72h). |
| **Model Explainability** | SHAP (SHapley Additive exPlanations) | Explains feature impacts on forecasts. |
| **Orchestration / CI** | GitHub Actions | Automatically triggers ingestion and training pipelines via crons. |
| **Serving Backend** | FastAPI, Uvicorn | High-performance REST API serving prediction endpoints. |
| **Frontend UI** | Streamlit | Visual dashboard with interactive sliders, graphs, and metric cards. |

---

## 🚀 Setup & Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Ateeb30/AQI_Predictor.git
cd AQI_Predictor
```

### 2. Configure Local Environment
Create a `.env` file in the root directory (use `.env.example` as a template) and add your API keys:
```ini
# Hopsworks Configurations
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT_NAME=your_hopsworks_project_name_here

# Weather & Air Quality API Keys
AQICN_API_KEY=your_aqicn_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Location Settings (Karachi - active station ID @-401143)
CITY=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011
```

### 3. Create a Virtual Environment and Install Dependencies
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Running the Pipelines Manually

### Step A: Backfill Historical Data
Run the backfill script to fetch historical data from Open-Meteo and populate your Hopsworks feature store:
```bash
python pipelines/backfill_pipeline.py --start-date 2026-01-01 --end-date 2026-06-05
```

### Step B: Run the Ingestion Feature Pipeline
Ingest the latest weather and air quality observations:
```bash
python pipelines/feature_pipeline.py
```

### Step C: Run the Training Pipeline
Train regression models, register the best model, and save evaluation metrics and plots:
```bash
python pipelines/training_pipeline.py
```

---

## 🖥️ Running the Application & Dashboard

To access the local web dashboard, start both the FastAPI backend server and the Streamlit frontend.

### 1. Start the FastAPI Serving Backend
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The interactive Swagger API documentation will be available at: http://127.0.0.1:8000/docs

### 2. Start the Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
Your browser will open automatically to: http://localhost:8501
