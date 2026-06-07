# Serverless ML System: 3-Day Air Quality Index (AQI) Predictor

**🔴 Live Dashboard:** [https://aqi-dashboard-y6po.onrender.com/](https://aqi-dashboard-y6po.onrender.com/)

A production-grade, serverless Machine Learning system that predicts Air Quality Index (AQI) forecasts for the next 24, 48, and 72 hours for the city of Karachi.

## Project Description

In this project, we built a fully automated, end-to-end Machine Learning pipeline to forecast air quality. Our system operates completely serverless by leveraging GitHub Actions as our CI/CD orchestrator.

**Data Sources:**
- **Historical Data:** We fetched historical weather and ambient air quality data spanning several months from the Open-Meteo API to train our baseline models.
- **Live Data:** To keep the models updated, we continuously fetch live, real-time pollutant measurements from the AQICN API (WAQI) and current meteorological conditions from the OpenWeather API.

**Architecture & Technologies Used:**
- **Local Parquet Feature Store:** We engineered features and stored them locally in Parquet files within the GitHub repository, eliminating the need for complex external databases.
- **Automated Pipelines:** We use GitHub Actions to execute our Feature Pipeline and Training Pipeline. Both the internal GitHub cron schedule and an external webhook service (`cron-job.org`) are configured to trigger the pipelines. While GitHub's internal scheduler often delays execution depending on server load, the external cron job guarantees strict and exact execution times.
- **FastAPI Backend:** We built a high-performance REST API using FastAPI to serve our model predictions.
- **Streamlit Dashboard:** We developed a premium, interactive frontend dashboard using Streamlit to visualize the current air quality, display 3-day forecasts, and present model explainability charts.

*For the comprehensive deep-dive into Data Acquisition, Feature Engineering, Model Training, Chronological Time-Series Splitting, and SHAP Explainability, please refer to the main Project Report document.*
