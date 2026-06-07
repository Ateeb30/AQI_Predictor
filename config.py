"""
config.py — Central configuration module for AQI Predictor.

Loads all environment variables via python-dotenv and exposes typed
constants used throughout the pipeline, API layers, and application.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate and load .env
# ---------------------------------------------------------------------------
_ROOT_DIR: Path = Path(__file__).resolve().parent
_ENV_FILE: Path = _ROOT_DIR / ".env"

load_dotenv(dotenv_path=_ENV_FILE, override=True)

def _get_required_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise KeyError(
            f"Missing required environment variable '{name}'. "
            "Please ensure it is set in your local .env file or registered as a GitHub Secret."
        )
    return val

AQICN_API_KEY: str = _get_required_env("AQICN_API_KEY")
OPENWEATHER_API_KEY: str = _get_required_env("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY: str = _get_required_env("HOPSWORKS_API_KEY")

# ---------------------------------------------------------------------------
# Hopsworks
# ---------------------------------------------------------------------------
HOPSWORKS_PROJECT_NAME: str = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
FEATURE_GROUP_NAME: str = "aqi_weather_features"
FEATURE_GROUP_VERSION: int = 1
MODEL_NAME: str = "aqi_predictor_model"
MODEL_VERSION: int = 1

# ---------------------------------------------------------------------------
# City / Geo
# ---------------------------------------------------------------------------
CITY: str = os.getenv("CITY", "Karachi")
CITY_LAT: float = float(os.getenv("CITY_LAT", "24.8607"))
CITY_LON: float = float(os.getenv("CITY_LON", "67.0011"))

# AQICN station ID — use '@<uid>' for a specific sensor.
# Default: University of Karachi (actively reporting).
# The old 'Karachi' query hit the US Consulate station which is frozen since Mar 2025.
AQICN_STATION_ID: str = os.getenv("AQICN_STATION_ID", "@-401143")

# ---------------------------------------------------------------------------
# File-system paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = _ROOT_DIR
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
MODELS_DIR: Path = OUTPUTS_DIR / "models"

# Ensure directories exist at import time
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
AQICN_BASE_URL: str = "https://api.waqi.info"
OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org"

# ---------------------------------------------------------------------------
# FastAPI / Streamlit
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
FASTAPI_BASE_URL: str = f"http://{API_HOST}:{API_PORT}"
