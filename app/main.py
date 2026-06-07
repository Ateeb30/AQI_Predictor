"""
app/main.py — FastAPI backend for the AQI Predictor.

Endpoints:
    GET /health            — liveness check
    GET /predict?city=...  — 3-day AQI forecast for a city
    GET /history?city=...&days=...  — historical feature rows for plotting

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_fetcher import AQICNFetcher, OpenWeatherFetcher
from src.feature_engineer import FEATURE_COLS, compute_features
from src.local_store import LocalFeatureStore

# Detect Hopsworks
try:
    import hopsworks as _hw  # noqa: F401
    _HOPSWORKS_AVAILABLE = True
except ImportError:
    _HOPSWORKS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("fastapi_app")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AQI Predictor API",
    description="Serverless ML pipeline for 3-day Air Quality Index forecasts.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# AQI category logic
# ---------------------------------------------------------------------------

def _aqi_category(aqi: float) -> str:
    """Map an AQI value to its EPA category label.

    Args:
        aqi: AQI value.

    Returns:
        Category string (e.g., ``"Good"``, ``"Hazardous"``).
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model_artefacts() -> tuple[Any, Any, Any, list[str], bool]:
    """Load the best trained model and preprocessing artefacts from disk.

    Tries the Hopsworks Model Registry first; falls back to local
    ``outputs/models/`` directory.

    Returns:
        Tuple of (model, scaler, imputer, feature_columns, is_lstm).

    Raises:
        RuntimeError: If no model artefacts are found anywhere.
    """
    # ── Try Hopsworks Model Registry ──────────────────────────────────────
    try:
        import hopsworks

        project = hopsworks.login(
            project=config.HOPSWORKS_PROJECT_NAME,
            api_key_value=config.HOPSWORKS_API_KEY,
        )
        mr = project.get_model_registry()
        hw_model = mr.get_model(name=config.MODEL_NAME, version=config.MODEL_VERSION)
        model_dir = Path(hw_model.download())
        logger.info("Loaded model from Hopsworks: %s", model_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hopsworks model load failed (%s); scanning local outputs/.", exc)
        model_dir = None

        # Search local models directory
        for candidate_dir in sorted(config.MODELS_DIR.iterdir(), reverse=True):
            if candidate_dir.is_dir():
                model_dir = candidate_dir
                break

        if model_dir is None:
            raise RuntimeError(
                "No trained model found. Run the training pipeline first."
            )

    model_dir = Path(model_dir)

    # ── Load artefacts ─────────────────────────────────────────────────────
    scaler = joblib.load(model_dir / "scaler.pkl")
    imputer = joblib.load(model_dir / "imputer.pkl")

    with open(model_dir / "feature_columns.json") as fh:
        feature_columns: list[str] = json.load(fh)

    # Detect LSTM (SavedModel directory) vs sklearn pickle
    saved_model_path = model_dir / "saved_model"
    model_pickle_path = model_dir / "model.pkl"
    is_lstm = False

    if saved_model_path.exists():
        import tensorflow as tf
        model = tf.keras.models.load_model(str(saved_model_path))
        is_lstm = True
        logger.info("Loaded LSTM (TensorFlow SavedModel).")
    elif model_pickle_path.exists():
        model = joblib.load(model_pickle_path)
        logger.info("Loaded sklearn model from pickle.")
    else:
        raise RuntimeError(f"No model file found in {model_dir}")

    return model, scaler, imputer, feature_columns, is_lstm


def _predict_aqi(features_df: pd.DataFrame) -> dict[str, float]:
    """Run inference for the 3 AQI horizons.

    Args:
        features_df: Single-row feature DataFrame.

    Returns:
        Dict with ``"24h"``, ``"48h"``, ``"72h"`` predictions.
    """
    model, scaler, imputer, feature_columns, is_lstm = _load_model_artefacts()

    X = features_df[feature_columns].values.astype(np.float64)
    X_imp = imputer.transform(X)
    X_sc = scaler.transform(X_imp)

    if is_lstm:
        X_in = X_sc.reshape(X_sc.shape[0], 1, X_sc.shape[1])
        preds: np.ndarray = model.predict(X_in, verbose=0)[0]
    else:
        preds = model.predict(X_sc)[0]

    return {
        "24h": round(float(preds[0]), 2),
        "48h": round(float(preds[1]), 2),
        "72h": round(float(preds[2]), 2),
    }


# ---------------------------------------------------------------------------
# Hopsworks history fetch
# ---------------------------------------------------------------------------

def _fetch_history_df(city: str, days: int) -> pd.DataFrame:
    """Fetch the last N days of feature rows for city.

    Uses Hopsworks if available, otherwise reads from local Parquet store.

    Args:
        city: City name.
        days: Number of days of history to return.

    Returns:
        Filtered and sorted DataFrame.
    """
    if _HOPSWORKS_AVAILABLE:
        try:
            import hopsworks
            project = hopsworks.login(
                project=config.HOPSWORKS_PROJECT_NAME,
                api_key_value=config.HOPSWORKS_API_KEY,
            )
            fs = project.get_feature_store()
            fg = fs.get_feature_group(
                name=config.FEATURE_GROUP_NAME,
                version=config.FEATURE_GROUP_VERSION,
            )
            df: pd.DataFrame = fg.read()
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df[(df["city"] == city) & (df["timestamp"] >= cutoff)]
            return df.sort_values("timestamp", ascending=True)
        except Exception as exc:
            logger.warning("Hopsworks history fetch failed: %s", exc)
            return pd.DataFrame()
    else:
        store = LocalFeatureStore()
        return store.read_last(city=city, days=days)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness / readiness check.

    Returns:
        JSON ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@app.get("/predict")
async def predict(city: str = Query(default=config.CITY, description="City name")) -> dict[str, Any]:
    """Generate a 3-day AQI forecast for the requested city.

    Fetches real-time AQI and weather data, engineers features, and runs
    the best trained model.

    Args:
        city: WAQI city / station identifier.

    Returns:
        JSON with current AQI, 3-horizon predictions, category, and alert flag.
    """
    try:
        # Fetch live data — use specific station ID for reliable data
        aqicn = AQICNFetcher()
        owm = OpenWeatherFetcher()

        aqicn_data = aqicn.fetch_current(config.AQICN_STATION_ID)
        aqicn_data["city"] = city  # label with user-friendly city name
        weather_data = owm.fetch_current(config.CITY_LAT, config.CITY_LON)

        features_df = compute_features(aqicn_data, weather_data)

        predictions = _predict_aqi(features_df)

        current_aqi: float = float(aqicn_data.get("aqi") or 0.0)
        max_pred = max(predictions.values())

        return {
            "city": city,
            "station": aqicn_data.get("station", city),
            "timestamp": aqicn_data["timestamp"].isoformat(),
            "current_aqi": round(current_aqi, 2),
            "predictions": predictions,
            "aqi_category": _aqi_category(current_aqi),
            "alert": bool(max_pred > 150),
            "pollutants": {
                "pm25": aqicn_data.get("pm25"),
                "pm10": aqicn_data.get("pm10"),
                "o3":   aqicn_data.get("o3"),
                "no2":  aqicn_data.get("no2"),
                "so2":  aqicn_data.get("so2"),
                "co":   aqicn_data.get("co"),
            },
        }
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/history")
async def history(
    city: str = Query(default=config.CITY, description="City name"),
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
) -> list[dict[str, Any]]:
    """Return historical AQI feature rows for the requested city and time window.

    Args:
        city: City name.
        days: Number of days of history (1–90).

    Returns:
        JSON array of feature rows ordered by timestamp ascending.
    """
    df = _fetch_history_df(city, days)
    if df.empty:
        return []

    # Select columns relevant for plotting
    cols_to_return = ["timestamp", "city", "aqi", "pm25", "pm10",
                      "temperature", "humidity", "wind_speed"]
    available = [c for c in cols_to_return if c in df.columns]
    df = df[available].copy()

    # Convert timestamps to ISO strings for JSON serialisation
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Replace NaN with None (JSON null)
    df = df.where(df.notna(), other=None)

    return df.to_dict(orient="records")
