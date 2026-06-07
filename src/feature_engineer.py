"""
src/feature_engineer.py — Feature engineering for the AQI Predictor pipeline.

Provides ``compute_features`` which merges AQICN and OpenWeather data into a
single, model-ready pandas DataFrame row that includes:
  - Raw pollutant / weather features
  - Cyclic time encodings
  - Lag and rate-of-change features (requires a ``previous_row`` dict)
  - Placeholder target columns populated during backfill
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name constants (also imported by ModelTrainer)
# ---------------------------------------------------------------------------

POLLUTANT_COLS: list[str] = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]

WEATHER_COLS: list[str] = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "pressure",
    "cloud_cover",
    "visibility",
]

TIME_COLS: list[str] = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]

LAG_COLS: list[str] = [
    "aqi_change_rate",
    "aqi_lag_1h",
    "pm25_lag_1h",
    "rolling_aqi_mean_3h",
]

TARGET_COLS: list[str] = ["aqi_next_24h", "aqi_next_48h", "aqi_next_72h"]

FEATURE_COLS: list[str] = (
    POLLUTANT_COLS + WEATHER_COLS + TIME_COLS + LAG_COLS
)

ALL_COLS: list[str] = ["timestamp", "city"] + FEATURE_COLS + TARGET_COLS


# ---------------------------------------------------------------------------
# Cyclic encoding helpers
# ---------------------------------------------------------------------------

def _cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """Encode a periodic value as (sin, cos) pair.

    Args:
        value: The raw periodic value (e.g., hour of day).
        period: The full period of the cycle (e.g., 24 for hours).

    Returns:
        Tuple (sin_encoding, cos_encoding).
    """
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    aqicn_data: dict[str, Any],
    weather_data: dict[str, Any],
    previous_row: dict[str, Any] | None = None,
    rolling_buffer: list[float] | None = None,
) -> pd.DataFrame:
    """Merge AQICN and weather data into a single feature-engineered DataFrame row.

    Args:
        aqicn_data: Output from ``AQICNFetcher.fetch_current`` or
            ``AQICNFetcher.fetch_historical``.
        weather_data: Output from ``OpenWeatherFetcher.fetch_current`` or
            ``OpenWeatherFetcher.fetch_historical``.
        previous_row: Dict representing the immediately prior feature row.
            Used to compute lag and rate-of-change features. If ``None``,
            lag features will be ``np.nan``.
        rolling_buffer: List of recent AQI values (up to last 3) used to
            compute ``rolling_aqi_mean_3h``. If fewer than 3 values are
            available, the mean is taken over available values.

    Returns:
        A single-row ``pd.DataFrame`` containing all feature and target columns.
        Target columns are set to ``np.nan`` (populated during backfill).
    """
    # --- Resolve timestamp (prefer AQICN timestamp) ---
    ts: datetime = aqicn_data.get("timestamp") or weather_data.get("timestamp")
    if ts is None:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    city: str = str(aqicn_data.get("city", "unknown"))

    # --- Raw pollutant features ---
    aqi: float = float(aqicn_data.get("aqi") or np.nan)
    pm25: float = float(aqicn_data.get("pm25") or np.nan)
    pm10: float = float(aqicn_data.get("pm10") or np.nan)
    o3: float = float(aqicn_data.get("o3") or np.nan)
    no2: float = float(aqicn_data.get("no2") or np.nan)
    so2: float = float(aqicn_data.get("so2") or np.nan)
    co: float = float(aqicn_data.get("co") or np.nan)

    # --- Raw weather features ---
    temperature: float = float(weather_data.get("temperature") or np.nan)
    humidity: float = float(weather_data.get("humidity") or np.nan)
    wind_speed: float = float(weather_data.get("wind_speed") or np.nan)
    wind_direction: float = float(weather_data.get("wind_direction") or np.nan)
    pressure: float = float(weather_data.get("pressure") or np.nan)
    cloud_cover: float = float(weather_data.get("cloud_cover") or np.nan)
    visibility: float = float(weather_data.get("visibility") or np.nan)

    # --- Time-based features ---
    hour: int = ts.hour
    day_of_week: int = ts.weekday()          # 0=Monday … 6=Sunday
    day_of_month: int = ts.day
    month: int = ts.month
    is_weekend: int = int(day_of_week >= 5)  # Saturday or Sunday

    hour_sin, hour_cos = _cyclic_encode(hour, 24)
    day_sin, day_cos = _cyclic_encode(day_of_week, 7)
    month_sin, month_cos = _cyclic_encode(month - 1, 12)  # month 1–12 → 0–11

    # --- Lag / derived features ---
    prev_aqi: float = np.nan
    prev_pm25: float = np.nan

    if previous_row is not None:
        prev_aqi = float(previous_row.get("aqi", np.nan) or np.nan)
        prev_pm25 = float(previous_row.get("pm25", np.nan) or np.nan)

    aqi_lag_1h: float = prev_aqi
    pm25_lag_1h: float = prev_pm25

    if not (math.isnan(aqi) or math.isnan(prev_aqi)) and prev_aqi != 0:
        aqi_change_rate: float = (aqi - prev_aqi) / prev_aqi
    else:
        aqi_change_rate = np.nan

    # Rolling 3-hour mean
    if rolling_buffer:
        valid_vals = [v for v in rolling_buffer[-3:] if not math.isnan(v)]
        rolling_aqi_mean_3h: float = float(np.mean(valid_vals)) if valid_vals else np.nan
    else:
        rolling_aqi_mean_3h = np.nan

    # --- Assemble row ---
    row: dict[str, Any] = {
        "timestamp": ts,
        "city": city,
        # pollutants
        "aqi": aqi,
        "pm25": pm25,
        "pm10": pm10,
        "o3": o3,
        "no2": no2,
        "so2": so2,
        "co": co,
        # weather
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "pressure": pressure,
        "cloud_cover": cloud_cover,
        "visibility": visibility,
        # time
        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        # lag / derived
        "aqi_change_rate": aqi_change_rate,
        "aqi_lag_1h": aqi_lag_1h,
        "pm25_lag_1h": pm25_lag_1h,
        "rolling_aqi_mean_3h": rolling_aqi_mean_3h,
        # targets — populated in backfill / training
        "aqi_next_24h": np.nan,
        "aqi_next_48h": np.nan,
        "aqi_next_72h": np.nan,
    }

    df = pd.DataFrame([row])

    # Ensure timestamp column is timezone-aware datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df
