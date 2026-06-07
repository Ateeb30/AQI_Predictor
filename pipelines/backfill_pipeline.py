"""
pipelines/backfill_pipeline.py — Historical data backfill using Open-Meteo.

Uses the Open-Meteo free API (no key required) to fetch true historical
hourly air quality + weather data for any city/location in 2 API calls.
This replaces the old per-hour AQICN/OWM approach which required paid plans
for historical data.

Data sources:
  - Air quality: https://air-quality-api.open-meteo.com (PM2.5, PM10, AQI, etc.)
  - Weather:     https://archive-api.open-meteo.com (temp, humidity, wind, etc.)

Usage:
    python pipelines/backfill_pipeline.py \\
        --start-date 2026-04-01 \\
        --end-date   2026-06-05 \\
        [--city Karachi] [--lat 24.8607] [--lon 67.0011]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_fetcher import OpenMeteoFetcher
from src.local_store import LocalFeatureStore

# ---------------------------------------------------------------------------
# Detect whether Hopsworks is available (FORCED FALSE FOR PLAN B)
# ---------------------------------------------------------------------------
_HOPSWORKS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("backfill_pipeline")


# ---------------------------------------------------------------------------
# Feature engineering on raw Open-Meteo DataFrame
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all feature columns matching FEATURE_COLS from feature_engineer.py.

    Args:
        df: Raw DataFrame from OpenMeteoFetcher with one row per hour.

    Returns:
        Enriched DataFrame with all columns expected by the training pipeline.
    """
    import math
    import numpy as np

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Time features (must match TIME_COLS in feature_engineer.py) ───────
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek      # 0=Mon…6=Sun
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Cyclic encodings
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]        / 24)
    df["day_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"]  = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # ── Lag features (must match LAG_COLS in feature_engineer.py) ─────────
    df["aqi_lag_1h"]  = df["aqi"].shift(1)
    df["pm25_lag_1h"] = df["pm25"].shift(1)

    # Rate of change: (current - prev) / prev
    prev_aqi = df["aqi"].shift(1)
    df["aqi_change_rate"] = (df["aqi"] - prev_aqi) / prev_aqi.replace(0, float("nan"))

    # Rolling 3-hour mean
    df["rolling_aqi_mean_3h"] = df["aqi"].rolling(3, min_periods=1).mean()

    # ── Targets (shift AQI forward) ────────────────────────────────────────
    df["aqi_next_24h"] = df["aqi"].shift(-24)
    df["aqi_next_48h"] = df["aqi"].shift(-48)
    df["aqi_next_72h"] = df["aqi"].shift(-72)

    return df


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------

def backfill(
    start_date: str,
    end_date: str,
    city: str,
    lat: float,
    lon: float,
) -> None:
    """Backfill historical features from Open-Meteo into local or Hopsworks store.

    Args:
        start_date: Start date ``YYYY-MM-DD``.
        end_date:   End date ``YYYY-MM-DD``.
        city:       City label stored in each row.
        lat:        Latitude.
        lon:        Longitude.
    """
    logger.info(
        "=== Backfill pipeline started — %s → %s for city='%s' ===",
        start_date, end_date, city,
    )
    logger.info(
        "Storage backend: %s",
        "Hopsworks" if _HOPSWORKS_AVAILABLE else "Local Parquet (dev mode)",
    )

    # ── 1. Fetch from Open-Meteo (2 API calls for the full range) ──────────
    fetcher = OpenMeteoFetcher()
    df = fetcher.fetch_historical_range(
        lat=lat, lon=lon,
        start_date=start_date, end_date=end_date,
        city=city,
    )

    if df.empty:
        logger.error("No data returned from Open-Meteo — aborting.")
        return

    logger.info("Raw rows from Open-Meteo: %d", len(df))

    # ── 2. Engineer features ───────────────────────────────────────────────
    df = _engineer_features(df)

    # ── 3. Drop tail rows without targets ─────────────────────────────────
    df = df.iloc[:-72].reset_index(drop=True)
    logger.info("Rows after dropping tail (no 72h target): %d", len(df))

    if len(df) < 100:
        logger.error(
            "Only %d rows available — need >= 100 for training. "
            "Try a wider date range (e.g. --start-date 3 months ago).",
            len(df),
        )
        return

    # ── 4. Store ───────────────────────────────────────────────────────────
    if _HOPSWORKS_AVAILABLE:
        import hopsworks
        project = hopsworks.login(
            project=config.HOPSWORKS_PROJECT_NAME,
            api_key_value=config.HOPSWORKS_API_KEY,
        )
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
            primary_key=["timestamp", "city"],
            event_time="timestamp",
            online_enabled=False,
            description="Hourly AQI + weather features for AQI Predictor model.",
        )
        fg.insert(df, write_options={"wait_for_job": True})
        logger.info("Inserted %d rows into Hopsworks.", len(df))
    else:
        store = LocalFeatureStore()
        store.insert(df)

    logger.info(
        "=== Backfill complete: %d rows for city='%s' (%s → %s) ===",
        len(df), city, start_date, end_date,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical AQI features using Open-Meteo (free)."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date",   required=True, help="YYYY-MM-DD")
    parser.add_argument("--city",  default=config.CITY)
    parser.add_argument("--lat",   type=float, default=config.CITY_LAT)
    parser.add_argument("--lon",   type=float, default=config.CITY_LON)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        city=args.city,
        lat=args.lat,
        lon=args.lon,
    )
