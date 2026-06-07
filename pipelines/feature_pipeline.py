"""
pipelines/feature_pipeline.py — Hourly feature ingestion pipeline.

Fetches the latest AQI + weather data, engineers features, and upserts
a single row into either the Hopsworks feature group (CI/CD / cloud) or
the local Parquet store (local dev, no Hopsworks installed).

Run manually:
    python pipelines/feature_pipeline.py [--city CITY]

Scheduled via GitHub Actions every hour (see .github/workflows/feature_pipeline.yml).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_fetcher import AQICNFetcher, OpenWeatherFetcher
from src.feature_engineer import compute_features
from src.local_store import LocalFeatureStore

# ---------------------------------------------------------------------------
# Detect whether Hopsworks is available
# ---------------------------------------------------------------------------
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
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("feature_pipeline")


# ---------------------------------------------------------------------------
# Hopsworks helpers (only called when hopsworks is available)
# ---------------------------------------------------------------------------

def _get_or_create_feature_group(project: object) -> object:
    """Return an existing Hopsworks feature group or create a new one."""
    fs = project.get_feature_store()
    try:
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
        logger.info("Found existing feature group '%s'.", config.FEATURE_GROUP_NAME)
        return fg
    except Exception:  # noqa: BLE001
        pass

    logger.info("Creating feature group '%s' …", config.FEATURE_GROUP_NAME)
    return fs.create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        primary_key=["timestamp", "city"],
        event_time="timestamp",
        online_enabled=False,
        description="Hourly AQI + weather features for AQI Predictor model.",
    )


def _fetch_last_row_hopsworks(project: object, city: str) -> dict | None:
    """Retrieve the most recent row for city from Hopsworks."""
    try:
        fs = project.get_feature_store()
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
        )
        df = fg.read()
        city_df = df[df["city"] == city].sort_values("timestamp", ascending=False)
        if city_df.empty:
            return None
        return city_df.iloc[0].to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch last row from Hopsworks: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(city: str, lat: float, lon: float) -> None:
    """Execute one iteration of the hourly feature pipeline.

    Automatically uses Hopsworks if installed, otherwise falls back to the
    local Parquet feature store.

    Args:
        city: City name / WAQI station identifier.
        lat: Latitude for OpenWeatherMap.
        lon: Longitude for OpenWeatherMap.
    """
    logger.info("=== Feature pipeline started for city='%s' ===", city)
    logger.info(
        "Storage backend: %s",
        "Hopsworks" if _HOPSWORKS_AVAILABLE else "Local Parquet (dev mode)",
    )

    # ── 1. Fetch AQI ──────────────────────────────────────────────────────
    aqicn = AQICNFetcher()
    aqicn_data = aqicn.fetch_current(city)
    logger.info("AQI fetched: %.1f", aqicn_data.get("aqi") or 0)

    # ── 2. Fetch weather ──────────────────────────────────────────────────
    owm = OpenWeatherFetcher()
    weather_data = owm.fetch_current(lat, lon)
    logger.info("Weather fetched: temp=%.1f°C", weather_data.get("temperature") or 0)

    # ── 3. Get previous row for lag features ──────────────────────────────
    if _HOPSWORKS_AVAILABLE:
        import hopsworks
        project = hopsworks.login(
            project=config.HOPSWORKS_PROJECT_NAME,
            api_key_value=config.HOPSWORKS_API_KEY,
        )
        previous_row = _fetch_last_row_hopsworks(project, city)
    else:
        store = LocalFeatureStore()
        previous_row = store.read_last_row(city)

    if previous_row:
        logger.info("Previous AQI (lag): %.1f", previous_row.get("aqi") or float("nan"))
    else:
        logger.info("No previous row found — lag features will be NaN.")

    # ── 4. Compute features ───────────────────────────────────────────────
    df = compute_features(aqicn_data, weather_data, previous_row=previous_row)
    logger.info("Features computed — shape: %s", df.shape)

    # ── 5. Insert into storage ─────────────────────────────────────────────
    if _HOPSWORKS_AVAILABLE:
        fg = _get_or_create_feature_group(project)
        fg.insert(df, write_options={"wait_for_job": False})
        logger.info("Inserted into Hopsworks feature group '%s'.", config.FEATURE_GROUP_NAME)
    else:
        store = LocalFeatureStore()
        store.insert(df)
        logger.info("Inserted into local Parquet store.")

    logger.info("=== Feature pipeline complete ===")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly AQI feature pipeline.")
    parser.add_argument("--city", default=config.CITY)
    parser.add_argument("--lat", type=float, default=config.CITY_LAT)
    parser.add_argument("--lon", type=float, default=config.CITY_LON)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(city=args.city, lat=args.lat, lon=args.lon)
