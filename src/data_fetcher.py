"""
src/data_fetcher.py — API client layer for AQICN, OpenWeatherMap, and Open-Meteo.

Provides three classes:
  - AQICNFetcher        : real-time AQI from WAQI/AQICN.
  - OpenWeatherFetcher  : real-time weather from OpenWeatherMap.
  - OpenMeteoFetcher    : FREE historical + forecast air quality & weather
                          from Open-Meteo (no API key required, global coverage,
                          data back to 2022). Used for backfill.

Open-Meteo docs: https://open-meteo.com/en/docs
Air quality API: https://air-quality-api.open-meteo.com
Archive API:     https://archive-api.open-meteo.com
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry_get(
    url: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    timeout: int = 15,
) -> dict[str, Any]:
    """Perform a GET request with exponential-backoff retries.

    Args:
        url: Target URL.
        params: Optional query parameters.
        max_retries: Maximum number of attempts.
        backoff_base: Base for exponential backoff (seconds).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            wait = backoff_base ** attempt
            logger.warning(
                "Request failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"All {max_retries} retries exhausted for {url}: {last_exc}"
    )


def _safe_float(value: Any) -> float:
    """Convert a value to float, returning np.nan on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# AQICN Fetcher
# ---------------------------------------------------------------------------

class AQICNFetcher:
    """Fetches air quality data from the World Air Quality Index (WAQI/AQICN) API.

    Attributes:
        token: AQICN API token.
        base_url: Base URL of the AQICN API.
    """

    def __init__(self, token: str = config.AQICN_API_KEY) -> None:
        """Initialise the fetcher with an API token."""
        self.token = token
        self.base_url = config.AQICN_BASE_URL

    # ------------------------------------------------------------------
    def _parse_iaqi(self, data: dict[str, Any]) -> dict[str, float]:
        """Extract individual AQI sub-indices from a WAQI response payload.

        Args:
            data: The ``data`` sub-dict of a WAQI API response.

        Returns:
            Dict mapping pollutant name → float value (np.nan if absent).
        """
        iaqi: dict[str, Any] = data.get("iaqi", {})
        return {
            "pm25": _safe_float(iaqi.get("pm25", {}).get("v")),
            "pm10": _safe_float(iaqi.get("pm10", {}).get("v")),
            "o3":   _safe_float(iaqi.get("o3",   {}).get("v")),
            "no2":  _safe_float(iaqi.get("no2",  {}).get("v")),
            "so2":  _safe_float(iaqi.get("so2",  {}).get("v")),
            "co":   _safe_float(iaqi.get("co",   {}).get("v")),
        }

    # ------------------------------------------------------------------
    def fetch_current(self, city: str) -> dict[str, Any]:
        """Fetch the current AQI and pollutant data for a city.

        Args:
            city: City name or station identifier accepted by WAQI.

        Returns:
            Dict with keys: aqi, pm25, pm10, o3, no2, so2, co, timestamp, city.
        """
        url = f"{self.base_url}/feed/{city}/"
        payload = _retry_get(url, params={"token": self.token})

        if payload.get("status") != "ok":
            raise ValueError(f"AQICN API error for city '{city}': {payload}")

        data: dict[str, Any] = payload["data"]
        pollutants = self._parse_iaqi(data)

        # Timestamp — WAQI returns ISO-8601 strings or epoch dicts
        raw_time = data.get("time", {})
        ts_str: str = raw_time.get("iso", datetime.now(timezone.utc).isoformat())
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        station_name: str = (
            data.get("city", {}).get("name", city)
            if isinstance(data.get("city"), dict)
            else city
        )

        return {
            "aqi": _safe_float(data.get("aqi")),
            "timestamp": timestamp,
            "city": city,
            "station": station_name,
            **pollutants,
        }

    # ------------------------------------------------------------------
    def fetch_historical(self, city: str, date: str) -> dict[str, Any]:
        """Fetch historical AQI data for a city on a specific date.

        The WAQI history endpoint may not always return sub-pollutants for
        older dates; missing values are filled with np.nan.

        Args:
            city: City name or station identifier.
            date: Date string in YYYY-MM-DD format.

        Returns:
            Dict with keys: aqi, pm25, pm10, o3, no2, so2, co, timestamp, city.
        """
        # Try the dedicated historical endpoint first
        url = f"{self.base_url}/feed/{city}/history/"
        try:
            payload = _retry_get(url, params={"token": self.token, "date": date})
        except RuntimeError:
            logger.warning("History endpoint failed; falling back to current feed.")
            return self.fetch_current(city)

        if payload.get("status") != "ok":
            logger.warning(
                "No historical data for city='%s' date='%s'. Using current.", city, date
            )
            return self.fetch_current(city)

        data: dict[str, Any] = payload["data"]

        # WAQI historical may return a list of hourly readings
        if isinstance(data, list) and data:
            # Pick the first entry for the requested date
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        else:
            return self.fetch_current(city)

        pollutants = self._parse_iaqi(entry)

        # Build timestamp from the date string
        try:
            timestamp = datetime.strptime(date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        return {
            "aqi": _safe_float(entry.get("aqi")),
            "timestamp": timestamp,
            "city": city,
            **pollutants,
        }


# ---------------------------------------------------------------------------
# OpenWeather Fetcher
# ---------------------------------------------------------------------------

class OpenWeatherFetcher:
    """Fetches weather data from the OpenWeatherMap API.

    Uses the ``/weather`` endpoint for current data and the
    ``One Call API 3.0 /timemachine`` endpoint for historical data.

    Attributes:
        api_key: OpenWeatherMap API key.
        base_url: Base URL of the OpenWeatherMap API.
    """

    def __init__(self, api_key: str = config.OPENWEATHER_API_KEY) -> None:
        """Initialise the fetcher with an API key."""
        self.api_key = api_key
        self.base_url = config.OPENWEATHER_BASE_URL

    # ------------------------------------------------------------------
    def _parse_weather_dict(self, data: dict[str, Any], timestamp: datetime) -> dict[str, Any]:
        """Parse a standardised weather dict from a One Call API response object.

        Args:
            data: A ``current`` or ``data[0]`` dict from the One Call response.
            timestamp: The observation timestamp (UTC).

        Returns:
            Dict with temperature, humidity, wind_speed, wind_direction,
            pressure, cloud_cover, visibility, timestamp.
        """
        return {
            "temperature":     _safe_float(data.get("temp")),
            "humidity":        _safe_float(data.get("humidity")),
            "wind_speed":      _safe_float(data.get("wind_speed")),
            "wind_direction":  _safe_float(data.get("wind_deg")),
            "pressure":        _safe_float(data.get("pressure")),
            "cloud_cover":     _safe_float(data.get("clouds")),
            "visibility":      _safe_float(data.get("visibility")),
            "timestamp":       timestamp,
        }

    # ------------------------------------------------------------------
    def fetch_current(self, lat: float, lon: float) -> dict[str, Any]:
        """Fetch current weather conditions for a latitude/longitude.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Dict with temperature, humidity, wind_speed, wind_direction,
            pressure, cloud_cover, visibility, timestamp.
        """
        url = f"{self.base_url}/data/2.5/weather"
        payload = _retry_get(
            url,
            params={
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
            },
        )

        main = payload.get("main", {})
        wind = payload.get("wind", {})
        clouds = payload.get("clouds", {})
        timestamp = datetime.fromtimestamp(
            payload.get("dt", time.time()), tz=timezone.utc
        )

        return {
            "temperature":    _safe_float(main.get("temp")),
            "humidity":       _safe_float(main.get("humidity")),
            "wind_speed":     _safe_float(wind.get("speed")),
            "wind_direction": _safe_float(wind.get("deg")),
            "pressure":       _safe_float(main.get("pressure")),
            "cloud_cover":    _safe_float(clouds.get("all")),
            "visibility":     _safe_float(payload.get("visibility")),
            "timestamp":      timestamp,
        }

    # ------------------------------------------------------------------
    def fetch_historical(self, lat: float, lon: float, unix_timestamp: int) -> dict[str, Any]:
        """Fetch historical weather for a location at a specific Unix timestamp.

        Attempts the One Call API 3.0 timemachine endpoint first. If the API
        returns 401 Unauthorized (free-tier accounts do not have access to
        historical data), falls back immediately to ``fetch_current`` so that
        backfill runs quickly without waiting for retries.

        Args:
            lat: Latitude.
            lon: Longitude.
            unix_timestamp: Unix epoch timestamp (seconds).

        Returns:
            Dict with temperature, humidity, wind_speed, wind_direction,
            pressure, cloud_cover, visibility, timestamp.
        """
        # Try One Call 3.0 timemachine (paid plan required)
        url = f"{self.base_url}/data/3.0/onecall/timemachine"
        try:
            resp = requests.get(
                url,
                params={
                    "lat": lat, "lon": lon, "dt": unix_timestamp,
                    "appid": self.api_key, "units": "metric",
                },
                timeout=10,
            )
            # 401 = free tier — skip retries, fall back immediately
            if resp.status_code == 401:
                raise ValueError("401 Unauthorized — free tier, no timemachine access")
            resp.raise_for_status()
            payload = resp.json()
            data_list = payload.get("data", [])
            data = data_list[0] if data_list else payload.get("current", {})
            ts = datetime.fromtimestamp(data.get("dt", unix_timestamp), tz=timezone.utc)
            return self._parse_weather_dict(data, ts)

        except ValueError:
            # Free-tier fallback: use current weather instead
            logger.debug(
                "OWM timemachine not available (free tier) — using current weather for ts=%d",
                unix_timestamp,
            )
            result = self.fetch_current(lat, lon)
            # Override timestamp to match the requested historical slot
            result["timestamp"] = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
            return result

        except Exception as exc:  # noqa: BLE001
            logger.warning("OWM historical fetch failed: %s — using current.", exc)
            result = self.fetch_current(lat, lon)
            result["timestamp"] = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
            return result


# ---------------------------------------------------------------------------
# Open-Meteo fetcher — FREE, no API key, global coverage, data back to 2022
# ---------------------------------------------------------------------------

class OpenMeteoFetcher:
    """Fetches historical and forecast data from Open-Meteo (free, no key).

    Combines:
      - Air Quality Archive API  → PM2.5, PM10, O3, NO2, SO2, CO, US AQI
      - Weather Archive API      → temperature, humidity, wind, pressure,
                                   cloud cover, visibility

    Both endpoints return hourly data for an arbitrary date range in a single
    HTTP request — much faster than per-hour polling.

    Usage::

        fetcher = OpenMeteoFetcher()
        df = fetcher.fetch_historical_range(
            lat=24.8607, lon=67.0011,
            start_date="2026-05-01", end_date="2026-06-05",
        )
        # df has columns: timestamp, city, aqi, pm25, pm10, o3, no2, so2, co,
        #                 temperature, humidity, wind_speed, wind_direction,
        #                 pressure, cloud_cover, visibility
    """

    AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def _get(self, url: str, params: dict) -> dict:
        """GET with one retry on transient errors."""
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Open-Meteo request failed: {exc}") from exc
        return {}

    def _safe(self, values: list, idx: int) -> float:
        """Return float value or NaN if missing/null."""
        try:
            v = values[idx]
            return float(v) if v is not None else float("nan")
        except (IndexError, TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _pm25_to_aqi(pm25: float) -> float:
        """Convert PM2.5 concentration (µg/m³) to US EPA AQI.

        Uses the same linear interpolation breakpoints as AQICN, so the
        training data AQI scale matches the live AQICN AQI values.

        Args:
            pm25: PM2.5 concentration in µg/m³.

        Returns:
            AQI value (0–500+), or NaN if pm25 is NaN.
        """
        import math
        if math.isnan(pm25) or pm25 < 0:
            return float("nan")

        # (C_low, C_high, I_low, I_high)
        breakpoints = [
            (0.0,   12.0,   0,   50),
            (12.1,  35.4,  51,  100),
            (35.5,  55.4, 101,  150),
            (55.5, 150.4, 151,  200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        for c_lo, c_hi, i_lo, i_hi in breakpoints:
            if c_lo <= pm25 <= c_hi:
                return round(
                    ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
                )
        return min(500.0, pm25 * 1.5)  # beyond scale

    def fetch_historical_range(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        city: str = "unknown",
    ) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Fetch hourly AQ + weather for an entire date range in two API calls.

        Args:
            lat: Latitude.
            lon: Longitude.
            start_date: Start date string ``YYYY-MM-DD``.
            end_date: End date string ``YYYY-MM-DD``.
            city: City label to embed in the ``city`` column.

        Returns:
            pandas DataFrame with one row per hour, containing both AQ and
            weather features merged on timestamp.
        """
        import pandas as pd  # local import to avoid top-level dep at import time

        logger.info(
            "Open-Meteo: fetching %s → %s for (%.4f, %.4f) …",
            start_date, end_date, lat, lon,
        )

        # ── 1. Air Quality ─────────────────────────────────────────────────
        aq_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }
        aq_data = self._get(self.AQ_URL, aq_params)
        aq_hourly = aq_data.get("hourly", {})
        timestamps_raw = aq_hourly.get("time", [])

        pm25_vals   = aq_hourly.get("pm2_5", [])
        pm10_vals   = aq_hourly.get("pm10", [])
        o3_vals     = aq_hourly.get("ozone", [])
        no2_vals    = aq_hourly.get("nitrogen_dioxide", [])
        so2_vals    = aq_hourly.get("sulphur_dioxide", [])
        co_vals     = aq_hourly.get("carbon_monoxide", [])

        logger.info("Open-Meteo AQ: %d hourly rows received.", len(timestamps_raw))

        # ── 2. Weather Archive ──────────────────────────────────────────────
        wx_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": (
                "temperature_2m,relative_humidity_2m,"
                "wind_speed_10m,wind_direction_10m,"
                "surface_pressure,cloud_cover,visibility"
            ),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }
        wx_data = self._get(self.ARCHIVE_URL, wx_params)
        wx_hourly = wx_data.get("hourly", {})

        temp_vals  = wx_hourly.get("temperature_2m", [])
        hum_vals   = wx_hourly.get("relative_humidity_2m", [])
        ws_vals    = wx_hourly.get("wind_speed_10m", [])
        wd_vals    = wx_hourly.get("wind_direction_10m", [])
        pres_vals  = wx_hourly.get("surface_pressure", [])
        cc_vals    = wx_hourly.get("cloud_cover", [])
        vis_vals   = wx_hourly.get("visibility", [])

        logger.info("Open-Meteo Weather: %d hourly rows received.", len(temp_vals))

        # ── 3. Merge into DataFrame ────────────────────────────────────────
        rows = []
        for i, ts_str in enumerate(timestamps_raw):
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            pm25_val = self._safe(pm25_vals, i)
            # Compute AQI from PM2.5 using EPA breakpoints (same as AQICN)
            aqi_val = self._pm25_to_aqi(pm25_val)

            rows.append({
                "timestamp":      ts,
                "city":           city,
                "aqi":            aqi_val,
                "pm25":           self._safe(pm25_vals, i),
                "pm10":           self._safe(pm10_vals, i),
                "o3":             self._safe(o3_vals, i),
                "no2":            self._safe(no2_vals, i),
                "so2":            self._safe(so2_vals, i),
                "co":             self._safe(co_vals, i),
                "temperature":    self._safe(temp_vals, i),
                "humidity":       self._safe(hum_vals, i),
                "wind_speed":     self._safe(ws_vals, i),
                "wind_direction": self._safe(wd_vals, i),
                "pressure":       self._safe(pres_vals, i),
                "cloud_cover":    self._safe(cc_vals, i),
                "visibility":     self._safe(vis_vals, i) / 1000.0,  # m → km
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Open-Meteo: merged DataFrame shape: %s", df.shape)
        return df
