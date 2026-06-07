"""
src/local_store.py — Local Parquet-based feature store for development.

When Hopsworks is not installed (e.g. Windows dev environment), this module
provides a drop-in replacement that reads/writes Parquet files under
outputs/data/.

Usage:
    from src.local_store import LocalFeatureStore
    store = LocalFeatureStore()
    store.insert(df)
    df_all = store.read()
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

# Where parquet files are stored locally
LOCAL_DATA_DIR: Path = config.OUTPUTS_DIR / "data"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH: Path = LOCAL_DATA_DIR / "features.parquet"


class LocalFeatureStore:
    """Lightweight local feature store backed by Parquet files.

    Mimics the Hopsworks FeatureGroup interface used by the pipelines
    (``insert`` and ``read`` methods) so pipeline code works unchanged
    whether Hopsworks is installed or not.
    """

    def __init__(self, path: Path = PARQUET_PATH) -> None:
        """Initialise the store pointing at a Parquet file.

        Args:
            path: Path to the Parquet file. Created on first insert.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def insert(self, df: pd.DataFrame, **kwargs: Any) -> None:
        """Append a DataFrame to the local Parquet store.

        Deduplicates on (timestamp, city) so re-running a pipeline
        for the same hour is idempotent.

        Args:
            df: Feature DataFrame to insert.
            **kwargs: Ignored (compatibility with Hopsworks API).
        """
        df = df.copy()

        # Normalise timestamp column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if self.path.exists():
            existing = pd.read_parquet(self.path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            combined = pd.concat([existing, df], ignore_index=True)
            # Deduplicate — keep latest insert for each (timestamp, city)
            combined = combined.drop_duplicates(
                subset=["timestamp", "city"], keep="last"
            )
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = df.sort_values("timestamp").reset_index(drop=True)

        combined.to_parquet(self.path, index=False)
        logger.info(
            "LocalFeatureStore: inserted %d rows → total %d rows in %s",
            len(df),
            len(combined),
            self.path,
        )

    # ------------------------------------------------------------------
    def read(self, city: str | None = None) -> pd.DataFrame:
        """Read all rows from the local Parquet store.

        Args:
            city: Optional city filter. If provided, returns only rows
                for that city.

        Returns:
            DataFrame sorted by timestamp ascending, or empty DataFrame
            if the store does not exist yet.
        """
        if not self.path.exists():
            logger.warning("LocalFeatureStore: no data at %s", self.path)
            return pd.DataFrame()

        df = pd.read_parquet(self.path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)

        if city is not None:
            df = df[df["city"] == city].reset_index(drop=True)

        logger.info("LocalFeatureStore: read %d rows from %s", len(df), self.path)
        return df

    # ------------------------------------------------------------------
    def read_last(self, city: str, days: int = 7) -> pd.DataFrame:
        """Read the most recent ``days`` days of data for a city.

        Args:
            city: City filter.
            days: Number of days to look back.

        Returns:
            Filtered DataFrame.
        """
        df = self.read(city=city)
        if df.empty:
            return df
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return df[df["timestamp"] >= cutoff].reset_index(drop=True)

    # ------------------------------------------------------------------
    def read_last_row(self, city: str) -> dict | None:
        """Return the most recent row for a city as a dict.

        Args:
            city: City to query.

        Returns:
            Most recent row as a dict, or None if no data exists.
        """
        df = self.read(city=city)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()
