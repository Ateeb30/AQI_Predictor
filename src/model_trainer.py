
"""
src/model_trainer.py — Multi-model training for the AQI Predictor pipeline.

Trains RandomForest, Ridge, XGBoost, LightGBM, and an LSTM model to
predict AQI at +24 h, +48 h, and +72 h horizons simultaneously.

Exports:
    ModelTrainer : class encapsulating training logic.
    FEATURE_COLUMNS : list of feature column names.
    TARGET_COLUMNS  : list of target column names.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_engineer import FEATURE_COLS, TARGET_COLS

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = FEATURE_COLS
TARGET_COLUMNS: list[str] = TARGET_COLS


# ---------------------------------------------------------------------------
# RMSE utility
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# LSTM builder
# ---------------------------------------------------------------------------

def _build_lstm(n_features: int, n_targets: int) -> Any:
    """Build and compile a two-layer LSTM model for multi-output regression.

    Args:
        n_features: Number of input features.
        n_targets: Number of output targets (3 for 24h/48h/72h).

    Returns:
        Compiled Keras Sequential model.
    """
    # Import inside function to avoid TF import at module load time
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential(
        [
            LSTM(64, input_shape=(1, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(n_targets),
        ],
        name="aqi_lstm",
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Encapsulates training of all candidate AQI prediction models.

    Models trained:
        - ``RandomForestRegressor`` (wrapped in ``MultiOutputRegressor``)
        - ``Ridge`` (wrapped in ``MultiOutputRegressor``)
        - ``XGBRegressor`` (natively supports multi-output via ``MultiOutputRegressor``)
        - ``LGBMRegressor`` (wrapped in ``MultiOutputRegressor``)
        - Keras LSTM (native multi-output Dense layer)

    All sklearn-family models are wrapped with ``MultiOutputRegressor`` so
    they can jointly predict the three target horizons.
    """

    def __init__(self) -> None:
        """Initialise ModelTrainer."""
        pass

    # ------------------------------------------------------------------
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Train all candidate models and return a results dict.

        Args:
            X_train: Training feature matrix, shape (n_samples, n_features).
            y_train: Training targets, shape (n_samples, 3).
            X_val: Validation feature matrix.
            y_val: Validation targets.

        Returns:
            Dict mapping model name → dict with keys:
                ``model``: fitted estimator,
                ``val_rmse_24h``, ``val_rmse_48h``, ``val_rmse_72h``,
                ``val_rmse_mean``.
        """
        results: dict[str, Any] = {}

        # ── Random Forest ──────────────────────────────────────────────
        logger.info("Training RandomForestRegressor …")
        rf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        )
        rf.fit(X_train, y_train)
        results["RandomForest"] = self._eval_sklearn(rf, X_val, y_val, "RandomForest")

        # ── Ridge Regression ──────────────────────────────────────────
        logger.info("Training Ridge Regression …")
        ridge = MultiOutputRegressor(Ridge(alpha=1.0))
        ridge.fit(X_train, y_train)
        results["Ridge"] = self._eval_sklearn(ridge, X_val, y_val, "Ridge")

        # ── XGBoost ────────────────────────────────────────────────────
        logger.info("Training XGBRegressor …")
        xgb = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
        )
        xgb.fit(X_train, y_train)
        results["XGBoost"] = self._eval_sklearn(xgb, X_val, y_val, "XGBoost")

        # ── LightGBM ───────────────────────────────────────────────────
        logger.info("Training LGBMRegressor …")
        lgbm = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        )
        lgbm.fit(X_train, y_train)
        results["LightGBM"] = self._eval_sklearn(lgbm, X_val, y_val, "LightGBM")

        # ── LSTM ────────────────────────────────────────────────────────
        logger.info("Training LSTM …")
        try:
            results["LSTM"] = self._train_lstm(X_train, y_train, X_val, y_val)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LSTM training failed: %s — skipping.", exc)

        return results

    # ------------------------------------------------------------------
    def _eval_sklearn(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        name: str,
    ) -> dict[str, Any]:
        """Evaluate a fitted sklearn-compatible model on the validation set.

        Args:
            model: A fitted estimator with a ``predict`` method.
            X_val: Validation feature matrix.
            y_val: Validation targets, shape (n_samples, 3).
            name: Human-readable model name for logging.

        Returns:
            Dict with model and per-horizon RMSE metrics.
        """
        y_pred = model.predict(X_val)
        rmse_24h = _rmse(y_val[:, 0], y_pred[:, 0])
        rmse_48h = _rmse(y_val[:, 1], y_pred[:, 1])
        rmse_72h = _rmse(y_val[:, 2], y_pred[:, 2])
        mean_rmse = float(np.mean([rmse_24h, rmse_48h, rmse_72h]))
        logger.info(
            "%s — Val RMSE  24h: %.2f  48h: %.2f  72h: %.2f  mean: %.2f",
            name,
            rmse_24h,
            rmse_48h,
            rmse_72h,
            mean_rmse,
        )
        return {
            "model": model,
            "val_rmse_24h": rmse_24h,
            "val_rmse_48h": rmse_48h,
            "val_rmse_72h": rmse_72h,
            "val_rmse_mean": mean_rmse,
        }

    # ------------------------------------------------------------------
    def _train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Build, train, and evaluate an LSTM model.

        Args:
            X_train: Training features, shape (n_train, n_features).
            y_train: Training targets, shape (n_train, 3).
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            Dict with model and per-horizon RMSE metrics.
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        n_features = X_train.shape[1]
        n_targets = y_train.shape[1]

        model = _build_lstm(n_features, n_targets)

        # Reshape for LSTM: (samples, timesteps=1, features)
        X_train_3d = X_train.reshape(X_train.shape[0], 1, n_features)
        X_val_3d = X_val.reshape(X_val.shape[0], 1, n_features)

        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        model.fit(
            X_train_3d,
            y_train,
            validation_data=(X_val_3d, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        y_pred = model.predict(X_val_3d, verbose=0)
        rmse_24h = _rmse(y_val[:, 0], y_pred[:, 0])
        rmse_48h = _rmse(y_val[:, 1], y_pred[:, 1])
        rmse_72h = _rmse(y_val[:, 2], y_pred[:, 2])
        mean_rmse = float(np.mean([rmse_24h, rmse_48h, rmse_72h]))

        logger.info(
            "LSTM — Val RMSE  24h: %.2f  48h: %.2f  72h: %.2f  mean: %.2f",
            rmse_24h,
            rmse_48h,
            rmse_72h,
            mean_rmse,
        )

        return {
            "model": model,
            "val_rmse_24h": rmse_24h,
            "val_rmse_48h": rmse_48h,
            "val_rmse_72h": rmse_72h,
            "val_rmse_mean": mean_rmse,
            "is_lstm": True,
        }

    # ------------------------------------------------------------------
    def best_model(
        self, results: dict[str, dict[str, Any]]
    ) -> tuple[str, Any]:
        """Return the model name and fitted object with the lowest mean validation RMSE.

        Args:
            results: Output of ``train_all_models``.

        Returns:
            Tuple (model_name, fitted_model_object).
        """
        best_name = min(results, key=lambda k: results[k]["val_rmse_mean"])
        best_obj = results[best_name]["model"]
        logger.info(
            "Best model: %s (mean val RMSE = %.2f)",
            best_name,
            results[best_name]["val_rmse_mean"],
        )
        return best_name, best_obj
