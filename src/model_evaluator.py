"""
src/model_evaluator.py — Evaluation utilities for the AQI Predictor.

Provides:
    evaluate()          — compute RMSE, MAE, R² per target horizon.
    plot_predictions()  — actual vs predicted plot saved to outputs/plots/.
    print_report()      — formatted console table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for servers/CI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_lstm: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute per-horizon RMSE, MAE, and R² for a fitted model.

    Args:
        model: A fitted estimator with a ``predict`` method.
            For LSTM models, input is automatically reshaped to 3-D.
        X_test: Test feature matrix, shape (n_samples, n_features).
        y_test: Ground-truth targets, shape (n_samples, 3).
        is_lstm: If ``True``, reshape X_test to (n_samples, 1, n_features)
            before calling ``predict``.

    Returns:
        Dict with keys ``"24h"``, ``"48h"``, ``"72h"``, each mapping to a
        sub-dict ``{"rmse": float, "mae": float, "r2": float}``.
    """
    if is_lstm:
        X_input = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        y_pred: np.ndarray = model.predict(X_input, verbose=0)
    else:
        y_pred = model.predict(X_test)

    horizons = ["24h", "48h", "72h"]
    metrics: dict[str, dict[str, float]] = {}

    for idx, horizon in enumerate(horizons):
        yt = y_test[:, idx]
        yp = y_pred[:, idx]
        metrics[horizon] = {
            "rmse": _rmse(yt, yp),
            "mae":  _mae(yt, yp),
            "r2":   _r2(yt, yp),
        }
        logger.info(
            "Test [%s] — RMSE: %.2f  MAE: %.2f  R²: %.4f",
            horizon,
            metrics[horizon]["rmse"],
            metrics[horizon]["mae"],
            metrics[horizon]["r2"],
        )

    return metrics


# ---------------------------------------------------------------------------
# Plot predictions
# ---------------------------------------------------------------------------

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: str,
    output_dir: Path | None = None,
) -> plt.Figure:
    """Plot actual vs predicted AQI for a given forecast horizon.

    A horizontal red dashed line is drawn at AQI = 150 (Unhealthy threshold).
    The figure is saved to ``outputs/plots/`` and also returned.

    Args:
        y_true: Ground-truth AQI values, shape (n_samples,).
        y_pred: Predicted AQI values, shape (n_samples,).
        horizon: Horizon label string, e.g. ``"24h"``.
        output_dir: Optional override for the output directory.
            Defaults to ``config.PLOTS_DIR``.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    if output_dir is None:
        output_dir = config.PLOTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    x_axis = np.arange(len(y_true))
    ax.plot(x_axis, y_true, label="Actual AQI", color="#2196F3", linewidth=1.5, alpha=0.85)
    ax.plot(x_axis, y_pred, label="Predicted AQI", color="#FF9800", linewidth=1.5,
            linestyle="--", alpha=0.85)
    ax.axhline(y=150, color="red", linestyle="--", linewidth=1.2, label="Unhealthy Threshold (150)")

    # Shade AQI zones
    ax.axhspan(0, 50, alpha=0.06, color="green")
    ax.axhspan(50, 100, alpha=0.06, color="yellow")
    ax.axhspan(100, 150, alpha=0.06, color="orange")
    ax.axhspan(150, max(y_true.max(), y_pred.max(), 200) + 10, alpha=0.06, color="red")

    ax.set_title(f"AQI Forecast — {horizon} Horizon: Actual vs Predicted", fontsize=14)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("AQI")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_path = output_dir / f"predictions_{horizon}.png"
    fig.savefig(save_path, dpi=150)
    logger.info("Saved prediction plot → %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(metrics: dict[str, dict[str, float]]) -> None:
    """Print a formatted evaluation table to the console.

    Args:
        metrics: Output of ``evaluate()`` — keys are horizon strings,
            values are dicts with ``rmse``, ``mae``, ``r2``.
    """
    header = f"{'Horizon':<10} {'RMSE':>10} {'MAE':>10} {'R²':>10}"
    sep = "-" * len(header)
    print(sep)
    print("  MODEL EVALUATION REPORT")
    print(sep)
    print(header)
    print(sep)
    for horizon, m in metrics.items():
        print(
            f"{horizon:<10} {m['rmse']:>10.3f} {m['mae']:>10.3f} {m['r2']:>10.4f}"
        )
    print(sep)
