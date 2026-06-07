"""
src/explainability.py — SHAP-based model explainability for AQI Predictor.

Provides:
    generate_shap_summary() — compute SHAP values and save a summary plot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHAP summary generator
# ---------------------------------------------------------------------------

def generate_shap_summary(
    model: Any,
    X_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path | None = None,
    is_lstm: bool = False,
    max_samples: int = 200,
) -> list[tuple[str, float]]:
    """Compute SHAP values and save a beeswarm summary plot.

    Uses ``shap.TreeExplainer`` for tree-based models (RandomForest, XGBoost,
    LightGBM) and ``shap.KernelExplainer`` for all others (Ridge, LSTM).

    Args:
        model: A fitted estimator. For sklearn ``MultiOutputRegressor``, the
            first estimator's SHAP values (24h target) are used for the plot
            since feature importances are consistent across targets.
        X_test: Test feature matrix, shape (n_samples, n_features).
        feature_names: Ordered list of feature names matching columns of
            ``X_test``.
        output_dir: Directory to save the plot. Defaults to
            ``config.PLOTS_DIR``.
        is_lstm: If ``True``, use ``KernelExplainer`` regardless of model type.
        max_samples: Maximum number of test samples to pass to KernelExplainer
            (kept small for performance).

    Returns:
        List of ``(feature_name, mean_abs_shap_value)`` tuples, sorted
        descending by importance, limited to the top 10.
    """
    if output_dir is None:
        output_dir = config.PLOTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Select explainer ──────────────────────────────────────────────────
    shap_values: np.ndarray

    try:
        if is_lstm:
            raise TypeError("LSTM — use KernelExplainer")

        # MultiOutputRegressor wraps a list of estimators; unwrap the first
        inner = model
        if hasattr(model, "estimators_"):
            inner = model.estimators_[0]

        explainer = shap.TreeExplainer(inner)
        shap_values_raw = explainer.shap_values(X_test)

        # shap_values_raw may be list (multi-class RF) or 2-D array
        if isinstance(shap_values_raw, list):
            shap_values = np.array(shap_values_raw[0])
        else:
            shap_values = np.array(shap_values_raw)

        logger.info("Using TreeExplainer for SHAP analysis.")

    except (TypeError, AttributeError, Exception) as exc:  # noqa: BLE001
        logger.warning("TreeExplainer failed (%s); falling back to KernelExplainer.", exc)

        # Limit samples for performance
        X_bg = X_test[:min(50, len(X_test))]
        X_explain = X_test[:min(max_samples, len(X_test))]

        if is_lstm:
            def lstm_predict(x: np.ndarray) -> np.ndarray:
                x3d = x.reshape(x.shape[0], 1, x.shape[1])
                preds = model.predict(x3d, verbose=0)
                return preds[:, 0]  # 24h target for SHAP
            predict_fn = lstm_predict
        else:
            def sklearn_predict(x: np.ndarray) -> np.ndarray:
                return model.predict(x)[:, 0]
            predict_fn = sklearn_predict

        explainer = shap.KernelExplainer(predict_fn, shap.kmeans(X_bg, 10))
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    # ── Beeswarm summary plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test[: len(shap_values)],
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance — AQI +24h Forecast", fontsize=13, pad=12)
    plt.tight_layout()

    save_path = output_dir / "shap_summary_plot.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP summary plot → %s", save_path)

    # ── Top-10 features ───────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_names, mean_abs_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    top_10 = feature_importance[:10]

    logger.info("Top 10 SHAP features:")
    for rank, (feat, importance) in enumerate(top_10, start=1):
        logger.info("  %2d. %-30s %.4f", rank, feat, importance)

    return top_10
