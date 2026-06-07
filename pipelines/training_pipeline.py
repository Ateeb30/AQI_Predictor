"""
pipelines/training_pipeline.py — Daily model training, evaluation, and registration.

Reads features from Hopsworks (CI/CD) or local Parquet store (dev mode).
Saves model artefacts locally and optionally uploads to Hopsworks Model Registry.

Run manually:
    python pipelines/training_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.feature_engineer import FEATURE_COLS, TARGET_COLS
from src.model_trainer import ModelTrainer
from src.model_evaluator import evaluate, plot_predictions, print_report
from src.explainability import generate_shap_summary
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
logger = logging.getLogger("training_pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_features(project: object | None = None) -> pd.DataFrame:
    """Fetch all feature rows from Hopsworks or local Parquet store.

    Args:
        project: Authenticated Hopsworks project handle (or None in dev mode).

    Returns:
        Full feature DataFrame sorted by timestamp ascending.
    """
    if _HOPSWORKS_AVAILABLE and project is not None:
        fs = project.get_feature_store()
        try:
            fg = fs.get_feature_group(
                name=config.FEATURE_GROUP_NAME,
                version=config.FEATURE_GROUP_VERSION,
            )
            logger.info("Reading from Hopsworks feature group '%s' …", config.FEATURE_GROUP_NAME)
            df: pd.DataFrame = fg.read()
        except Exception as exc:
            logger.error("Could not read feature group '%s' from Hopsworks: %s", config.FEATURE_GROUP_NAME, exc)
            df = pd.DataFrame()
    else:
        logger.info("Reading from local Parquet store (dev mode) …")
        store = LocalFeatureStore()
        df = store.read()

    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    logger.info("Loaded %d rows.", len(df))
    return df


def _chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically into train, val, and test sets.

    Args:
        df: Feature DataFrame sorted ascending by timestamp.
        train_frac: Fraction allocated to training.
        val_frac: Fraction allocated to validation.

    Returns:
        Tuple (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Split — train: %d  val: %d  test: %d", len(train_df), len(val_df), len(test_df)
    )
    return train_df, val_df, test_df


def _register_sklearn_model(
    mr: "hopsworks.model_registry.ModelRegistry",  # type: ignore[name-defined]
    model_name: str,
    model_dir: Path,
    metrics: dict,
) -> None:
    """Register a scikit-learn compatible model in Hopsworks Model Registry.

    Args:
        mr: Hopsworks Model Registry handle.
        model_name: Name of the model.
        model_dir: Local directory containing model artefacts.
        metrics: Dict of evaluation metrics to attach.
    """
    sklearn_model = mr.sklearn.create_model(
        name=model_name,
        metrics=metrics,
        description="Best AQI predictor model — 24h/48h/72h forecasts.",
        model_schema=None,
    )
    sklearn_model.save(str(model_dir))
    logger.info("Registered sklearn model '%s' in Hopsworks Model Registry.", model_name)


def _register_tf_model(
    mr: "hopsworks.model_registry.ModelRegistry",  # type: ignore[name-defined]
    model_name: str,
    model_dir: Path,
    metrics: dict,
) -> None:
    """Register a TensorFlow/Keras model in Hopsworks Model Registry.

    Args:
        mr: Hopsworks Model Registry handle.
        model_name: Name of the model.
        model_dir: Local directory containing the SavedModel.
        metrics: Dict of evaluation metrics to attach.
    """
    tf_model = mr.tensorflow.create_model(
        name=model_name,
        metrics=metrics,
        description="Best AQI predictor LSTM — 24h/48h/72h forecasts.",
    )
    tf_model.save(str(model_dir))
    logger.info("Registered TensorFlow model '%s' in Hopsworks Model Registry.", model_name)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    """Execute the full daily training pipeline.

    In dev mode (no Hopsworks), reads from local Parquet and saves artefacts
    locally without uploading to the model registry.
    """
    logger.info("=== Training pipeline started ===")
    logger.info(
        "Mode: %s",
        "Hopsworks (cloud)" if _HOPSWORKS_AVAILABLE else "Local dev (Parquet store)",
    )

    # ── 1. Connect to Hopsworks (if available) and load data ──────────────
    project = None
    if _HOPSWORKS_AVAILABLE:
        import hopsworks
        project = hopsworks.login(
            project=config.HOPSWORKS_PROJECT_NAME,
            api_key_value=config.HOPSWORKS_API_KEY,
        )

    df = _load_features(project)

    # ── 2. Drop rows with missing targets ─────────────────────────────────
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    logger.info("Rows after dropping NaN targets: %d", len(df))

    if len(df) < 100:
        logger.error("Insufficient data (%d rows) — aborting. Run backfill first.", len(df))
        return

    # ── 3. Chronological split ────────────────────────────────────────────
    train_df, val_df, test_df = _chronological_split(df)

    X_train_raw = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COLS].values

    X_val_raw = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COLS].values

    X_test_raw = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COLS].values

    # ── 4. Impute + scale ─────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp   = imputer.transform(X_val_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_val   = scaler.transform(X_val_imp)
    X_test  = scaler.transform(X_test_imp)

    logger.info("Feature matrix shapes — train: %s  val: %s  test: %s",
                X_train.shape, X_val.shape, X_test.shape)

    # ── 5. Train all models ───────────────────────────────────────────────
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)

    # ── 6. Evaluate on test set ───────────────────────────────────────────
    test_metrics_all: dict[str, dict] = {}
    for model_name, res in results.items():
        is_lstm = res.get("is_lstm", False)
        m = evaluate(res["model"], X_test, y_test, is_lstm=is_lstm)
        test_metrics_all[model_name] = m
        logger.info("=== %s test metrics ===", model_name)
        print_report(m)

        # Save prediction plots
        y_pred_test: np.ndarray
        if is_lstm:
            X_in = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            y_pred_test = res["model"].predict(X_in, verbose=0)
        else:
            y_pred_test = res["model"].predict(X_test)

        for idx, horizon in enumerate(["24h", "48h", "72h"]):
            plot_predictions(y_test[:, idx], y_pred_test[:, idx], horizon=f"{model_name}_{horizon}")

    # ── 7. Select best model ──────────────────────────────────────────────
    best_name, best_model = trainer.best_model(results)
    is_lstm_best = results[best_name].get("is_lstm", False)
    best_test_metrics = test_metrics_all[best_name]
    logger.info("Best model: %s", best_name)

    # ── 8. SHAP summary ───────────────────────────────────────────────────
    try:
        top_features = generate_shap_summary(
            best_model,
            X_test,
            feature_names=FEATURE_COLS,
            is_lstm=is_lstm_best,
        )
        logger.info("Top SHAP features: %s", [f for f, _ in top_features[:5]])
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP generation failed: %s", exc)

    # ── 9. Persist artefacts ──────────────────────────────────────────────
    model_export_dir = config.MODELS_DIR / best_name
    model_export_dir.mkdir(parents=True, exist_ok=True)

    # Save scaler & imputer
    joblib.dump(scaler, model_export_dir / "scaler.pkl")
    joblib.dump(imputer, model_export_dir / "imputer.pkl")
    logger.info("Saved scaler and imputer → %s", model_export_dir)

    # Save feature column list
    with open(model_export_dir / "feature_columns.json", "w") as fh:
        json.dump(FEATURE_COLS, fh, indent=2)

    # Build metrics dict for registry
    flat_metrics: dict[str, float] = {}
    for horizon, m in best_test_metrics.items():
        flat_metrics[f"rmse_{horizon}"] = round(m["rmse"], 4)
        flat_metrics[f"mae_{horizon}"]  = round(m["mae"],  4)
        flat_metrics[f"r2_{horizon}"]   = round(m["r2"],   4)

    with open(model_export_dir / "metrics.json", "w") as fh:
        json.dump(flat_metrics, fh, indent=2)

    # Save the model itself
    if is_lstm_best:
        tf_save_path = model_export_dir / "saved_model"
        best_model.save(str(tf_save_path))
        logger.info("Saved LSTM SavedModel → %s", tf_save_path)
    else:
        model_path = model_export_dir / "model.pkl"
        joblib.dump(best_model, model_path)
        logger.info("Saved model pickle → %s", model_path)

    # ── 10. Register in Hopsworks (cloud/CI only) ─────────────────────────
    if _HOPSWORKS_AVAILABLE and project is not None:
        mr = project.get_model_registry()
        if is_lstm_best:
            _register_tf_model(mr, config.MODEL_NAME, model_export_dir, flat_metrics)
        else:
            _register_sklearn_model(mr, config.MODEL_NAME, model_export_dir, flat_metrics)
    else:
        logger.info(
            "Dev mode: skipping Hopsworks model registry upload. "
            "Artefacts saved locally at: %s", model_export_dir
        )

    logger.info("=== Training pipeline complete. Best model: %s ===", best_name)



# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
