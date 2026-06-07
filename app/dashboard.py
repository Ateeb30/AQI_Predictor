"""
app/dashboard.py — Streamlit interactive AQI prediction dashboard.

Run with:
    streamlit run app/dashboard.py

Features:
    - Current AQI card with colour-coded category
    - Individual pollutant metrics (PM2.5, PM10, O3, NO2, SO2, CO)
    - 3-day forecast cards (+24h / +48h / +72h)
    - Hazardous AQI alert banner
    - 7-day historical trend with coloured AQI bands and forecast overlay
    - SHAP feature importance chart
    - Model evaluation metrics table
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AQI Predictor Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    }

    /* Main background */
    .main .block-container {
        padding-top: 1.5rem;
    }

    /* AQI card colours */
    .aqi-card {
        border-radius: 16px;
        padding: 24px 32px;
        text-align: center;
        color: #fff;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .aqi-good        { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .aqi-moderate    { background: linear-gradient(135deg, #f7971e, #ffd200); color: #333; }
    .aqi-sensitive   { background: linear-gradient(135deg, #f46b45, #eea849); }
    .aqi-unhealthy   { background: linear-gradient(135deg, #c94b4b, #4b134f); }
    .aqi-very        { background: linear-gradient(135deg, #7b4397, #dc2430); }
    .aqi-hazardous   { background: linear-gradient(135deg, #1a1a2e, #7f0000); }

    /* Metric card */
    .metric-pill {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-pill .label { font-size: 0.75rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-pill .value { font-size: 1.4rem; font-weight: 700; color: #fff; margin-top: 4px; }

    /* Section headings */
    .section-heading {
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.75rem;
        color: #e2e8f0;
        border-left: 4px solid #7c3aed;
        padding-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL: str = config.FASTAPI_BASE_URL
DEFAULT_TIMEOUT: float = 15.0


def _aqi_css_class(aqi: float) -> str:
    """Map AQI to a CSS class name for the card background gradient.

    Args:
        aqi: AQI value.

    Returns:
        CSS class string.
    """
    if aqi <= 50:
        return "aqi-good"
    elif aqi <= 100:
        return "aqi-moderate"
    elif aqi <= 150:
        return "aqi-sensitive"
    elif aqi <= 200:
        return "aqi-unhealthy"
    elif aqi <= 300:
        return "aqi-very"
    else:
        return "aqi-hazardous"


def _aqi_emoji(aqi: float) -> str:
    """Return an appropriate emoji for the AQI level."""
    if aqi <= 50:
        return "😊"
    elif aqi <= 100:
        return "😐"
    elif aqi <= 150:
        return "😷"
    elif aqi <= 200:
        return "🤢"
    elif aqi <= 300:
        return "☠️"
    else:
        return "💀"


@st.cache_data(ttl=300)  # Cache for 5 minutes; Refresh button clears this
def _fetch_predict(city: str) -> dict | None:
    """Call the /predict endpoint and return parsed JSON.

    Args:
        city: City name.

    Returns:
        Parsed JSON dict or None on error.
    """
    try:
        resp = httpx.get(
            f"{BASE_URL}/predict",
            params={"city": city},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Could not fetch prediction: {exc}")
        return None


@st.cache_data(ttl=300)
def _fetch_history(city: str, days: int = 7) -> list[dict]:
    """Call the /history endpoint and return a list of records.

    Args:
        city: City name.
        days: Number of days of history.

    Returns:
        List of record dicts.
    """
    try:
        resp = httpx.get(
            f"{BASE_URL}/history",
            params={"city": city, "days": days},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"⚠️ History unavailable: {exc}")
        return []


def _nan_str(value: float | None, decimals: int = 1) -> str:
    """Format a nullable float for display.

    Args:
        value: Float value or None.
        decimals: Decimal places.

    Returns:
        Formatted string or ``"N/A"``.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "N/A"
    return f"{value:.{decimals}f}"


def _aqi_category_from_val(aqi: float) -> str:
    """Return human-readable AQI category label.

    Args:
        aqi: AQI value.

    Returns:
        Category string.
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<h2 style='color:#a78bfa; margin-bottom:0.2rem;'>🌫️ AQI Predictor</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85rem;'>Real-time air quality intelligence powered by ML.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    city_input: str = st.text_input(
        "🏙️ City",
        value=config.CITY,
        help="Enter any city supported by the WAQI network (e.g. Karachi, Delhi, Beijing)",
    )
    history_days: int = st.slider("📅 History (days)", min_value=1, max_value=30, value=7)
    refresh_btn: bool = st.button("🔄 Refresh Data", use_container_width=True)
    if refresh_btn:
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b;'>Data: AQICN + OpenWeatherMap<br>"
        "Store: Hopsworks Feature Store<br>"
        "Models: RF · Ridge · XGB · LGBM · LSTM</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data fetching (with spinner)
# ---------------------------------------------------------------------------

with st.spinner(f"Fetching live AQI data for **{city_input}** …"):
    predict_data = _fetch_predict(city_input)
    history_records = _fetch_history(city_input, days=history_days)

if predict_data is None:
    st.error(
        "⚠️ The FastAPI server is not reachable. "
        "Please start it with: `uvicorn app.main:app --reload`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Extract values
# ---------------------------------------------------------------------------

current_aqi: float = predict_data.get("current_aqi", 0.0)
aqi_category: str = predict_data.get("aqi_category", "Unknown")
alert: bool = predict_data.get("alert", False)
predictions: dict = predict_data.get("predictions", {})
pollutants: dict = predict_data.get("pollutants", {})
timestamp_str: str = predict_data.get("timestamp", "")
station_name: str = predict_data.get("station", city_input)

pred_24h: float = predictions.get("24h", 0.0)
pred_48h: float = predictions.get("48h", 0.0)
pred_72h: float = predictions.get("72h", 0.0)

# ---------------------------------------------------------------------------
# ── Section 1: Current AQI Card ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("### 📍 Current Air Quality")
css_class = _aqi_css_class(current_aqi)
emoji = _aqi_emoji(current_aqi)

col_card, col_pols = st.columns([1, 2])

with col_card:
    st.markdown(
        f"""
        <div class="aqi-card {css_class}">
            <div style="font-size:3.5rem;line-height:1">{emoji}</div>
            <div style="font-size:4rem;font-weight:800;line-height:1.1">{current_aqi:.0f}</div>
            <div style="font-size:1rem;font-weight:500;margin-top:6px;opacity:0.9">AQI</div>
            <div style="font-size:0.85rem;margin-top:4px;opacity:0.8">{aqi_category}</div>
            <div style="font-size:0.72rem;margin-top:6px;opacity:0.65">📍 {station_name}</div>
            <div style="font-size:0.7rem;margin-top:4px;opacity:0.6">{timestamp_str[:19] if timestamp_str else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_pols:
    st.markdown("<div class='section-heading'>Pollutant Breakdown</div>", unsafe_allow_html=True)
    pol_cols = st.columns(6)
    pol_labels = ["PM2.5", "PM10", "O₃", "NO₂", "SO₂", "CO"]
    pol_keys   = ["pm25", "pm10", "o3", "no2", "so2", "co"]

    for i, (label, key) in enumerate(zip(pol_labels, pol_keys)):
        val = pollutants.get(key)
        display = _nan_str(val)
        pol_cols[i].markdown(
            f"""
            <div class="metric-pill">
                <div class="label">{label}</div>
                <div class="value">{display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# ── Section 2: 3-Day Forecast ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("### 📆 3-Day AQI Forecast")

if alert:
    st.warning("⚠️ **Hazardous AQI Alert!** Predicted AQI exceeds 150 — take precautions.")

forecast_cols = st.columns(3)
forecast_items = [
    ("+24 Hours", pred_24h, "🕐"),
    ("+48 Hours", pred_48h, "🕑"),
    ("+72 Hours", pred_72h, "🕒"),
]

for col, (label, pred_val, icon) in zip(forecast_cols, forecast_items):
    delta_val = pred_val - current_aqi
    delta_str = f"{delta_val:+.1f} vs now"
    fc_class = _aqi_css_class(pred_val)
    col.markdown(
        f"""
        <div class="aqi-card {fc_class}" style="padding:18px 20px;">
            <div style="font-size:1rem;font-weight:500;opacity:0.85">{icon} {label}</div>
            <div style="font-size:2.8rem;font-weight:800;line-height:1.2">{pred_val:.0f}</div>
            <div style="font-size:0.8rem;margin-top:4px;opacity:0.8">{_aqi_category_from_val(pred_val) if False else _aqi_category_from_val(pred_val)}</div>
            <div style="font-size:0.78rem;margin-top:6px;opacity:0.65">{delta_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )





st.markdown("---")

# ---------------------------------------------------------------------------
# ── Section 3: Historical Trend + Forecast Overlay ───────────────────────
# ---------------------------------------------------------------------------

st.markdown("### 📈 Historical AQI Trend & Forecast Overlay")

if history_records:
    hist_df = pd.DataFrame(history_records)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
    hist_df = hist_df.sort_values("timestamp")

    # Build Plotly figure
    fig = go.Figure()

    # ── Coloured AQI band shapes ──────────────────────────────────────
    band_ranges = [
        (0, 50,   "rgba(0,200,100,0.06)",   "Good"),
        (50, 100,  "rgba(255,210,0,0.08)",   "Moderate"),
        (100, 150, "rgba(255,140,0,0.09)",   "Unhealthy for Sensitive"),
        (150, 200, "rgba(255,50,50,0.10)",   "Unhealthy"),
        (200, 350, "rgba(160,0,200,0.08)",   "Very Unhealthy / Hazardous"),
    ]
    for y0, y1, color, label in band_ranges:
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="gray",
        )

    # ── Historical AQI line ───────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=hist_df["timestamp"],
            y=hist_df["aqi"],
            name="Historical AQI",
            line=dict(color="#60a5fa", width=2),
            mode="lines",
            hovertemplate="<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>",
        )
    )

    # ── Forecast overlay (dashed line from last point) ────────────────
    if not hist_df.empty:
        last_ts = hist_df["timestamp"].iloc[-1]
        forecast_times = [
            last_ts + pd.Timedelta(hours=24),
            last_ts + pd.Timedelta(hours=48),
            last_ts + pd.Timedelta(hours=72),
        ]
        forecast_vals = [pred_24h, pred_48h, pred_72h]

        # Connect from last historical point
        fig.add_trace(
            go.Scatter(
                x=[last_ts] + forecast_times,
                y=[current_aqi] + forecast_vals,
                name="Forecast",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                mode="lines+markers",
                marker=dict(size=8, symbol="diamond"),
                hovertemplate="<b>Forecast %{x}</b><br>AQI: %{y:.0f}<extra></extra>",
            )
        )

    fig.add_hline(
        y=150, line_color="red", line_dash="dash", line_width=1.2,
        annotation_text="Unhealthy Threshold (150)",
        annotation_position="top left",
        annotation_font_color="red",
    )

    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        xaxis=dict(
            title="Date",
            gridcolor="rgba(255,255,255,0.07)",
            showline=True,
            linecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            title="AQI",
            gridcolor="rgba(255,255,255,0.07)",
            rangemode="tozero",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
        ),
        hovermode="x unified",
        margin=dict(l=40, r=60, t=20, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical data available yet. Run the backfill pipeline to populate history.")

st.markdown("---")

# ---------------------------------------------------------------------------
# ── Section 4: Feature Importance (SHAP) ────────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("### 🔍 Feature Importance (SHAP)")

shap_col1, shap_col2 = st.columns([1, 1])

shap_plot_path = config.PLOTS_DIR / "shap_summary_plot.png"

with shap_col1:
    if shap_plot_path.exists():
        st.image(str(shap_plot_path), caption="SHAP Beeswarm — AQI +24h", use_container_width=True)
    else:
        st.info("SHAP plot not yet generated. Run the training pipeline first.")

with shap_col2:
    # Show top-10 features as a horizontal bar chart
    # These are hard-coded representative defaults until the pipeline runs
    top_features_default = [
        ("aqi_lag_1h", 0.42),
        ("rolling_aqi_mean_3h", 0.38),
        ("pm25", 0.31),
        ("aqi_change_rate", 0.25),
        ("pm25_lag_1h", 0.22),
        ("humidity", 0.18),
        ("temperature", 0.15),
        ("wind_speed", 0.12),
        ("hour_sin", 0.09),
        ("pressure", 0.07),
    ]

    # Prefer loading from a generated file if available
    importance_json_path = config.MODELS_DIR / "top_features.json"
    if importance_json_path.exists():
        import json as _json
        with open(importance_json_path) as _fh:
            top_features_display = _json.load(_fh)
    else:
        top_features_display = top_features_default

    feat_names = [f for f, _ in top_features_display]
    feat_vals  = [v for _, v in top_features_display]

    bar_fig = go.Figure(
        go.Bar(
            x=feat_vals[::-1],
            y=feat_names[::-1],
            orientation="h",
            marker=dict(
                color=feat_vals[::-1],
                colorscale="Viridis",
                showscale=False,
            ),
        )
    )
    bar_fig.update_layout(
        title="Top 10 Features (Mean |SHAP|)",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        xaxis=dict(title="Mean |SHAP|", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# ── Section 5: Model Info ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("### 🧠 Model Performance Summary")

metrics_json_path = None
for candidate_dir in sorted(config.MODELS_DIR.iterdir(), reverse=True) if config.MODELS_DIR.exists() else []:
    candidate_metrics = candidate_dir / "metrics.json" if candidate_dir.is_dir() else None
    if candidate_metrics and candidate_metrics.exists():
        metrics_json_path = candidate_metrics
        model_dir_name = candidate_dir.name
        break

if metrics_json_path and metrics_json_path.exists():
    import json as _json
    with open(metrics_json_path) as _fh:
        loaded_metrics = _json.load(_fh)

    rows_data = []
    for horizon in ["24h", "48h", "72h"]:
        rows_data.append(
            {
                "Horizon": horizon,
                "RMSE": f"{loaded_metrics.get(f'rmse_{horizon}', 'N/A'):.3f}",
                "MAE":  f"{loaded_metrics.get(f'mae_{horizon}',  'N/A'):.3f}",
                "R²":   f"{loaded_metrics.get(f'r2_{horizon}',   'N/A'):.4f}",
            }
        )
    metrics_display_df = pd.DataFrame(rows_data)

    st.markdown(
        f"<p style='color:#94a3b8; font-size:0.85rem;'>Model: <b style='color:#a78bfa'>{model_dir_name}</b></p>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        metrics_display_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    # Placeholder table before any model is trained
    placeholder_df = pd.DataFrame(
        [
            {"Horizon": "24h", "RMSE": "—", "MAE": "—", "R²": "—"},
            {"Horizon": "48h", "RMSE": "—", "MAE": "—", "R²": "—"},
            {"Horizon": "72h", "RMSE": "—", "MAE": "—", "R²": "—"},
        ]
    )
    st.info("No trained model found. Run the training pipeline to see metrics.")
    st.dataframe(placeholder_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='text-align:center;color:#475569;font-size:0.78rem;margin-top:2rem;'>
        🌫️ AQI Predictor · Powered by AQICN, OpenWeatherMap & Hopsworks ·
        Built with Streamlit & FastAPI
    </div>
    """,
    unsafe_allow_html=True,
)
