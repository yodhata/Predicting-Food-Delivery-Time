from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from pathlib import Path

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Food Delivery Time Prediction (Minutes)")
st.caption("Predict delivery time and provide simple operational insights.")

# ======================
# Load artifact
# ======================
BASE_DIR = Path(__file__).resolve().parent
artifact_path = BASE_DIR / "delivery_lr_artifact.pkl"

@st.cache_resource
def load_artifact(path_str: str):
    return joblib.load(path_str)

try:
    artifact = load_artifact(str(artifact_path))
    model = artifact["model"]
    features = artifact["features"]                  # list kolom final (after OHE)
    scaler = artifact.get("scaler", None)            # scaler training (optional)
    scale_cols = artifact.get("scale_cols", [])      # numeric cols scaled in training
except Exception as e:
    st.error(f"Failed to load artifact: {e}")
    st.stop()

# ======================
# IQR thresholds
# ======================
iqr = artifact.get("iqr", None)
if isinstance(iqr, dict) and ("q1" in iqr) and ("q3" in iqr):
    q1 = float(iqr["q1"])
    q3 = float(iqr["q3"])
else:
    q1, q3 = 41.0, 71.0

# Segment RMSE (pakai angka hasil evaluasi kamu)
RMSE_SHORT  = 4.075   # <= 41
RMSE_MEDIUM = 7.571   # 41 - 71
RMSE_LONG   = 14.365  # > 71

# ======================
# Helpers
# ======================
def get_prediction_interval(pred_value: float, q1: float, q3: float):
    """
    Range prediksi berdasarkan error model per segmen.
    """
    if pred_value <= q1:
        margin = RMSE_SHORT
        seg = "Fast (High Accuracy)"
        level = "low"
    elif pred_value <= q3:
        margin = RMSE_MEDIUM
        seg = "Normal (Moderate Accuracy)"
        level = "medium"
    else:
        margin = RMSE_LONG
        seg = "Long Risk (Low Accuracy)"
        level = "high"

    min_time = max(0.0, pred_value - margin)
    max_time = pred_value + margin
    return min_time, max_time, seg, level

def ops_recommendation(level: str):
    """
    Rekomendasi operasional mengikuti insight project:
    fokus utama = distance & prep time, lalu traffic/weather.
    batching hanya aman untuk low-risk.
    """
    if level == "high":
        return [
            "High delay risk. Prioritize assigning the closest available courier (distance-first).",
            "Reduce restaurant waiting time (prep-time control). Dispatch courier when order is near-ready.",
            "Avoid batching for this order. Do not add extra stops before delivery.",
            "Send proactive delivery-time updates and use a more conservative ETA."
        ]
    elif level == "medium":
        return [
            "Moderate risk. Use nearest-courier assignment and monitor preparation status.",
            "Allow limited batching only if the added stop is very close and does not increase ETA materially.",
            "Adjust ETA if traffic or weather worsens."
        ]
    else:
        return [
            "Low risk. This order is a good candidate for careful batching with nearby orders.",
            "Still prioritize short distance assignment and minimize prep delays to keep ETA stable."
        ]

def build_input_encoded(features, distance, prep, exp, traffic, weather):
    """
    Build 1-row dataframe with columns exactly matching `features` (OHE-ready).
    """
    row = {f: 0.0 for f in features}

    # numeric raw (akan di-scale jika scaler ada)
    for col, val in {
        "Distance_km": distance,
        "Preparation_Time_min": prep,
        "Courier_Experience_yrs": exp
    }.items():
        if col in row:
            row[col] = float(val)

    # OHE (must match training dummy names)
    t_col = f"Traffic_Level_{traffic}"
    w_col = f"Weather_{weather}"

    if t_col in row:
        row[t_col] = 1.0
    else:
        # jika training kamu pakai nama beda, ini akan ngasih warning
        st.warning(f"Traffic dummy not found in features: {t_col}")

    if w_col in row:
        row[w_col] = 1.0
    else:
        st.warning(f"Weather dummy not found in features: {w_col}")

    df = pd.DataFrame([row]).reindex(columns=features, fill_value=0.0)
    return df

def apply_scaling_if_needed(input_df, scaler, scale_cols):
    """
    Apply scaler TRAINING dari artifact untuk kolom numeric yang memang di-scale saat training.
    """
    if scaler is None or not scale_cols:
        return input_df

    cols_exist = [c for c in scale_cols if c in input_df.columns]
    if not cols_exist:
        return input_df

    out = input_df.copy()
    out.loc[:, cols_exist] = scaler.transform(out[cols_exist])
    return out

# ======================
# Sidebar inputs
# ======================
st.sidebar.header("Order Inputs")

distance = st.sidebar.number_input("Distance (km)", min_value=0.0, value=5.0, step=0.1)
prep = st.sidebar.number_input("Preparation Time (min)", min_value=0.0, value=15.0, step=1.0)
exp = st.sidebar.number_input("Courier Experience (years)", min_value=0.0, value=2.0, step=1.0)

traffic = st.sidebar.selectbox("Traffic Level", ["Low", "Medium", "High"], index=1)
weather = st.sidebar.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy", "Windy"], index=0)

predict_btn = st.sidebar.button("Predict Delivery Time", type="primary")

# ======================
# Main layout
# ======================
colA, colB = st.columns([2, 1])

with colB:
    st.info(
        f"**Segment thresholds (IQR):**\n"
        f"- Fast: ≤ {q1:.0f} min\n"
        f"- Normal: {q1:.0f}–{q3:.0f} min\n"
        f"- Long Risk: > {q3:.0f} min"
    )

    with st.expander("Model Debug Info"):
        st.write(f"Artifact path: {artifact_path.name}")
        st.write(f"Features: {len(features)} columns")
        st.write(f"Scaler loaded: {'Yes' if scaler is not None else 'No'}")
        st.write(f"Scale cols: {scale_cols}")

with colA:
    st.subheader("Prediction Result")

    # 1) build encoded input
    input_df = build_input_encoded(features, distance, prep, exp, traffic, weather)

    # 2) apply scaling (if exists)
    input_df_model = apply_scaling_if_needed(input_df, scaler, scale_cols)

    if predict_btn:
        try:
            pred_val = float(model.predict(input_df_model)[0])
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.stop()

        # range + segment
        min_t, max_t, seg_name, level = get_prediction_interval(pred_val, q1, q3)

        m1, m2, m3 = st.columns(3)
        m1.metric("Point Prediction", f"{pred_val:.1f} min")
        m2.metric("Estimated Range", f"{min_t:.0f} – {max_t:.0f} min")
        m3.metric("Segment", seg_name)

        # feedback box
        if level == "low":
            st.success("Low risk. Prediction is typically more stable in this segment.")
        elif level == "medium":
            st.warning("Moderate risk. Expect moderate variance; monitor prep status and traffic.")
        else:
            st.error("High risk. Long deliveries are harder to predict; treat as a rough estimate.")

        st.markdown("### Operational Actions (Practical)")
        for bullet in ops_recommendation(level):
            st.write(f"- {bullet}")

        with st.expander("Show model input (encoded, before scaling)"):
            st.dataframe(input_df, use_container_width=True)

        with st.expander("Show model input (used by model, after scaling if any)"):
            st.dataframe(input_df_model, use_container_width=True)

    else:
        st.info("Fill the inputs on the sidebar and click **Predict Delivery Time**.")