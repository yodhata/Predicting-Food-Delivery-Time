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
    features = artifact["features"]                 # list kolom final (OHE)
    scaler = artifact.get("scaler", None)           # scaler training
    scale_cols = artifact.get("scale_cols", [])     # kolom numerik yang di-scale
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

# ======================
# Helpers
# ======================
def segment_label(pred, q1, q3):
    if pred <= q1:
        return f"Fast (≤ {q1:.1f} min)"
    elif pred <= q3:
        return f"Normal ({q1:.1f}–{q3:.1f} min)"
    return f"Long Risk (> {q3:.1f} min)"

def ops_recommendation(pred, q3):
    if pred <= q3:
        return "Standard monitoring is sufficient."
    return (
        "Long-risk order: assign a closer courier, avoid adding extra orders "
        "(limit batching), and send proactive delivery-time updates."
    )

def build_input_encoded(features, distance, prep, exp, traffic, weather):
    """
    Build 1-row dataframe with OHE columns matching `features`.
    """
    row = {f: 0.0 for f in features}

    # numeric raw (nanti di-scale sesuai scale_cols)
    for col, val in {
        "Distance_km": distance,
        "Preparation_Time_min": prep,
        "Courier_Experience_yrs": exp
    }.items():
        if col in row:
            row[col] = float(val)

    # OHE traffic & weather (harus sesuai nama dummy di training)
    t_col = f"Traffic_Level_{traffic}"
    w_col = f"Weather_{weather}"

    if t_col in row:
        row[t_col] = 1.0
    else:
        st.warning(f"Traffic dummy not found: {t_col}")

    if w_col in row:
        row[w_col] = 1.0
    else:
        st.warning(f"Weather dummy not found: {w_col}")

    df = pd.DataFrame([row])

    # safety: pastikan urutan & kolom sama persis
    df = df.reindex(columns=features, fill_value=0.0)
    return df

def apply_scaling_if_needed(input_df, scaler, scale_cols):
    """
    Apply scaler from artifact to scale_cols only.
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

predict_btn = st.sidebar.button("Predict Delivery Time")

# ======================
# Main layout
# ======================
colA, colB = st.columns([2, 1])

with colB:
    st.subheader("IQR Reference (Segments)")
    st.write(f"Q1: **{q1:.1f} min**")
    st.write(f"Q3: **{q3:.1f} min**")

    st.subheader("Model Info")
    st.write(f"Total features: **{len(features)}**")
    st.write(f"Scaler saved: **{'Yes' if scaler is not None else 'No'}**")
    st.write(f"Scaled columns: **{scale_cols if scale_cols else 'None'}**")

with colA:
    st.subheader("Prediction")

    # 1) encoded input (OHE)
    input_df = build_input_encoded(features, distance, prep, exp, traffic, weather)

    # 2) apply scaling (pakai scaler artifact)
    input_df_model = apply_scaling_if_needed(input_df, scaler, scale_cols)

    if predict_btn:
        try:
            pred = float(model.predict(input_df_model)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        seg = segment_label(pred, q1, q3)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Delivery Time", f"{pred:.1f} min")
        m2.metric("Risk Segment", seg)
        m3.metric("Traffic / Weather", f"{traffic} / {weather}")

        st.subheader("Operational Recommendation")
        st.info(ops_recommendation(pred, q3))

        with st.expander("Show model input (encoded)"):
            st.dataframe(input_df, use_container_width=True)

        with st.expander("Show model input (after scaling)"):
            st.dataframe(input_df_model, use_container_width=True)

    else:
        st.info("Fill the inputs on the sidebar and click **Predict Delivery Time**.")