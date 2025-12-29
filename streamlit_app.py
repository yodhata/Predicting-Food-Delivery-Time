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

RMSE_SHORT  = 3.9   # RMSE untuk prediksi <= 41 menit
RMSE_MEDIUM = 7.6   # RMSE untuk prediksi 41 - 71 menit
RMSE_LONG   = 14.2  # RMSE untuk prediksi > 71 menit

# ======================
# Helpers
# ======================
def get_prediction_interval(pred_value, q1, q3):
    """
    Menentukan rentang waktu (min-max) berdasarkan segmen error model.
    """
    if pred_value <= q1:
        margin = RMSE_SHORT
        segment_name = "Short Duration (High Accuracy)"
        color = "green"
    elif pred_value <= q3:
        margin = RMSE_MEDIUM
        segment_name = "Medium Duration (Moderate Accuracy)"
        color = "orange"
    else:
        margin = RMSE_LONG
        segment_name = "Long Duration (Low Accuracy)"
        color = "red"
    
    # Hitung batas bawah (tidak boleh negatif) dan batas atas
    min_time = max(0, pred_value - margin)
    max_time = pred_value + margin
    
    return min_time, max_time, segment_name, color

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

predict_btn = st.sidebar.button("Predict Delivery Time", type="primary")

# ======================
# Main layout
# ======================
colA, colB = st.columns([2, 1])

with colB:
    st.info(f"""
    **Segment Thresholds:**
    - Fast: ≤ {Q1:.0f} min
    - Normal: {Q1:.0f} - {Q3:.0f} min
    - Long: > {Q3:.0f} min
    """)
    
    with st.expander("Model Debug Info"):
        st.write(f"Features: {len(features)}")
        st.write(f"Scaler: {'Loaded' if scaler else 'Missing'}")
        st.write(f"Scale Cols: {scale_cols}")

with colA:
    st.subheader("Prediction Result")

    # 1. Build & Scale Input
    input_df = build_input_encoded(features, distance, prep, exp, traffic, weather)
    input_df_model = apply_scaling_if_needed(input_df, scaler, scale_cols)

    if predict_btn:
        try:
            # 2. Predict
            pred_val = float(model.predict(input_df_model)[0])
            
            # 3. Calculate Interval (Range) based on Segments
            min_t, max_t, seg_name, color_code = get_prediction_interval(pred_val, Q1, Q3)

            # 4. Display Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Exact Prediction", f"{pred_val:.1f} min")
            m2.metric("Estimated Range", f"{min_t:.0f} - {max_t:.0f} min")
            m3.metric("Segment Type", seg_name.split('(')[0].strip()) # Ambil nama depannya aja

            # 5. Visual Feedback
            if color_code == "green":
                st.success(f"Prediction is likely highly accurate (Segment: {seg_name})")
            elif color_code == "orange":
                st.warning(f"Prediction has moderate variance (Segment: {seg_name})")
            else:
                st.error(f"Prediction is a rough estimate due to high variance (Segment: {seg_name})")

            st.markdown("---")
            st.markdown("### 📢 Operational Action")
            st.write(ops_recommendation(pred_val, Q3))

            # Debugging Inputs
            with st.expander("Show Detailed Input Data"):
                st.dataframe(input_df_model)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.write("👈 Please adjust inputs and click **Predict**.")
