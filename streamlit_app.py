#!/usr/bin/env python3
"""
streamlit_app.py
Interactive Streamlit UI to pick TFET params and visualize predicted curve.
Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from tfet_feature_engineer import register_pickle_aliases

register_pickle_aliases("main", "__main__")

try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
except Exception:
    pass
st.title("Line-TFET ML Predictor (Streamlit)")

# Load latest artifacts
MODEL_PATH = "saved/tfet_latest_rf.pkl"
SCALER_PATH = "saved/tfet_latest_scaler.pkl"
FE_PATH = "saved/tfet_latest_fe.pkl"

@st.cache_data
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    fe = joblib.load(FE_PATH)
    return model, scaler, fe

try:
    model, scaler, fe = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}\nRun training (python train_pipeline.py) first.")
    st.stop()

# Sidebar controls
st.sidebar.header("Device Parameters")
Lov = st.sidebar.slider("Lov (nm)", 0.1, 10.0, 4.0, 0.1)
tn  = st.sidebar.slider("tn (nm)", 1.0, 5.0, 2.0, 0.1)
tox = st.sidebar.slider("tox (nm)", 1.0, 6.0, 2.5, 0.05)
WK  = st.sidebar.slider("Work function WK (eV)", 4.1, 4.5, 4.25, 0.01)
points = st.sidebar.slider("VG points", 50, 400, 200)

if st.sidebar.button("Predict ID–VG Curve"):
    VG = np.linspace(0.0, 0.6, points)
    data = np.column_stack([np.full(points, Lov),
                            np.full(points, tn),
                            np.full(points, tox),
                            np.full(points, WK),
                            VG])
    data_fe = fe.transform(data)
    data_scaled = scaler.transform(data_fe)
    ylog = model.predict(data_scaled)
    y = np.exp(ylog)

    df = pd.DataFrame({"VG": VG, "ID": y})
    # Plot using matplotlib for semilog
    plt.figure(figsize=(8,4))
    plt.semilogy(df["VG"], df["ID"], lw=2)
    plt.xlabel("VG (V)")
    plt.ylabel("ID (A)")
    plt.title(f"Predicted ID–VG (Lov={Lov}, tn={tn}, tox={tox}, WK={WK})")
    plt.grid(True, which='both', ls='--')
    st.pyplot(plt)

    st.subheader("Predicted data (download)")
    st.dataframe(df.head(20))
    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "predicted_curve.csv", "text/csv")

# Feature importance display
if st.sidebar.button("Show Feature Importances"):
    feature_names = ["Lov","tn","tox","WK","VG","Lov_pow","exp_neg_tn","exp_neg_tox","VG_sq","Lov_VG","Lov_VG_pow","WK_shift"]
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        st.subheader("Feature importances")
        st.table(fi_df)
    except Exception as e:
        st.error(f"Could not get feature importances: {e}")
