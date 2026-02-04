# =========================================
# STREAMLIT APP – CUSTOMER CHURN & SEGMENTS
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------------------
# CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide"
)

st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("**Customer Segmentation + Churn Prediction (ML-driven)**")

# -----------------------------------------
# PATHS (ROBUST FOR STREAMLIT CLOUD)
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# -----------------------------------------
# LOAD MODELS
# -----------------------------------------
gb_pipeline = joblib.load(MODELS_DIR / "gb_churn_pipeline.pkl")
kmeans = joblib.load(MODELS_DIR / "kmeans_customer_segmentation.pkl")
kmeans_scaler = joblib.load(MODELS_DIR / "kmeans_scaler.pkl")
metadata = joblib.load(MODELS_DIR / "model_metadata.pkl")

# -----------------------------------------
# METADATA (SAFE ACCESS)
# -----------------------------------------
THRESHOLD = metadata.get("churn_threshold_recommended", 0.30)
cluster_mapping = metadata.get("cluster_definitions", {})
features = metadata.get(
    "features_used",
    [
        "recency_days",
        "frequency",
        "monetary",
        "avg_order_value",
        "tenure_days",
        "purchase_velocity",
    ],
)

# -----------------------------------------
# SIDEBAR INPUT
# -----------------------------------------
st.sidebar.header("🛠 Ajusta las variables")

input_data = {
    "recency_days": st.sidebar.slider(
        "Recency (days since last purchase)",
        min_value=0,
        max_value=750,
        value=90
    ),
    "frequency": st.sidebar.slider(
        "Purchase Frequency",
        min_value=1,
        max_value=400,
        value=5
    ),
    "monetary": st.sidebar.slider(
        "Total Spend",
        min_value=0.0,
        max_value=60000.0,
        value=500.0,
        step=50.0
    ),
    "avg_order_value": st.sidebar.slider(
        "Average Order Value",
        min_value=0.0,
        max_value=11000.0,
        value=75.0,
        step=10.0
    ),
    "tenure_days": st.sidebar.slider(
        "Customer Tenure (days)",
        min_value=1,
        max_value=750,
        value=365
    ),
    "purchase_velocity": st.sidebar.slider(
        "Purchase Velocity",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.01
    ),
}

input_df = pd.DataFrame([input_data])

# -----------------------------------------
# PREDICTION
# -----------------------------------------
if st.sidebar.button("🔮 Predict Customer Risk"):
    # -------------------------------
    # CLUSTER PREDICTION
    # -------------------------------
    X_scaled = kmeans_scaler.transform(input_df)
    cluster_id = kmeans.predict(X_scaled)[0]
    cluster_label = cluster_mapping.get(cluster_id, f"Cluster {cluster_id}")

    # -------------------------------
    # CHURN PREDICTION
    # -------------------------------
    churn_proba = gb_pipeline.predict_proba(input_df)[0, 1]
    churn_flag = int(churn_proba >= THRESHOLD)

    # -------------------------------------
    # BUSINESS LOGIC
    # -------------------------------------
    if churn_proba >= 0.6:
        action = "🔥 Immediate retention action"
    elif churn_proba >= THRESHOLD:
        action = "⚠️ Monitor & targeted engagement"
    else:
        action = "✅ No action required"

    # -------------------------------------
    # OUTPUT
    # -------------------------------------
    st.subheader("🧠 Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Customer Segment", cluster_label)
    col2.metric("Churn Probability", f"{churn_proba:.2%}")
    col3.metric("Churn Risk", "YES" if churn_flag else "NO")

    st.markdown("### 🎯 Recommended Action")
    st.success(action)

    st.caption(f"Threshold used: {THRESHOLD}")

