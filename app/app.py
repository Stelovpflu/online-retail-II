# =========================================
# STREAMLIT APP – CUSTOMER CHURN & SEGMENTS
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
# LOAD MODELS
# -----------------------------------------
gb_pipeline = joblib.load("gb_churn_pipeline.pkl")
kmeans = joblib.load("kmeans_customer_segmentation.pkl")
kmeans_scaler = joblib.load("kmeans_scaler.pkl")
metadata = joblib.load("model_metadata.pkl")

THRESHOLD = metadata["churn_threshold_recommended"]
cluster_mapping = metadata["cluster_definitions"]
features = metadata["features_used"]

# -----------------------------------------
# SIDEBAR INPUT
# -----------------------------------------
st.sidebar.header("🔢 Customer Features")

input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature,
        min_value=0.0,
        value=10.0
    )

input_df = pd.DataFrame([input_data])

# -----------------------------------------
# PREDICTION
# -----------------------------------------
if st.sidebar.button("🔮 Predict Customer Risk"):
    # Cluster prediction
    X_scaled = kmeans_scaler.transform(input_df)
    cluster_id = kmeans.predict(X_scaled)[0]
    cluster_label = cluster_mapping[cluster_id]

    # Churn prediction
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

    st.markdown(f"### 🎯 Recommended Action")
    st.success(action)

    st.caption(f"Threshold used: {THRESHOLD}")