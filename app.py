# app.py — Credit Card Default Prediction

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="centered"
)

# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():

    model = joblib.load("credit_model.pkl")

    with open("feature_names.json") as f:
        feature_names = json.load(f)

    return model, feature_names


model, feature_names = load_model()

# ==============================
# Sidebar
# ==============================

with st.sidebar:

    st.header("ℹ️ About Model")

    st.write("Model : Random Forest")

    st.write("Task : Credit Default Classification")

    st.warning(
        "This system predicts the probability that a customer "
        "will default on their credit card payment."
    )

# ==============================
# Header
# ==============================

st.title("💳 Credit Card Default Prediction")

st.write(
    "Enter customer financial information to predict the risk "
    "of credit card default."
)

st.divider()

# ==============================
# Input Section
# ==============================

st.subheader("Customer Information")

input_data = {}

col1, col2 = st.columns(2)

with col1:

    input_data["LIMIT_BAL"] = st.number_input(
        "Credit Limit",
        min_value=0,
        value=200000
    )

    input_data["AGE"] = st.number_input(
        "Age",
        min_value=18,
        value=30
    )

    input_data["PAY_0"] = st.number_input(
        "Repayment Status (Last Month)",
        value=0
    )

    input_data["PAY_2"] = st.number_input(
        "Repayment Status (2 Months Ago)",
        value=0
    )

with col2:

    input_data["BILL_AMT1"] = st.number_input(
        "Bill Amount (Last Month)",
        value=50000
    )

    input_data["PAY_AMT1"] = st.number_input(
        "Payment Amount (Last Month)",
        value=20000
    )

    input_data["BILL_AMT2"] = st.number_input(
        "Bill Amount (2 Months Ago)",
        value=40000
    )

    input_data["PAY_AMT2"] = st.number_input(
        "Payment Amount (2 Months Ago)",
        value=15000
    )

st.divider()

# ==============================
# Predict Button
# ==============================

predict = st.button("🔍 Predict Default Risk")

if predict:

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    with st.spinner("Predicting..."):

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(f"""
        ⚠️ High Risk of Default

        Probability : {probability*100:.2f}%
        """)

    else:

        st.success(f"""
        ✅ Low Risk of Default

        Probability : {(1-probability)*100:.2f}%
        """)

    st.progress(
        float(probability),
        text=f"Default Risk : {probability*100:.2f}%"
    )

    with st.expander("View Input Data"):

        st.dataframe(input_df)