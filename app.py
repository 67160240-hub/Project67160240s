# Credit Default Prediction Web App

import streamlit as st
import pandas as pd
import joblib
import json

# ==============================
# Page Setting
# ==============================

st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Default Risk Prediction")

st.write("กรอกข้อมูลลูกค้าเพื่อทำนายความเสี่ยงในการผิดนัดชำระบัตรเครดิต")

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
# Input Form
# ==============================

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    LIMIT_BAL = st.number_input("Credit Limit", value=200000)
    SEX = st.selectbox("Sex", [1,2])
    EDUCATION = st.selectbox("Education", [1,2,3,4])
    MARRIAGE = st.selectbox("Marriage", [1,2,3])
    AGE = st.number_input("Age", value=30)

with col2:

    PAY_0 = st.number_input("PAY_0 (Last month payment status)", value=0)
    PAY_2 = st.number_input("PAY_2", value=0)
    PAY_3 = st.number_input("PAY_3", value=0)
    PAY_4 = st.number_input("PAY_4", value=0)
    PAY_5 = st.number_input("PAY_5", value=0)
    PAY_6 = st.number_input("PAY_6", value=0)

st.subheader("Bill Amount")

col3, col4 = st.columns(2)

with col3:

    BILL_AMT1 = st.number_input("BILL_AMT1", value=50000)
    BILL_AMT2 = st.number_input("BILL_AMT2", value=40000)
    BILL_AMT3 = st.number_input("BILL_AMT3", value=30000)

with col4:

    BILL_AMT4 = st.number_input("BILL_AMT4", value=20000)
    BILL_AMT5 = st.number_input("BILL_AMT5", value=15000)
    BILL_AMT6 = st.number_input("BILL_AMT6", value=10000)

st.subheader("Payment Amount")

col5, col6 = st.columns(2)

with col5:

    PAY_AMT1 = st.number_input("PAY_AMT1", value=20000)
    PAY_AMT2 = st.number_input("PAY_AMT2", value=15000)
    PAY_AMT3 = st.number_input("PAY_AMT3", value=10000)

with col6:

    PAY_AMT4 = st.number_input("PAY_AMT4", value=8000)
    PAY_AMT5 = st.number_input("PAY_AMT5", value=5000)
    PAY_AMT6 = st.number_input("PAY_AMT6", value=3000)

# ==============================
# Prediction
# ==============================

if st.button("Predict Risk"):

    input_values = [
        LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
        PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
        BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
        PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
    ]

    input_df = pd.DataFrame([input_values], columns=feature_names)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(f"⚠️ High Risk of Default\n\nProbability: {probability*100:.2f}%")

    else:

        st.success(f"✅ Low Risk of Default\n\nProbability: {(1-probability)*100:.2f}%")

    st.progress(float(probability))