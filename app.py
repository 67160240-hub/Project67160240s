import streamlit as st
import pandas as pd
import joblib
import json

# =========================
# Page Setup
# =========================

st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Default Risk Predictor")
st.caption("AI Model for Predicting Credit Card Default Risk")

st.divider()

# =========================
# Load Model
# =========================

model = joblib.load("credit_model.pkl")

with open("feature_names.json") as f:
    feature_names = json.load(f)

# =========================
# Input Section
# =========================

st.subheader("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    limit_bal = st.number_input("💰 Credit Limit", value=200000)
    age = st.slider("🎂 Age", 18, 80, 30)

with col2:
    bill_amt = st.number_input("🧾 Bill Amount Last Month", value=50000)
    pay_amt = st.number_input("💵 Payment Last Month", value=20000)

st.subheader("📊 Payment History")

pay_status = st.slider("Payment Delay (months)", -2, 8, 0)

st.info("""
Payment Status Guide  
-2 = No consumption  
0 = Paid on time  
1 = Delay 1 month  
2 = Delay 2 months  
""")

st.divider()

# =========================
# Prediction
# =========================

if st.button("🔎 Predict Risk"):

    data = {}

    for col in feature_names:

        if col == "LIMIT_BAL":
            data[col] = limit_bal

        elif col == "AGE":
            data[col] = age

        elif "PAY_" in col:
            data[col] = pay_status

        elif "BILL_AMT" in col:
            data[col] = bill_amt

        elif "PAY_AMT" in col:
            data[col] = pay_amt

        elif col == "SEX":
            data[col] = 1

        elif col == "EDUCATION":
            data[col] = 2

        elif col == "MARRIAGE":
            data[col] = 1

        else:
            data[col] = 0

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📈 Prediction Result")

    if prediction == 1:

        st.error("⚠️ High Risk of Default")

        st.metric(
            label="Default Probability",
            value=f"{prob*100:.2f}%"
        )

        st.progress(prob)

    else:

        st.success("✅ Low Risk of Default")

        st.metric(
            label="Safe Probability",
            value=f"{(1-prob)*100:.2f}%"
        )

        st.progress(1-prob)

st.divider()

st.caption("Model: Random Forest | Dataset: UCI Credit Card Default")