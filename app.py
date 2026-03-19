import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳")

st.title("💳 Credit Default Risk Predictor")

st.write("ระบบประเมินความเสี่ยงการผิดนัดชำระบัตรเครดิต")

# โหลดโมเดล
model = joblib.load("credit_model.pkl")

# โหลดชื่อ feature
with open("feature_names.json") as f:
    feature_names = json.load(f)

# =====================
# INPUT USER
# =====================

st.header("Customer Information")

limit_bal = st.number_input("Credit Limit", value=200000)

age = st.slider("Age", 18, 80, 30)

bill_amt = st.number_input("Bill Amount Last Month", value=50000)

pay_amt = st.number_input("Payment Last Month", value=20000)

pay_status = st.slider("Payment Delay (months)", -2, 8, 0)

# =====================
# PREDICT
# =====================

if st.button("Predict Risk"):

    # dictionary ของ feature
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

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk ({prob*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk ({(1-prob)*100:.1f}%)")

    st.progress(float(prob))