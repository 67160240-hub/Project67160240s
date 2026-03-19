import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳")

st.title("💳 Credit Default Risk Predictor")

st.write("ระบบประเมินความเสี่ยงการผิดนัดชำระบัตรเครดิต")

# load model
model = joblib.load("credit_model.pkl")

# ========================
# USER INPUT
# ========================

st.header("Customer Information")

limit_bal = st.number_input("Credit Limit", value=200000)

age = st.slider("Age", 18, 80, 30)

bill_amt = st.number_input("Bill Amount Last Month", value=50000)

pay_amt = st.number_input("Payment Last Month", value=20000)

pay_status = st.slider("Payment Delay (months)", -2, 8, 0)

# ========================
# PREDICT
# ========================

if st.button("Predict Risk"):

    input_data = np.array([[
        limit_bal,  # LIMIT_BAL
        1,          # SEX
        2,          # EDUCATION
        1,          # MARRIAGE
        age,        # AGE

        pay_status, # PAY_0
        pay_status, # PAY_2
        pay_status, # PAY_3
        pay_status, # PAY_4
        pay_status, # PAY_5
        pay_status, # PAY_6

        bill_amt,   # BILL_AMT1
        bill_amt,   # BILL_AMT2
        bill_amt,   # BILL_AMT3
        bill_amt,   # BILL_AMT4
        bill_amt,   # BILL_AMT5
        bill_amt,   # BILL_AMT6

        pay_amt,    # PAY_AMT1
        pay_amt,    # PAY_AMT2
        pay_amt,    # PAY_AMT3
        pay_amt,    # PAY_AMT4
        pay_amt,    # PAY_AMT5
        pay_amt     # PAY_AMT6
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default ({prob*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Default ({(1-prob)*100:.1f}%)")

    st.progress(float(prob))