import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳")

st.title("💳 Credit Default Risk Predictor")
st.write("กรอกข้อมูลเพื่อประเมินความเสี่ยงการผิดนัดชำระบัตรเครดิต")

# โหลดโมเดล
model = joblib.load("credit_model.pkl")

# ======================
# INPUT
# ======================

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    limit_bal = st.number_input("Credit Limit", value=200000)
    age = st.slider("Age", 18, 80, 30)

with col2:
    bill_amt = st.number_input("Bill Amount Last Month", value=50000)
    pay_amt = st.number_input("Payment Last Month", value=20000)

st.header("Payment History")

pay_status = st.slider("Payment Delay (months)", -2, 8, 0)

st.info("""
Payment Guide  
-2 = No consumption  
0 = Paid on time  
1 = Delay 1 month  
2 = Delay 2 months
""")

# ======================
# PREDICT
# ======================

if st.button("Predict Risk"):

    # โมเดลต้องการ 23 feature
    input_data = np.array([[
        limit_bal, 1, 2, 1, age,      # demographic
        pay_status,0,0,0,0,0,         # payment history
        bill_amt,bill_amt,bill_amt,bill_amt,bill_amt,bill_amt,  # bill
        pay_amt,pay_amt,pay_amt,pay_amt,pay_amt,pay_amt          # payment
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default ({prob*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk ({(1-prob)*100:.1f}%)")

    st.progress(float(prob))