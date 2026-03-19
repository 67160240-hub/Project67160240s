import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳")

st.title("💳 Credit Default Risk Predictor")

st.write("กรอกข้อมูลลูกค้าเพื่อประเมินความเสี่ยงการผิดนัดชำระบัตรเครดิต")

# โหลดโมเดล
model = joblib.load("credit_model.pkl")

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    LIMIT_BAL = st.number_input("Credit Limit", value=200000)
    AGE = st.slider("Age", 18, 80, 30)

with col2:
    BILL_AMT1 = st.number_input("Bill Amount Last Month", value=50000)
    PAY_AMT1 = st.number_input("Payment Last Month", value=20000)

st.header("Payment History")

PAY_0 = st.slider("Payment Status Last Month", -2, 8, 0)

st.info("""
Payment Status Guide  
-2 = No consumption  
0 = Paid on time  
1 = Delay 1 month  
2 = Delay 2 months  
""")

if st.button("Predict Risk"):

    input_data = np.array([[

        LIMIT_BAL, 1, 2, 1, AGE,

        PAY_0, 0, 0, 0, 0, 0,

        BILL_AMT1, BILL_AMT1, BILL_AMT1,
        BILL_AMT1, BILL_AMT1, BILL_AMT1,

        PAY_AMT1, PAY_AMT1, PAY_AMT1,
        PAY_AMT1, PAY_AMT1, PAY_AMT1

    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default ({probability*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Default ({(1-probability)*100:.1f}%)")

    st.progress(float(probability))