import streamlit as st
import pickle
import numpy as np

# Load model
with open("final_exam_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Loan Approval Prediction")

# Numerical inputs
applications = st.number_input("Applications", 1, 10, 1)
granted_loan_amount = st.number_input("Granted Loan Amount", 5000, 2000000, 50000, step=1000)
requested_loan_amount = st.number_input("Requested Loan Amount", 5000, 2500000, 60000, step=1000)
fico_score = st.number_input("FICO Score", 300, 850, 650)
monthly_gross_income = st.number_input("Monthly Gross Income", 0, 20000, 5000)
monthly_housing_payment = st.number_input("Monthly Housing Payment", 300, 50000, 1500)

# Categorical inputs mapped to integers manually
reason_map = {"Debt Consolidation":0,"Home Improvement":1,"Car Purchase":2,"Medical":3,"Other":4}
employment_map = {"Employed":0,"Self-Employed":1,"Unemployed":2,"Student":3,"Retired":4}
lender_map = {"Bank A":0,"Bank B":1,"Bank C":2,"Credit Union":3,"Other":4}
fico_group_map = {"300-579":0,"580-669":1,"670-739":2,"740-799":3,"800-850":4}
sector_map = {"Private":0,"Government":1,"Non-Profit":2,"Self-Employed":3,"Other":4}

reason = st.selectbox("Reason", list(reason_map.keys()))
employment_status = st.selectbox("Employment Status", list(employment_map.keys()))
lender = st.selectbox("Lender", list(lender_map.keys()))
fico_score_group = st.selectbox("FICO Score Group", list(fico_group_map.keys()))
employment_sector = st.selectbox("Employment Sector", list(sector_map.keys()))
ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclose", [0, 1], format_func=lambda x:"Yes" if x else "No")

if st.button("Predict Loan Approval"):
    # Convert all features to numeric in the same order as training
    input_array = np.array([[
        applications,
        granted_loan_amount,
        requested_loan_amount,
        fico_score,
        monthly_gross_income,
        monthly_housing_payment,
        granted_loan_amount / requested_loan_amount,
        monthly_housing_payment / monthly_gross_income,
        reason_map[reason],
        employment_map[employment_status],
        lender_map[lender],
        fico_group_map[fico_score_group],
        sector_map[employment_sector],
        ever_bankrupt_or_foreclose
    ]], dtype=float)

    # Predict
    pred = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]

    if pred==1:
        st.success(f"Loan Approved ✅ (Probability: {proba:.2f})")
        st.balloons()
    else:
        st.error(f"Loan Not Approved ❌ (Probability: {proba:.2f})")
