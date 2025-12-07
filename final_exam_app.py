import streamlit as st
import pickle
import pandas as pd

# --- Load model ---
with open("final_exam_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Loan Approval Prediction")

# --- Inputs ---
applications = st.number_input("Applications", 1, 10, 1)
granted_loan_amount = st.number_input("Granted Loan Amount", 5000, 2000000, 50000, step=1000)
requested_loan_amount = st.number_input("Requested Loan Amount", 5000, 2500000, 60000, step=1000)
fico_score = st.number_input("FICO Score", 300, 850, 650)
monthly_gross_income = st.number_input("Monthly Gross Income", 0, 20000, 5000)
monthly_housing_payment = st.number_input("Monthly Housing Payment", 300, 50000, 1500)

reason = st.selectbox("Reason", ["Debt Consolidation", "Home Improvement", "Car Purchase", "Medical", "Other"])
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Student", "Retired"])
lender = st.selectbox("Lender", ["Bank A", "Bank B", "Bank C", "Credit Union", "Other"])
fico_score_group = st.selectbox("FICO Score Group", ["300-579", "580-669", "670-739", "740-799", "800-850"])
employment_sector = st.selectbox("Employment Sector", ["Private", "Government", "Non-Profit", "Self-Employed", "Other"])
ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclose", [0, 1], format_func=lambda x: "Yes" if x else "No")

# --- Predict ---
if st.button("Predict Loan Approval"):
    # Put all features in the same order used in training
    input_data = [[
        applications,
        granted_loan_amount,
        requested_loan_amount,
        fico_score,
        monthly_gross_income,
        monthly_housing_payment,
        reason,
        employment_status,
        lender,
        fico_score_group,
        employment_sector,
        ever_bankrupt_or_foreclose
    ]]

    # The model must have been trained with exactly the same order of features
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Loan Approved ✅ (Probability: {probability:.2f})")
        st.balloons()
    else:
        st.error(f"Loan Not Approved ❌ (Probability: {probability:.2f})")

