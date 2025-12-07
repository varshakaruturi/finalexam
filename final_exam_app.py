import streamlit as st
import pickle
import pandas as pd
import numpy as np

#with open("final_exam_model.pkl", "rb") as file:
with open("test_final_model.pkl", "rb") as file:
    model = pickle.load(file)

# columns
#feature_columns = model.feature_names_in_  

# --- Columns used in the model (hardcoded) ---
feature_columns = [
    'applications', 'Granted_Loan_Amount', 'Requested_Loan_Amount', 'FICO_score',
    'Monthly_Gross_Income', 'Monthly_Housing_Payment', 'granted_requested_ratio', 'housing_to_income_ratio',
    'Reason_Home Improvement', 'Reason_Car Purchase', 'Reason_Medical', 'Reason_Other',
    'Employment_Status_Self-Employed', 'Employment_Status_Unemployed', 'Employment_Status_Student', 'Employment_Status_Retired',
    'Lender_Bank B', 'Lender_Bank C', 'Lender_Credit Union', 'Lender_Other',
    'Fico_Score_group_580-669', 'Fico_Score_group_670-739', 'Fico_Score_group_740-799', 'Fico_Score_group_800-850',
    'Employment_Sector_Government', 'Employment_Sector_Non-Profit', 'Employment_Sector_Self-Employed', 'Employment_Sector_Other',
    'Ever_Bankrupt_or_Foreclose_1'
]
# title
st.title("Loan Approval Prediction")

# Numerical inputs
applications = st.number_input("Applications", 1, 10, 1)
granted_loan_amount = st.number_input("Granted Loan Amount", 5000, 2000000, 50000, step=1000)
requested_loan_amount = st.number_input("Requested Loan Amount", 5000, 2500000, 60000, step=1000)
fico_score = st.number_input("FICO Score", 300, 850, 650)
monthly_gross_income = st.number_input("Monthly Gross Income", 0, 20000, 5000)
monthly_housing_payment = st.number_input("Monthly Housing Payment", 300, 50000, 1500)

# Categorical inputs
reason = st.selectbox("Reason", ["Debt Consolidation", "Home Improvement", "Car Purchase", "Medical", "Other"])
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Student", "Retired"])
lender = st.selectbox("Lender", ["Bank A", "Bank B", "Bank C", "Credit Union", "Other"])
fico_score_group = st.selectbox("FICO Score Group", ["300-579", "580-669", "670-739", "740-799", "800-850"])
employment_sector = st.selectbox("Employment Sector", ["Private", "Government", "Non-Profit", "Self-Employed", "Other"])
ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclose", [0, 1], format_func=lambda x: "Yes" if x else "No")

# predict
if st.button("Predict Loan Approval"):
    df = pd.DataFrame({
        'applications': [applications],
        'Granted_Loan_Amount': [granted_loan_amount],
        'Requested_Loan_Amount': [requested_loan_amount],
        'FICO_score': [fico_score],
        'Monthly_Gross_Income': [monthly_gross_income],
        'Monthly_Housing_Payment': [monthly_housing_payment],
        'Reason': [reason],
        'Employment_Status': [employment_status],
        'Lender': [lender],
        'Fico_Score_group': [fico_score_group],
        'Employment_Sector': [employment_sector],
        'Ever_Bankrupt_or_Foreclose': [ever_bankrupt_or_foreclose]
    })

    df['granted_requested_ratio'] = df['Granted_Loan_Amount'] / df['Requested_Loan_Amount']
    df['housing_to_income_ratio'] = df['Monthly_Housing_Payment'] / df['Monthly_Gross_Income']
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=False)

    # Add missing columns and reorder to match training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # prediction
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    # print result
    if pred == 1:
        st.success(f"Loan Approved - (Probability: {proba:.2f})")
        st.balloons()
    else:
        st.error(f"Loan Not Approved - (Probability: {proba:.2f})")

