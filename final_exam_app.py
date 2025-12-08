import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open("final_exam_model.pkl", "rb") as file:
    model = pickle.load(file)

# columns
feature_columns = model.feature_names_in_  

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
    # --- 1. CREATE INITIAL DATAFRAME (Defines 'df') ---
    df = pd.DataFrame({
        'Granted_Loan_Amount': [granted_loan_amount],
        'FICO_score': [fico_score],
        'Monthly_Gross_Income': [monthly_gross_income],
        'Monthly_Housing_Payment': [monthly_housing_payment],
        'Ever_Bankrupt_or_Foreclose': [ever_bankrupt_or_foreclose],
        'Reason': [reason],
        'Employment_Status': [employment_status],
        'Employment_Sector': [employment_sector],
        'Lender': [lender]
    })

    # --- 3. ONE-HOT ENCODING (Defines 'df_encoded') ---
    # Create the df_encoded variable here
    df_encoded = pd.get_dummies(df, drop_first=False)
    # The definitive list of features the model expects
    feature_columns = [
        'Granted_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment',
        'Ever_Bankrupt_or_Foreclose',
        'Reason_credit_card_refinancing', 'Reason_debt_conslidation', 'Reason_home_improvement',
        'Reason_major_purchase', 'Reason_other',
        'Employment_Status_part_time', 'Employment_Status_unemployed',
        'Employment_Sector_energy', 'Employment_Sector_finance', 'Employment_Sector_healthcare',
        'Employment_Sector_industrials', 'Employment_Sector_information_technology',
        'Employment_Sector_materials', 'Employment_Sector_real_estate', 'Employment_Sector_retail',
        'Employment_Sector_utilities', 'Employment_Sector_Unknown',
        'Lender_B', 'Lender_C'
    ]
    
    # 1. Add Missing Columns (Set to 0)
    # This handles cases where a category was not selected in the form.
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # 2. Final Alignment and Reordering (The Critical Fix for ValueError)
    # This ensures only the expected columns are kept, and they are in the exact order.
    df = df_encoded[feature_columns]
    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=False)

    # Use the aligned and encoded data for prediction
    prediction_proba = model.predict_proba(df)[:, 1]
    pred = (prediction_proba >= 0.5).astype(int) # Using the 0.5 threshold

    try:
        
        # 3. Display Results
        if pred == 1:
            st.success(f"Loan Approved — Probability: {prediction_proba:.2f}")
            st.balloons()
        else:
            st.error(f"Loan Not Approved — Probability: {prediction_proba:.2f}")

    except Exception as e:
        st.error("Prediction failed — see debug below.")
        st.write("Exception:", str(e))
        st.text(traceback.format_exc())
    # print result
    if pred == 1:
        st.success(f"Loan Approved - (Probability: {proba:.2f})")
        st.balloons()
    else:
        st.error(f"Loan Not Approved - (Probability: {proba:.2f})")
