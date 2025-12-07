import streamlit as st
import pickle
import pandas as pd

# --- Load model ---
with open("final_exam_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Feature columns exactly as trained ---
feature_columns = [
    'Granted_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment',
    'Ever_Bankrupt_or_Foreclose',
    'Reason_credit_card_refinancing', 'Reason_debt_conslidation', 'Reason_home_improvement',
    'Reason_major_purchase', 'Reason_other',
    'Employment_Status_employed', 'Employment_Status_self_employed',
    'Employment_Status_unemployed', 'Employment_Status_student', 'Employment_Status_retired',
    'Employment_Sector_energy', 'Employment_Sector_financials', 'Employment_Sector_health_care',
    'Employment_Sector_industrials', 'Employment_Sector_information_technology',
    'Employment_Sector_materials', 'Employment_Sector_real_estate', 'Employment_Sector_utilities',
    'Lender_B', 'Lender_C'
]

# --- Streamlit UI ---
st.markdown("<h1 style='text-align:center;'>Loan Approval Prediction</h1>", unsafe_allow_html=True)

# Numeric inputs
granted_loan = st.number_input("Granted Loan Amount", 5000, 2000000, 50000, step=1000)
fico_score = st.slider("FICO Score", 300, 850, 650)
monthly_income = st.number_input("Monthly Gross Income", 0, 20000, 5000)
monthly_housing = st.number_input("Monthly Housing Payment", 300, 50000, 1500)
ever_bankrupt = st.selectbox("Ever Bankrupt or Foreclose", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Categorical inputs
reason = st.selectbox("Reason for Loan", ['credit_card_refinancing', 'debt_conslidation', 'home_improvement', 'major_purchase', 'other'])
employment_status = st.selectbox("Employment Status", ['employed', 'self_employed', 'unemployed', 'student', 'retired'])
employment_sector = st.selectbox("Employment Sector", ['energy', 'financials', 'health_care', 'industrials', 'information_technology', 'materials', 'real_estate', 'utilities'])
lender = st.selectbox("Lender", ['B', 'C', 'Other'])

if st.button("Predict Loan Approval"):
    # Initialize all columns to 0/False
    input_dict = {col: 0 for col in feature_columns}

    # Fill numeric columns
    input_dict['Granted_Loan_Amount'] = granted_loan
    input_dict['FICO_score'] = fico_score
    input_dict['Monthly_Gross_Income'] = monthly_income
    input_dict['Monthly_Housing_Payment'] = monthly_housing
    input_dict['Ever_Bankrupt_or_Foreclose'] = ever_bankrupt

    # Fill one-hot encoded categorical columns
    # Reason
    reason_col = f"Reason_{reason}"
    if reason_col in feature_columns:
        input_dict[reason_col] = 1
    # Employment Status
    status_col = f"Employment_Status_{employment_status}"
    if status_col in feature_columns:
        input_dict[status_col] = 1
    # Employment Sector
    sector_col = f"Employment_Sector_{employment_sector}"
    if sector_col in feature_columns:
        input_dict[sector_col] = 1
    # Lender
    lender_col = f"Lender_{lender}"
    if lender_col in feature_columns:
        input_dict[lender_col] = 1

    # Convert to DataFrame with proper column order
    df_input = pd.DataFrame([input_dict], columns=feature_columns)

    # Predict
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    # Display results
    if prediction == 1:
        st.success(f"Loan Approved ✅ (Probability: {proba:.2f})")
        st.balloons()
    else:
        st.error(f"Loan Not Approved ❌ (Probability: {proba:.2f})")
