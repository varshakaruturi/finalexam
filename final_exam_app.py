import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("final_exam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define categorical options directly
categorical_options = {
    "Reason": ["Debt Consolidation", "Home Improvement", "Car Purchase", "Medical", "Other"],
    "Employment_Status": ["Employed", "Self-Employed", "Unemployed", "Student", "Retired"],
    "Lender": ["Bank A", "Bank B", "Bank C", "Credit Union", "Other"],
    "Fico_Score_group": ["300-579", "580-669", "670-739", "740-799", "800-850"],
    "Employment_Sector": ["Private", "Government", "Non-Profit", "Self-Employed", "Other"]
}

    # --- Separate columns ---
numerical_cols = [
        'applications', 'Granted_Loan_Amount', 'Requested_Loan_Amount',
        'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment',
        'granted_requested_ratio', 'housing_to_income_ratio'
]
categorical_cols = [
        'Reason', 'Employment_Status', 'Lender', 'Fico_Score_group',
        'Employment_Sector', 'Ever_Bankrupt_or_Foreclose'
]

feature_columns = numerical_cols + [
    # one-hot encoded categorical columns
    'Reason_Home Improvement',
    'Reason_Car Purchase',
    'Reason_Medical',
    'Reason_Other',
    'Employment_Status_Self-Employed',
    'Employment_Status_Unemployed',
    'Employment_Status_Student',
    'Employment_Status_Retired',
    'Lender_Bank B',
    'Lender_Bank C',
    'Lender_Credit Union',
    'Lender_Other',
    'Fico_Score_group_580-669',
    'Fico_Score_group_670-739',
    'Fico_Score_group_740-799',
    'Fico_Score_group_800-850',
    'Employment_Sector_Government',
    'Employment_Sector_Non-Profit',
    'Employment_Sector_Self-Employed',
    'Employment_Sector_Other',
    'Ever_Bankrupt_or_Foreclose_1'
]

# Title
st.markdown(
    "<h1 style='text-align: center; background-color: #f0f2f6; padding: 10px; color: #31333F;'><b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Applicant's Details")

# --- Numerical Inputs ---
st.subheader("Numerical Features")
applications = st.number_input("Applications (APPLICATIONS)", min_value=1, max_value=10, value=1)
granted_loan_amount = st.slider("Granted Loan Amount", min_value=5000, max_value=2000000, value=50000, step=1000)
requested_loan_amount = st.slider("Requested Loan Amount", min_value=5000, max_value=2500000, value=60000, step=1000)
fico_score = st.slider("FICO Score", min_value=300, max_value=850, value=650, step=1)
monthly_gross_income = st.slider("Monthly Gross Income", min_value=0, max_value=20000, value=5000, step=100)
monthly_housing_payment = st.slider("Monthly Housing Payment", min_value=300, max_value=50000, value=1500, step=100)

# --- Categorical Inputs ---
st.subheader("Categorical Features")
reason = st.selectbox("Reason for Loan", categorical_options['Reason'])
employment_status = st.selectbox("Employment Status", categorical_options['Employment_Status'])
lender = st.selectbox("Lender", categorical_options['Lender'])
fico_score_group = st.selectbox("FICO Score Group", categorical_options['Fico_Score_group'])
employment_sector = st.selectbox("Employment Sector", categorical_options['Employment_Sector'])
ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclose", [0, 1],
                                         format_func=lambda x: "Yes" if x == 1 else "No")


# --- Prediction Button ---
if st.button("Predict Loan Approval"):

    # Build input_df
    input_df = pd.DataFrame({
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

    # Feature engineering
    input_df['granted_requested_ratio'] = input_df['Granted_Loan_Amount'] / input_df['Requested_Loan_Amount']
    input_df['housing_to_income_ratio'] = input_df['Monthly_Housing_Payment'] / input_df['Monthly_Gross_Income']
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df.fillna(0, inplace=True)

    # One-hot encode
    input_categorical_ohe = pd.get_dummies(
        input_df[['Reason','Employment_Status','Lender','Fico_Score_group','Employment_Sector','Ever_Bankrupt_or_Foreclose']], 
        drop_first=True
    )

    # Combine numerical + categorical
final_input = pd.concat([input_df[numerical_cols], input_categorical_ohe], axis=1)

    # Reindex to match training features
final_input = final_input.reindex(columns=feature_columns, fill_value=0)

    # --- Make prediction ---
prediction = model.predict(final_input)
prediction_proba = model.predict_proba(final_input)[:, 1]

    # --- Show results ---
st.subheader("Prediction Results")
if prediction[0] == 1:
    st.success(f"Loan Approval: YES (Probability: {prediction_proba[0]:.2f})")
    st.balloons()
else:
    st.error(f"Loan Approval: NO (Probability: {prediction_proba[0]:.2f})")

st.write("Note: Probability closer to 1 indicates higher likelihood of approval.")
