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

import traceback
import pandas as pd
import numpy as np  # <-- FIX: NumPy must be imported if you use np.inf

# --- ensure model is loaded ---
if 'model' not in globals():
    # If this is run in a cell/script where 'model' isn't loaded higher up
    # you might need to load it here (e.g., model = pd.read_pickle('final_exam_model.pkl'))
    st.error("Model not loaded. Make sure the top of this file loads final_exam_model.pkl into variable `model`.")
else:
    try:
        # Assuming 'df' (the input DataFrame from Streamlit form) is defined just before this block
        if 'df' not in locals() and 'df' not in globals():
            st.error("Input DataFrame 'df' is not present. Make sure you create 'df' before predicting.")
        else:
            X_raw = df.copy()

            # --- Feature Engineering (Check for required columns) ---
            if 'Granted_Loan_Amount' in X_raw.columns and 'Requested_Loan_Amount' in X_raw.columns:
                if 'granted_requested_ratio' not in X_raw.columns:
                    # Avoid division by zero
                    X_raw['Requested_Loan_Amount'] = X_raw['Requested_Loan_Amount'].replace(0, 1e-6)
                    X_raw['granted_requested_ratio'] = X_raw['Granted_Loan_Amount'] / X_raw['Requested_Loan_Amount']
            
            if 'Monthly_Housing_Payment' in X_raw.columns and 'Monthly_Gross_Income' in X_raw.columns:
                if 'housing_to_income_ratio' not in X_raw.columns:
                    # avoid division by zero
                    X_raw['Monthly_Gross_Income'] = X_raw['Monthly_Gross_Income'].replace(0, 1e-6)
                    X_raw['housing_to_income_ratio'] = X_raw['Monthly_Housing_Payment'] / X_raw['Monthly_Gross_Income']

            # convert infinite to zero
            X_raw.replace([np.inf, -np.inf], 0, inplace=True)

            # --- Encoding ---
            # NOTE: pd.get_dummies needs to be robustly applied ONLY to the categorical columns
            # For simplicity, we keep your current implementation, assuming non-numeric columns are categorical.
            X_enc = pd.get_dummies(X_raw, drop_first=False)

            # --- Feature Alignment ---
            feature_columns = None
            if hasattr(model, 'feature_names_in_'):
                feature_columns = model.feature_names_in_
            elif hasattr(model, 'named_steps') and 'model' in model.named_steps and hasattr(model.named_steps['model'], 'feature_names_in_'):
                # Check for models inside scikit-learn pipelines
                feature_columns = model.named_steps['model'].feature_names_in_
            else:
                # Fallback: relies on a pre-saved list, which is the most robust method
                st.warning("Model does not expose `feature_names_in_`. Using best-effort fallback.")
                # You should load a saved feature list here if possible, e.g.,
                # feature_columns = load_feature_list_from_file()
                feature_columns = list(X_enc.columns) # Fallback is risky

            if feature_columns is None:
                 st.error("Cannot determine expected feature list. Prediction halted.")
                 return # Stop execution if features are unknown

            # Align columns safely (add missing, drop extras, and maintain order)
            X_aligned = X_enc.reindex(columns=feature_columns, fill_value=0)

            # quick debug (comment out in production)
            st.write("DEBUG: model expects", len(feature_columns), "features")
            st.write("DEBUG: X_aligned shape:", X_aligned.shape)

            # --- Predictions ---
            # Ensure X_aligned has only one row if the input 'df' was a single user input
            pred = model.predict(X_aligned)[0]
            
            # some models do not have predict_proba; guard it
            proba = None
            if hasattr(model, 'predict_proba'):
                # Assumes the positive class (approved) is at index 1
                proba = model.predict_proba(X_aligned)[0, 1]

            # --- display results ---
            if pred == 1:
                if proba is not None:
                    st.success(f"Loan Approved — Probability: {proba:.2f}")
                else:
                    st.success("Loan Approved")
                st.balloons()
            else:
                if proba is not None:
                    st.error(f"Loan Not Approved — Probability: {proba:.2f}")
                else:
                    st.error("Loan Not Approved")

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
