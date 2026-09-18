import streamlit as st
import pandas as pd
import joblib


model = joblib.load("loan_risk_model.pkl")

st.set_page_config(
    page_title="AI Loan Risk Detector",
    page_icon="💳",
    layout="centered"
)

st.title("💳 AI Loan Risk Detector")
st.write("Enter applicant details to estimate loan default risk.")



age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000
)

home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

emp_length = st.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    value=3.0
)

loan_intent = st.selectbox(
    "Loan Purpose",
    [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    value=12.5
)

default_history = st.selectbox(
    "Previous Default",
    ["N", "Y"]
)

credit_history = st.number_input(
    "Credit History Length (Years)",
    min_value=0,
    value=4
)


if income > 0:
    loan_percent_income = loan_amount / income
else:
    loan_percent_income = 0

st.write(
    f"Loan-to-Income Ratio: **{loan_percent_income:.2f}**"
)

if st.button("Analyze Loan Risk"):

    customer = pd.DataFrame({
        "person_age": [age],
        "person_income": [income],
        "person_home_ownership": [home_ownership],
        "person_emp_length": [emp_length],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_amnt": [loan_amount],
        "loan_int_rate": [interest_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [default_history],
        "cb_person_cred_hist_length": [credit_history]
    })

    # Predict probability
    probability = model.predict_proba(customer)[0][1]

    probability_percent = probability * 100


    if probability < 0.30:
        risk = "LOW RISK"

    elif probability < 0.60:
        risk = "MEDIUM RISK"

    else:
        risk = "HIGH RISK"


    st.subheader("Risk Analysis")

    st.metric(
        "Predicted Default Probability",
        f"{probability_percent:.2f}%"
    )

    st.progress(float(probability))

    if risk == "LOW RISK":
        st.success(f"Risk Level: {risk}")

    elif risk == "MEDIUM RISK":
        st.warning(f"Risk Level: {risk}")

    else:
        st.error(f"Risk Level: {risk}")

    st.caption(
        "This is an educational ML prediction and should not "
        "be used as a real lending decision."
    )