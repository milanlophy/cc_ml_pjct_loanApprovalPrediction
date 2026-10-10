import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load CSS
with open("style.css", "r") as file:
    st.markdown(
        f"<style>{file.read()}</style>",
        unsafe_allow_html=True
    )

# Load dataset
df = pd.read_csv("loan_data.csv")

# Select features and target
X = df[["Income", "Credit_Score"]]
y = df["Loan_Approved"]

# Train the machine learning model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
model.fit(X, y)

# Application heading
st.markdown(
    '<h1 class="main-title">🏦 Loan Approval Prediction</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Project</p>',
    unsafe_allow_html=True
)

st.divider()

# Applicant details
st.subheader("Enter Applicant Details")

with st.form("loan_form"):
    income = st.number_input(
        "Monthly Income (₹)",
        min_value=0,
        max_value=1000000,
        value=30000,
        step=1000
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=10
    )

    submit = st.form_submit_button("Predict Loan Status")

# Prediction and eligibility checking
if submit:

    # Prepare input data
    applicant = pd.DataFrame(
        [[income, credit_score]],
        columns=["Income", "Credit_Score"]
    )

    # Machine Learning prediction
    prediction = model.predict(applicant)[0]

    st.subheader("1. Machine Learning Prediction")

    if prediction == 1:
        st.success("Model Prediction: Loan Approved")
    else:
        st.error("Model Prediction: Loan Not Approved")

    # Example eligibility rules
    st.subheader("2. Eligibility Rule Check")

    if income >= 30000 and credit_score >= 650:
        st.success("Meets the sample eligibility criteria")
    else:
        st.error("Does not meet the sample eligibility criteria")

    # Explain the rules
    st.info(
        "Sample rules: Monthly income must be at least "
        "₹30,000 and credit score must be at least 650."
    )

    st.caption(
        "The ML prediction and rule check are separate. "
        "These educational examples are not official bank "
        "criteria and must not be used for real lending decisions."
    )

# Display dataset
with st.expander("View Training Dataset"):
    st.dataframe(df, use_container_width=True)

st.markdown(
    '<p class="footer">Built with Python, Streamlit and Machine Learning</p>',
    unsafe_allow_html=True
)