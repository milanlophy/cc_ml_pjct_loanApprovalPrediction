
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

# Features and target
X = df[["Income", "Credit_Score"]]
y = df["Loan_Approved"]

# Train Machine Learning model
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
        value=None,
        step=1000
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=None,
        value=None,
        step=10
    )

    submit = st.form_submit_button("Predict Loan Status")

# Prediction
if submit:

    st.subheader("Loan Decision")

    # Eligibility rules have priority
    if income < 30000 or credit_score < 650:

        st.error("❌ Loan Rejected")
        st.write("Reason: Eligibility criteria not satisfied.")

        if income < 30000:
            st.write("• Monthly income must be at least ₹30,000.")

        if credit_score < 650:
            st.write("• Credit score must be at least 650.")

    else:
        # Applicant meets the example eligibility rules
        applicant = pd.DataFrame(
            [[income, credit_score]],
            columns=["Income", "Credit_Score"]
        )

        # Machine Learning prediction
        prediction = model.predict(applicant)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
            st.write(
                "Eligibility criteria satisfied and "
                "the ML model predicts approval."
            )
        else:
            st.error("❌ Loan Rejected")
            st.write(
                "Eligibility criteria satisfied, but "
                "the ML model predicts rejection."
            )

    st.caption(
        "Educational project only. These thresholds are "
        "illustrative, not official bank lending rules."
    )

# Display dataset
with st.expander("View Training Dataset"):
    st.dataframe(df, use_container_width=True)

st.markdown(
    '<p class="footer">Built with Python, Streamlit and Machine Learning</p>',
    unsafe_allow_html=True
)