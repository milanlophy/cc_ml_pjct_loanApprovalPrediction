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

# Applicant form
st.subheader("Enter Applicant Details")

with st.form("loan_form"):
    income_input = st.text_input(
        "Monthly Income (₹)",
        placeholder="Enter monthly income"
    )

    credit_input = st.text_input(
        "Credit Score",
        placeholder="Enter credit score"
    )

    submit = st.form_submit_button("Predict Loan Status")

# Process form submission
if submit:

    # Required-field validation
    if not income_input.strip():
        st.error("Monthly Income: This field is required.")

    if not credit_input.strip():
        st.error("Credit Score: This field is required.")

    # Continue only when both fields are filled
    if income_input.strip() and credit_input.strip():

        try:
            income = float(income_input)
            credit_score = float(credit_input)

            # Validate input values
            if income < 0:
                st.error(
                    "Monthly Income must be zero or greater."
                )

            elif not credit_score.is_integer():
                st.error("Credit Score must be a whole number.")

            # elif not 300 <= credit_score <= 850:
            #     st.error(
            #         "Credit Score must be between 300 and 850."
            #     )

            else:
                st.subheader("Loan Decision")

                # Eligibility rules have priority
                if income < 30000 or credit_score < 650:

                    st.error("❌ Loan Rejected")
                    st.write(
                        "Reason: Eligibility criteria not satisfied."
                    )

                    if income < 30000:
                        st.write(
                            "Monthly income must be at least ₹30,000."
                        )

                    if credit_score < 650:
                        st.write(
                            "Credit score must be at least 650."
                        )

                else:
                    # Prepare input for ML model
                    applicant = pd.DataFrame(
                        [[income, int(credit_score)]],
                        columns=["Income", "Credit_Score"]
                    )

                    # Predict loan approval
                    prediction = model.predict(applicant)[0]

                    if prediction == 1:
                        st.success("✅ Loan Approved")
                        st.write(
                            "Eligibility criteria satisfied "
                            "and the ML model predicts approval."
                        )
                    else:
                        st.error("❌ Loan Rejected")
                        st.write(
                            "Eligibility criteria satisfied, "
                            "but the ML model predicts rejection."
                        )

        except ValueError:
            st.error("Please enter valid numeric values.")

# Display training dataset
with st.expander("View Training Dataset"):
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown(
    '<p class="footer">Built with Python, Streamlit and Machine Learning</p>',
    unsafe_allow_html=True
)

st.caption(
    "Educational demonstration only. These sample eligibility "
    "rules are not official bank lending criteria."
)