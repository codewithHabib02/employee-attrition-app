
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Simple styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #0b1f3a;
    font-size: 36px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #667085;
    margin-bottom: 30px;
}

.result {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    background-color: white;
    border: 1px solid #e4e7ec;
}

.result h2 {
    color: #0b1f3a;
}

.footer {
    text-align: center;
    color: #667085;
    font-size: 14px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model
# -----------------------------
pipeline = joblib.load("attrition_model.pkl")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">Employee Attrition Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Employee Attrition Prediction'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Employee information
# -----------------------------
st.subheader("Employee Information")

col1, col2 = st.columns(2)

with col1:

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

    stock_option = st.selectbox(
        "Stock Option Level",
        [0, 1, 2, 3]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

with col2:

    years_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=1
    )

    years_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=3
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=20000,
        value=3000,
        step=100
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

# -----------------------------
# Prediction
# -----------------------------
st.write("")

if st.button(
    "Predict Employee Attrition",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "OverTime": [overtime],
        "JobLevel": [job_level],
        "StockOptionLevel": [stock_option],
        "MaritalStatus": [marital_status],
        "YearsWithCurrManager": [years_manager],
        "YearsAtCompany": [years_company],
        "MonthlyIncome": [monthly_income],
        "EEAge": [age],
        "JobRole": [job_role]
    })

    prediction = pipeline.predict(input_data)[0]

    probability = pipeline.predict_proba(input_data)[0][1]

    # -----------------------------
    # Result
    # -----------------------------
    if prediction == 1:
        result = "Likely to Leave"
    else:
        result = "Likely to Stay"

    st.write("")

    st.markdown(
        f"""
        <div class="result">

        <h2>{result}</h2>

        <p>
        Attrition Probability
        </p>

        <h2>{probability:.1%}</h2>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(probability))

# -----------------------------
# Footer
# -----------------------------
st.write("")

st.markdown(
    """
    <div class="footer">
         • Habibulie • Alias: Ibn Leda
    </div>
    """,
    unsafe_allow_html=True
)