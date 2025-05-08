import streamlit as st
import base64
from PIL import Image
import pickle
import pandas as pd

# ========== Page Config ==========
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide",
)

# ========== Load Font and Apply CSS ==========

st.markdown(
    """
    <style>
    @import url('https://raw.githubusercontent.com/MahmoudOweis11/Churn-Prediction-App/main/KickerTrial-Black.ttf');

    html, body, [class*="css"] {
        font-family: 'Kicker Trial Black', sans-serif;
        background-color: #0e1117;
        color: white;
    }

    .main {
        background-color: #0e1117;
    }

    h1, h2, h3 {
        color: #00c0f2;
    }

    .stButton > button {
        color: white;
        background-color: #00c0f2;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #009ec3;
        color: white;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Load Model ==========
with open("models/final_churn_model.pkl", "rb") as file:
    model = pickle.load(file)

# ========== Title ==========
st.markdown("<h1 style='text-align: center;'>📉 Customer Churn Prediction</h1>", unsafe_allow_html=True)

# ========== Center GIF ==========
gif_path = "images/Pay Per Click Digital Marketing.gif"
with open(gif_path, "rb") as f:
    gif_data = f.read()
    encoded_gif = base64.b64encode(gif_data).decode("utf-8")
st.markdown(
    f"""
    <div style="display: flex; justify-content: center; margin-bottom: 30px;">
        <img src="data:image/gif;base64,{encoded_gif}" width="600"/>
    </div>
    """,
    unsafe_allow_html=True
)

# ========== Input Fields ==========
st.sidebar.header("📋 Enter Customer Data")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# ========== Prediction Button ==========
if st.button("✨ Predict Churn Probability"):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
    })

    prediction = model.predict_proba(input_data)[0][1]
    churn_risk = round(prediction * 100, 2)

    st.success(f"🧠 Churn Probability: **{churn_risk}%**")
