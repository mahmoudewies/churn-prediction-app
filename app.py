import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import time
import random

# Load model and threshold
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

# Page configuration
st.set_page_config(page_title="Churn Prediction App", layout="centered")

# Custom CSS for stunning effects
st.markdown("""
    <style>
    .header {
        text-align: center;
        font-size: 3rem;
        color: #f1c40f;
        font-family: 'Courier New', Courier, monospace;
        animation: colorChange 5s infinite;
    }

    @keyframes colorChange {
        0% { color: #f1c40f; }
        25% { color: #e74c3c; }
        50% { color: #2ecc71; }
        75% { color: #3498db; }
        100% { color: #f1c40f; }
    }

    .moving-box {
        position: relative;
        animation: moveBox 3s ease-in-out infinite;
        width: 150px;
        height: 150px;
        background-color: #9b59b6;
        margin: 20px auto;
    }

    @keyframes moveBox {
        0% { transform: translateX(-50%); }
        50% { transform: translateX(50%); }
        100% { transform: translateX(-50%); }
    }

    .stButton>button:hover {
        background-color: #3498db;
        color: white;
    }

    </style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown('<div class="header">🔮 Churn Prediction App</div>', unsafe_allow_html=True)

# Input form
def user_input():
    SeniorCitizen = st.selectbox("Is the customer a senior citizen?", [0, 1])
    Partner = st.selectbox("Has a partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)
    TotalServices = st.slider("Total Services Used", 0, 10, 5)

    data = pd.DataFrame({
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'TotalServices': [TotalServices]
    })
    return data

# Function to encode the input data
def encode_input_data(input_df):
    le = LabelEncoder()
    columns_to_encode = ['Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 
                         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                         'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    for column in columns_to_encode:
        input_df[column] = le.fit_transform(input_df[column])
    
    return input_df

input_df = user_input()

# Encode the input data
encoded_input_df = encode_input_data(input_df)

# Prediction
if st.button("🔍 Predict Now"):
    prediction_proba = model.predict_proba(encoded_input_df)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0

    # Play sound effect based on prediction (if churn)
    if prediction == 1:
        st.audio("churn_sound.mp3")  # Add a sound effect when the customer churns

    st.subheader("📊 Result:")
    if prediction == 1:
        st.error(f"🚨 The customer is likely to churn with a probability of {prediction_proba:.2%}")
    else:
        st.success(f"✅ The customer is likely to stay with a probability of {(1 - prediction_proba):.2%}")

    # GIF of customer leaving
    st.image("churn_gif.gif", caption="Customer Leaving", use_column_width=True)

    # Pie chart with animation
    fig = go.Figure(data=[go.Pie(
        labels=['No Churn', 'Churn'],
        values=[1 - prediction_proba, prediction_proba],
        marker_colors=['green', 'red'],
        hole=0.4
    )])
    fig.update_layout(title="Churn Probability", width=500, height=400)
    st.plotly_chart(fig)

    # Add a moving box as a decoration
    st.markdown('<div class="moving-box"></div>', unsafe_allow_html=True)
