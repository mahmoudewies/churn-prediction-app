import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
import time
import base64
import requests
from io import BytesIO

# --- Load Model ---
@st.cache_resource
def load_model():
    with open("final_stacked_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["threshold"]

model, threshold = load_model()

# --- Page Config ---
st.set_page_config(
    page_title="✨ Customer Churn Predictor",
    layout="centered",
    page_icon="🔮",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
        
        :root {{
            --primary: #6a11cb;
            --secondary: #2575fc;
            --background: #f8f9fa;
            --card: #ffffff;
            --text: #212529;
        }}
        
        body {{
            font-family: 'Poppins', sans-serif;
            background-color: var(--background);
        }}
        
        .title-text {{
            font-size: 2.5rem;
            font-weight: 600;
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }}
        
        .stButton>button {{
            background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- Animated GIF from URL ---
gif_url = "https://raw.githubusercontent.com/mahmoudewies/churn-prediction-app/main/Pay%20Per%20Click%20Digital%20Marketing.gif"

st.markdown(
    f'<div style="display: flex; justify-content: center; margin: 1rem 0;">'
    f'<img src="{gif_url}" alt="churn prediction gif" width="400" style="border-radius: 12px;">'
    f'</div>',
    unsafe_allow_html=True
)

# --- App Title ---
st.markdown('<h1 class="title-text">Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: var(--text); margin-bottom: 2rem;">'
            'Predict customer churn with machine learning</p>', unsafe_allow_html=True)

# --- Input Form ---
def user_input_form():
    with st.form("user_input"):
        col1, col2 = st.columns(2)
        
        with col1:
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        with col2:
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, format="%.2f")
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, format="%.2f")
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        return pd.DataFrame({
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'InternetService': [InternetService],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges]
        })
    return None

# --- Prediction ---
def predict_churn(input_df):
    le = LabelEncoder()
    categorical = ['Partner', 'Dependents', 'InternetService', 'Contract', 'PaperlessBilling']
    for col in categorical:
        input_df[col] = le.fit_transform(input_df[col])
    
    proba = model.predict_proba(input_df)[0][1]
    return proba

# --- Main App ---
input_data = user_input_form()

if input_data is not None:
    with st.spinner('Analyzing...'):
        time.sleep(1)
        churn_prob = predict_churn(input_data)
        
        if churn_prob >= threshold:
            st.error(f"🚨 Churn Risk: {churn_prob:.1%}")
        else:
            st.success(f"✅ Retention Probability: {1-churn_prob:.1%}")
        
        fig = go.Figure(go.Pie(
            labels=['Stay', 'Churn'],
            values=[1-churn_prob, churn_prob],
            hole=0.5,
            marker_colors=['#4CAF50', '#F44336']
        ))
        st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center;'>"
            "🔮 Powered by Machine Learning | "
            "📊 Data Science App</div>", 
            unsafe_allow_html=True)
