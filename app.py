import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
import base64
import time

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

# --- Custom CSS for Light Theme ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
        
        /* Light theme variables */
        :root {{
            --primary: #6a11cb;
            --secondary: #2575fc;
            --background: #f8f9fa;
            --card: #ffffff;
            --text: #212529;
            --border: #dee2e6;
        }}
        
        /* Base styles */
        body {{
            background-color: var(--background) !important;
            color: var(--text) !important;
            font-family: 'Poppins', sans-serif;
        }}
        
        /* Title styles */
        .title-text {{
            font-family: 'Poppins', sans-serif;
            font-size: 2.5rem !important;
            font-weight: 600 !important;
            color: var(--primary) !important;
            text-align: center;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        /* Input container */
        .input-container {{
            background-color: var(--card) !important;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}
        
        /* Interactive elements */
        .stButton>button {{
            background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            transition: all 0.3s ease !important;
            width: 100%;
        }}
        
        .stButton>button:hover {{
            transform: scale(1.02) !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
        }}
        
        /* Result boxes */
        .success-box {{
            background: linear-gradient(45deg, #11998e, #38ef7d) !important;
            color: white !important;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
            animation: pulse 2s infinite;
        }}
        
        .danger-box {{
            background: linear-gradient(45deg, #f12711, #f5af19) !important;
            color: white !important;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
            animation: pulse 2s infinite;
        }}
        
        /* Animations */
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.02); }}
            100% {{ transform: scale(1); }}
        }}
        
        /* Form elements */
        .stSelectbox, .stSlider, .stNumberInput, 
        .stTextInput>div>div>input {{
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 8px 12px !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- Animated GIF Display ---
def display_animated_gif(gif_url):
    try:
        st.markdown(
            f'<div style="display: flex; justify-content: center; margin: 1rem 0 2rem 0;">'
            f'<img src="{gif_url}" alt="churn prediction" width="400" style="border-radius: 12px;">'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not load animated GIF: {e}")

# --- Header Section ---
display_animated_gif("https://github.com/mahmoudewies/churn-prediction-app/blob/main/Pay%20Per%20Click%20Digital%20Marketing.gif?raw=true")

st.markdown('<h1 class="title-text">Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: var(--text); font-size: 1.1rem; margin-bottom: 2rem;">'
            'Predict customer churn with machine learning precision</p>', unsafe_allow_html=True)

# --- Input Form ---
def user_input_form():
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Demographics")
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], help="Is the customer 65 or older?")
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            
            st.subheader("Internet Services")
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            
        with col2:
            st.subheader("Additional Services")
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            
            st.subheader("Billing Information")
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, format="%.2f")
        
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, format="%.2f")
        TotalServices = st.slider("Total Services Used", 0, 10, 5)
        
        st.markdown('</div>', unsafe_allow_html=True)

    return pd.DataFrame({
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

# --- Cache LabelEncoder for Faster Encoding ---
@st.cache_data
def get_encoder():
    le = LabelEncoder()
    return le

def encode_input_data(input_df):
    le = get_encoder()  # Reuse cached encoder
    categorical_cols = ['Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])
    return input_df

# --- Make Prediction Without Delay ---
def make_prediction(input_data):
    # Removed time.sleep() to speed up the process
    encoded_data = encode_input_data(input_data.copy())
    proba = model.predict_proba(encoded_data)[0][1]
    prediction = 1 if proba >= threshold else 0
    return prediction, proba

# --- Results Display ---
def show_results(prediction, probability):
    if prediction == 1:
        st.markdown(f"""
            <div class="danger-box">
                <h2 style='color: white; text-align: center;'>🚨 High Churn Risk</h2>
                <p style='color: white; text-align: center; font-size: 1.5rem;'>
                    Probability: {probability:.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
            <div class="success-box">
                <h2 style='color: white; text-align: center;'>✅ Loyal Customer</h2>
                <p style='color: white; text-align: center; font-size: 1.5rem;'>
                    Retention Probability: {(1 - probability):.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.snow()
    
    # Interactive chart
    fig = go.Figure(data=[go.Pie(
        labels=['Retain', 'Churn'],
        values=[1 - probability, probability],
        marker_colors=['#38ef7d', '#f12711'],
        hole=0.5,
        pull=[0.1 if prediction == 1 else 0, 0.1 if prediction == 0 else 0],
        textinfo='percent+label',
        hoverinfo='label+percent'
    )])
    
    fig.update_layout(
        title="Churn Probability Breakdown",
        height=350,
        margin=dict(t=0, b=0, l=0, r=0)
    )
    st.plotly_chart(fig)

# --- Main Flow ---
if __name__ == "__main__":
    user_data = user_input_form()
    
    if st.button("Make Prediction"):
        with st.spinner("Predicting..."):
            prediction, probability = make_prediction(user_data)
        show_results(prediction, probability)
