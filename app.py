import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
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

# --- Custom CSS ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

        html, body, [class*="css"]  {{
            font-family: 'Poppins', sans-serif;
            background-color: #f8f9fa;
            color: #212529;
        }}

        .title-text {{
            font-size: 2.5rem;
            font-weight: 600;
            text-align: center;
            background: linear-gradient(45deg, #6a11cb, #2575fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .input-container {{
            background-color: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border: 1px solid #dee2e6;
            margin-bottom: 2rem;
        }}

        .stButton>button {{
            background: linear-gradient(45deg, #6a11cb, #2575fc);
            color: white;
            font-weight: 600;
            border-radius: 12px;
            padding: 12px 24px;
            width: 100%;
        }}

        .stButton>button:hover {{
            transform: scale(1.02);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .success-box {{
            background: linear-gradient(45deg, #11998e, #38ef7d);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            animation: pulse 2s infinite;
        }}

        .danger-box {{
            background: linear-gradient(45deg, #f12711, #f5af19);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.02); }}
            100% {{ transform: scale(1); }}
        }}
    </style>
""", unsafe_allow_html=True)

# --- Display GIF from GitHub raw URL ---
def display_animated_gif_from_url(url):
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin: 1rem 0 2rem 0;">
            <img src="{url}" alt="GIF" width="400" style="border-radius: 12px;">
        </div>
        """,
        unsafe_allow_html=True
    )

gif_url = "https://raw.githubusercontent.com/mahmoudewies/churn-prediction-app/main/Pay%20Per%20Click%20Digital%20Marketing.gif"
display_animated_gif_from_url(gif_url)

# --- Header ---
st.markdown('<h1 class="title-text">Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.1rem; margin-bottom: 2rem;">Predict customer churn with machine learning precision.</p>', unsafe_allow_html=True)

# --- Input Form ---
def user_input_form():
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Demographics")
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
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

            st.subheader("Billing")
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

# --- Encode Input ---
def encode_input_data(input_df):
    le = LabelEncoder()
    categorical_cols = ['Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])
    return input_df

# --- Make Prediction ---
def make_prediction(input_data):
    with st.spinner('Analyzing customer data...'):
        time.sleep(1.5)
        encoded_data = encode_input_data(input_data.copy())
        proba = model.predict_proba(encoded_data)[0][1]
        prediction = 1 if proba >= threshold else 0
        return prediction, proba

# --- Show Results ---
def show_results(prediction, probability):
    if prediction == 1:
        st.markdown(f"""
            <div class="danger-box">
                <h2 style='text-align: center;'>🚨 High Churn Risk</h2>
                <p style='text-align: center; font-size: 1.5rem;'>Probability: {probability:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
            <div class="success-box">
                <h2 style='text-align: center;'>✅ Low Churn Risk</h2>
                <p style='text-align: center; font-size: 1.5rem;'>Retention Probability: {(1 - probability):.2%}</p>
            </div>
        """, unsafe_allow_html=True)
        st.snow()

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
        height=400,
        showlegend=False,
        annotations=[dict(
            text=f"{probability:.0%}", 
            x=0.5, y=0.5, 
            font_size=36, 
            showarrow=False
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Main App Flow ---
input_data = user_input_form()

if st.button("✨ Predict Churn Probability", type="primary", use_container_width=True):
    prediction, probability = make_prediction(input_data)
    show_results(prediction, probability)

    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Churn_Predictions")
        with mlflow.start_run():
            mlflow.log_params(input_data.iloc[0].to_dict())
            mlflow.log_metric("churn_probability", probability)
            mlflow.log_metric("prediction", prediction)
    except:
        pass

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #212529; margin-top: 2rem;">
        <p>🔮 Powered by Machine Learning | 📊 Data-Driven Insights</p>
    </div>
""", unsafe_allow_html=True)
