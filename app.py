import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
import time

# Page config
st.set_page_config(page_title="Churn Prediction App", layout="centered")

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header GIF and Title
gif_url = "Pay Per Click Digital Marketing.gif"
st.markdown(f"""
<div class="header-container">
    <img src="{gif_url}" class="center-gif" alt="churn gif">
    <h1 class="main-title">Customer Churn Predictor</h1>
</div>
""", unsafe_allow_html=True)

# Load model and threshold
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

def user_input():
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            SeniorCitizen = st.selectbox("Senior Citizen?", [0, 1])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure", 0, 72, 12)
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        with col2:
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0)
            TotalServices = st.slider("Total Services Used", 0, 10, 5)

        submitted = st.form_submit_button("🔍 Predict Now")
        
    if submitted:
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
    return None

def encode_input_data(input_df):
    le = LabelEncoder()
    columns_to_encode = ['Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 
                         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                         'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    for column in columns_to_encode:
        input_df[column] = le.fit_transform(input_df[column])
    
    return input_df

input_df = user_input()
if input_df is not None:
    encoded_input_df = encode_input_data(input_df)

    # Predict
    with st.spinner("🔮 Making prediction..."):
        time.sleep(2)
        prediction_proba = model.predict_proba(encoded_input_df)[0][1]
        prediction = 1 if prediction_proba >= threshold else 0

    # Log with MLflow
    with mlflow.start_run():
        mlflow.log_params(encoded_input_df.iloc[0].to_dict())
        mlflow.log_metric("churn_probability", float(prediction_proba))
        mlflow.log_metric("churn_prediction", int(prediction))

    # Result section
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-bubble error">
            🚨 <strong>Churn Alert!</strong> The customer is likely to leave with a probability of {prediction_proba:.2%}.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-bubble success">
            ✅ <strong>All Good!</strong> The customer is likely to stay with a probability of {(1 - prediction_proba):.2%}.
        </div>
        """, unsafe_allow_html=True)

    # Chart
    fig = go.Figure(data=[go.Pie(
        labels=['No Churn', 'Churn'],
        values=[1 - prediction_proba, prediction_proba],
        marker_colors=['green', 'red'],
        hole=0.4
    )])
    fig.update_layout(title="Churn Probability", width=500, height=400)
    st.plotly_chart(fig)
