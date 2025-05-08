# app.py
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn

# Page configuration
st.set_page_config(page_title="Churn Prediction App", layout="centered")

# Apply external CSS for styling
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title & GIF
st.markdown("""
    <div class="header-container">
        <img src="https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/Pay%20Per%20Click%20Digital%20Marketing.gif" class="center-gif" alt="Customer Leaving">
        <h1 class="main-title">🔮 Churn Prediction App</h1>
        <p>Enter customer information to predict the likelihood of churn.</p>
    </div>
""", unsafe_allow_html=True)

# Load model and threshold
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

# Input form
def user_input():
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            Partner = st.selectbox("Has a partner?", ["Yes", "No"])
            Dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)

        with col2:
            SeniorCitizen = st.selectbox("Is the customer a senior citizen?", [0, 1])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
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
encoded_input_df = encode_input_data(input_df)

if st.button("🔍 Predict Now"):
    with mlflow.start_run():
        prediction_proba = model.predict_proba(encoded_input_df)[0][1]
        prediction = 1 if prediction_proba >= threshold else 0

        # Log with MLflow
        mlflow.log_params(input_df.iloc[0].to_dict())
        mlflow.log_metric("churn_probability", prediction_proba)
        mlflow.log_metric("prediction", prediction)

        # Show result
        st.markdown(f"""
            <div class='prediction-bubble {'success' if prediction == 0 else 'error'}'>
                {'✅ The customer is likely to stay with a probability of {:.2%}'.format(1 - prediction_proba)
                if prediction == 0 else
                '🚨 The customer is likely to churn with a probability of {:.2%}'.format(prediction_proba)}
            </div>
        """, unsafe_allow_html=True)

        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['No Churn', 'Churn'],
            values=[1 - prediction_proba, prediction_proba],
            marker_colors=['green', 'red'],
            hole=0.4
        )])
        fig.update_layout(title="Churn Probability", width=500, height=400)
        st.plotly_chart(fig)
