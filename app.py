import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow

# Load model and threshold
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

# Page configuration
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("🔮 Churn Prediction App")
st.markdown("Enter customer information to predict the likelihood of churn.")

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

# Add the prediction button
if st.button("🔍 Predict Now"):
    prediction_proba = model.predict_proba(encoded_input_df)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0

    st.subheader("📊 Result:")
    if prediction == 1:
        st.error(f"🚨 The customer is likely to churn with a probability of {prediction_proba:.2%}")
    else:
        st.success(f"✅ The customer is likely to stay with a probability of {(1 - prediction_proba):.2%}")

    # Pie chart for the prediction
    fig = go.Figure(data=[go.Pie(
        labels=['No Churn', 'Churn'],
        values=[1 - prediction_proba, prediction_proba],
        marker_colors=['green', 'red'],
        hole=0.4
    )])
    fig.update_layout(title="Churn Probability", width=500, height=400)
    st.plotly_chart(fig)

    # Log prediction to MLflow
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Churn_Prediction_App")

    with mlflow.start_run(run_name="User_Prediction"):
        mlflow.log_params(encoded_input_df.to_dict(orient="records")[0])
        mlflow.log_metric("prediction_proba", float(prediction_proba))
        mlflow.log_metric("prediction_class", int(prediction))
