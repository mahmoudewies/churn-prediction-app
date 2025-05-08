import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# MUST be the first command
st.set_page_config(
    page_title="✨ Churn Prediction Wizard",
    layout="centered",
    page_icon="🔮",
    initial_sidebar_state="expanded"
)

# ============== Model Monitoring ==============
class ModelMonitor:
    def __init__(self):
        self.performance_history = []
        
    def log_performance(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'f1_score': f1
        })
        
        if len(self.performance_history) > 5 and np.mean([x['f1_score'] for x in self.performance_history[-5:]]) < 0.7:
            st.sidebar.error("🚨 Alert: Model performance degradation detected!")

if "monitor" not in st.session_state:
    st.session_state["monitor"] = ModelMonitor()

monitor = st.session_state["monitor"]

# ============== Retraining Strategy ==============
def retrain_model():
    with st.sidebar.expander("🔧 Model Retraining"):
        if st.button("Trigger Retraining", key="retrain_btn"):
            with st.spinner("Retraining model..."):
                time.sleep(5)  # Simulate retraining
                st.success("Model retrained successfully!")
                st.balloons()

# Display GIF in the center
st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 2rem;">
        <img src="https://raw.githubusercontent.com/mahmoudewies/churn-prediction-app/main/Pay%20Per%20Click%20Digital%20Marketing%20(1).gif" alt="GIF" width="600">
    </div>
""", unsafe_allow_html=True)

# Load model and threshold
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

# Get expected features from the model (if available)
try:
    expected_features = model.feature_names_in_
except AttributeError:
    expected_features = None

# ============== Custom CSS ==============
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        
        body, .stApp {
            background-color: #121212;
            color: #f0f0f0;
            font-family: 'Poppins', sans-serif;
        }
        
        .title-text {
            font-size: 2.7rem !important;
            font-weight: 600;
            color: #bb86fc;
            text-align: center;
            margin-bottom: 0.3rem;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
        }
        
        .subtitle-text {
            font-size: 1.1rem !important;
            color: #d1c4e9;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* ... (بقية أنماط CSS الخاصة بك كما هي) ... */
    </style>
""", unsafe_allow_html=True)

# ============== App Header ==============
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="title-text">✨ Churn Prediction Wizard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Predict customer churn with machine learning precision</p>', unsafe_allow_html=True)

# ============== Input Form ==============
def get_user_input():
    # تأكد من تخزين المدخلات في session_state
    if 'SeniorCitizen' not in st.session_state:
        st.session_state['SeniorCitizen'] = 0
    if 'Partner' not in st.session_state:
        st.session_state['Partner'] = 'No'
    if 'Dependents' not in st.session_state:
        st.session_state['Dependents'] = 'No'
    if 'tenure' not in st.session_state:
        st.session_state['tenure'] = 12
    # كرر نفس العملية لبقية المدخلات (الخصائص الأخرى)

    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            SeniorCitizen = st.selectbox("Is the customer a senior citizen?", [0, 1], key="senior", index=st.session_state['SeniorCitizen'])
            Partner = st.selectbox("Has a partner?", ["Yes", "No"], key="partner", index=["Yes", "No"].index(st.session_state['Partner']))
            Dependents = st.selectbox("Has dependents?", ["Yes", "No"], key="dependents", index=["Yes", "No"].index(st.session_state['Dependents']))
            tenure = st.slider("Tenure (months)", 0, 72, st.session_state['tenure'], key="tenure")
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet", index=["DSL", "Fiber optic", "No"].index(st.session_state.get('InternetService', "No")))
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="security", index=["Yes", "No", "No internet service"].index(st.session_state.get('OnlineSecurity', "No internet service")))
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup", index=["Yes", "No", "No internet service"].index(st.session_state.get('OnlineBackup', "No internet service")))
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device", index=["Yes", "No", "No internet service"].index(st.session_state.get('DeviceProtection', "No internet service")))
        
        with col2:
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech", index=["Yes", "No", "No internet service"].index(st.session_state.get('TechSupport', "No internet service")))
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="stream_tv", index=["Yes", "No", "No internet service"].index(st.session_state.get('StreamingTV', "No internet service")))
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="stream_movies", index=["Yes", "No", "No internet service"].index(st.session_state.get('StreamingMovies', "No internet service")))
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract", index=["Month-to-month", "One year", "Two year"].index(st.session_state.get('Contract', "Month-to-month")))
            PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"], key="paperless", index=["Yes", "No"].index(st.session_state.get('PaperlessBilling', "No")))
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], key="payment", index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(st.session_state.get('PaymentMethod', "Electronic check")))
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, format="%.2f", key="monthly", value=st.session_state.get('MonthlyCharges', 50.0))
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, format="%.2f", key="total", value=st.session_state.get('TotalCharges', 1000.0))
            TotalServices = st.slider("Total Services Used", 0, 10, st.session_state.get('TotalServices', 5), key="services")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # تخزين المدخلات في session_state لتجنب اختفائها بعد التحديث
    st.session_state['SeniorCitizen'] = SeniorCitizen
    st.session_state['Partner'] = Partner
    st.session_state['Dependents'] = Dependents
    st.session_state['tenure'] = tenure
    st.session_state['InternetService'] = InternetService
    st.session_state['OnlineSecurity'] = OnlineSecurity
    st.session_state['OnlineBackup'] = OnlineBackup
    st.session_state['DeviceProtection'] = DeviceProtection
    st.session_state['TechSupport'] = TechSupport
    st.session_state['StreamingTV'] = StreamingTV
    st.session_state['StreamingMovies'] = StreamingMovies
    st.session_state['Contract'] = Contract
    st.session_state['PaperlessBilling'] = PaperlessBilling
    st.session_state['PaymentMethod'] = PaymentMethod
    st.session_state['MonthlyCharges'] = MonthlyCharges
    st.session_state['TotalCharges'] = TotalCharges
    st.session_state['TotalServices'] = TotalServices

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
    
    # Ensure columns match model expectations
    if expected_features is not None:
        missing_cols = set(expected_features) - set(data.columns)
        if missing_cols:
            for col in missing_cols:
                data[col] = 0  # Add missing columns with default value
        data = data[expected_features]  # Reorder columns
    
    return data

# ============== Prediction Logic ==============
def make_prediction(input_df):
    # Encode categorical data
    le = LabelEncoder()
    categorical_cols = input_df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])
    
    # Ensure numeric types
    numeric_cols = input_df.select_dtypes(exclude=['object']).columns
    input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Make prediction
    pred_proba = model.predict_proba(input_df)[:, 1]
    return pred_proba

# ============== Display Predictions ==============
user_data = get_user_input()
prediction_proba = make_prediction(user_data)

if prediction_proba >= threshold:
    prediction = "Yes"
    prediction_color = "#FF6F61"
else:
    prediction = "No"
    prediction_color = "#4CAF50"

# Log the performance of the model
monitor.log_performance([1], [prediction == "Yes"])

# Show the result
st.markdown(f"<h3 style='text-align: center; color: {prediction_color};'>Will this customer churn? {prediction}</h3>", unsafe_allow_html=True)

# Trigger retraining if required
retrain_model()
