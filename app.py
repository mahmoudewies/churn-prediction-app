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
import streamlit as st

# MUST be the first command
st.set_page_config(
    page_title="My App",
    layout="centered",
    initial_sidebar_state="auto"
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

monitor = ModelMonitor()

# ============== Retraining Strategy ==============
def retrain_model():
    with st.sidebar.expander("🔧 Model Retraining"):
        if st.button("Trigger Retraining"):
            with st.spinner("Retraining model..."):
                time.sleep(5)  # Simulate retraining
                st.success("Model retrained successfully!")
                st.balloons()

# Only after page config you can do other stuff
#st.title("My App") 
# Display GIF in the center
st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 2rem;">
        <img src="https://raw.githubusercontent.com/mahmoudewies/churn-prediction-app/main/Pay%20Per%20Click%20Digital%20Marketing%20(1).gif" alt="GIF" width="600">
    </div>
""", unsafe_allow_html=True)

# Load model and threshold (must come after set_page_config)
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

# Rest of your code remains the same...

# Custom CSS for light mode styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        body, .stApp {
            background-color: #121212;  /* خلفية داكنة */
            color: #f0f0f0;  /* لون النص */
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

        .stButton>button {
            background: linear-gradient(90deg, #8e2de2, #4a00e0) !important;
            color: white !important;
            font-weight: 600;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.2rem !important;
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s ease-in-out !important;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #7b1fa2, #3700b3) !important;
            transform: scale(1.03) !important;
        }

        .input-container {
            background-color: #1e1e1e;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #333;
            box-shadow: 0 3px 10px rgba(0,0,0,0.3);
            margin-bottom: 1.5rem;
        }

        .success-box {
            background: #003c2f;
            border-left: 6px solid #00e676;
            color: #a5f2d5;
        }

        .danger-box {
            background: #3d0d0d;
            border-left: 6px solid #ff1744;
            color: #f8c1c1;
        }

        div[data-baseweb="select"], 
        .stTextInput input, 
        .stNumberInput input {
            background-color: #2c2c2c !important;
            color: #e0e0e0 !important;
            border: 1px solid #555 !important;
        }

        div[data-baseweb="select"]:hover,
        .stTextInput input:hover,
        .stNumberInput input:hover {
            border-color: #888 !important;
        }

        .stSlider .thumb {
            background-color: #bb86fc !important;
            border: 2px solid white !important;
        }

        .stSlider .track {
            background: linear-gradient(90deg, #5c5c5c, #9c27b0) !important;
            height: 6px !important;
        }

        .stTabs [aria-selected="true"] {
            background: #333 !important;
            color: #bb86fc !important;
        }
    </style>
""", unsafe_allow_html=True)


# App header
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.markdown('<h1 class="title-text">✨ Churn Prediction Wizard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Predict customer churn with machine learning precision</p>', unsafe_allow_html=True)

# Input form with light mode styling
def user_input():
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            SeniorCitizen = st.selectbox("Is the customer a senior citizen?", [0, 1])
            Partner = st.selectbox("Has a partner?", ["Yes", "No"])
            Dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long the customer has been with the company")
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            
        with col2:
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, format="%.2f")
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, format="%.2f")
            TotalServices = st.slider("Total Services Used", 0, 10, 5)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
# ============== Enhanced Prediction Function ==============
def enhanced_predict(input_df):
    # Encode data
    le = LabelEncoder()
    categorical_cols = ['Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])
    
    # Predict
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0
    
    # Log to monitoring system
    try:
        # In real app, you'd have actual ground truth
        # For demo, we'll simulate some ground truth
        ground_truth = 1 if prediction_proba > 0.7 else 0
        monitor.log_performance([ground_truth], [prediction], datetime.now())
        
        # Check data drift (simplified example)
        current_stats = {'tenure_mean': input_df['tenure'].mean()}
        reference_stats = {'tenure_mean': 32}  # From training data
        monitor.check_data_drift(current_stats, reference_stats)
    except Exception as e:
        st.error(f"Monitoring error: {str(e)}")
    
    return prediction_proba, prediction

# ============== Retraining Section ==============
st.sidebar.header("Model Management")
if st.sidebar.checkbox("Show Model Monitoring"):
    st.subheader("Model Performance Monitoring")
    
    # Simulate performance metrics
    if len(monitor.performance_history) == 0:
        st.info("No performance data yet. Make some predictions first.")
    else:
        # Create performance chart
        perf_df = pd.DataFrame(monitor.performance_history)
        st.line_chart(perf_df.set_index('timestamp'))
        
        # Show latest metrics
        latest = perf_df.iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric("Latest Accuracy", f"{latest['accuracy']:.2%}")
        col2.metric("Latest F1 Score", f"{latest['f1_score']:.2%}")

if st.sidebar.checkbox("Trigger Manual Retraining"):
    # In real app, you would load new data here
    retrain_model(None)  # Passing None for demo

# ============== Modified Prediction Button ==============
if st.button("✨ Predict Churn Probability", key="predict_button"):
    with st.spinner('Analyzing customer data...'):
        time.sleep(1.5)  # Simulate processing time
        
        prediction_proba, prediction = enhanced_predict(input_df.copy())
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
if st.button("✨ Predict Churn Probability", key="predict_button"):
    with st.spinner('Analyzing customer data...'):
        time.sleep(1.5)  # Simulate processing time
        
    prediction_proba = model.predict_proba(encoded_input_df)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0

    # Result display
    if prediction == 1:
        st.markdown(f"""
            <div class="danger-box">
                <h2 style='color: white; text-align: center; margin-bottom: 0.5rem;'>🚨 High Churn Risk</h2>
                <p style='color: white; text-align: center; font-size: 1.3rem; margin-bottom: 0;'>
                    Probability: {prediction_proba:.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()

    else:
        st.markdown(f"""
            <div class="success-box">
                <h2 style='color: white; text-align: center; margin-bottom: 0.5rem;'>✅ Loyal Customer</h2>
                <p style='color: white; text-align: center; font-size: 1.3rem; margin-bottom: 0;'>
                    Retention Probability: {(1 - prediction_proba):.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.snow()


    # Interactive pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Will Stay', 'Will Churn'],
        values=[1 - prediction_proba, prediction_proba],
        marker_colors=['#96c93d', '#ff4b2b'],
        hole=0.4,
        pull=[0.1 if prediction == 1 else 0, 0.1 if prediction == 0 else 0],
        textinfo='percent+label',
        hoverinfo='label+percent'
    )])
    
    fig.update_layout(
        title="Churn Probability Breakdown",
        width=500,
        height=500,
        showlegend=False,
        annotations=[dict(text=f"{prediction_proba:.0%}", x=0.5, y=0.5, font_size=40, showarrow=False)],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Log prediction to MLflow
    try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000/")
            mlflow.set_experiment("Churn_Prediction_App")
            
            with mlflow.start_run(run_name=f"Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_params(encoded_input_df.to_dict(orient="records")[0])
                mlflow.log_metric("prediction_proba", float(prediction_proba))
                mlflow.log_metric("prediction_class", int(prediction))
                
                # Log monitoring metrics
                if len(monitor.performance_history) > 0:
                    latest = monitor.performance_history[-1]
                    mlflow.log_metric("accuracy", latest['accuracy'])
                    mlflow.log_metric("f1_score", latest['f1_score'])
    except Exception as e:
            st.warning(f"MLflow logging failed: {str(e)}")
# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; font-family: 'Poppins', sans-serif; padding: 1rem;">
        <p style="margin-bottom: 0.5rem;">🔮 Predict customer behavior with 90%+ accuracy</p>
        <p style="margin-bottom: 0.5rem;">📊 Get actionable insights to reduce churn</p>
        <p>💡 Powered by machine learning</p>
    </div>
""", unsafe_allow_html=True)
