import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
import time

# Load model and threshold
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
threshold = model_data["threshold"]

# Page configuration with custom theme
st.set_page_config(
    page_title="✨ Churn Prediction App",
    layout="centered",
    page_icon="🔮",
    initial_sidebar_state="expanded"
    default_theme="light" 
)

# Custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');
        
        .title-text {
            font-family: 'Poppins', sans-serif;
            font-size: 3rem !important;
            font-weight: 600 !important;
            color: #6a11cb !important;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .subtitle-text {
            font-family: 'Poppins', sans-serif;
            font-size: 1.2rem !important;
            color: #6a11cb !important;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%) !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
        }
        
        .input-container {
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .success-box {
            background: linear-gradient(45deg, #11998e, #38ef7d) !important;
            color: white !important;
            padding: 20px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            animation: pulse 2s infinite;
        }
        
        .danger-box {
            background: linear-gradient(45deg, #f12711, #f5af19) !important;
            color: white !important;
            padding: 20px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

# Animated header with GIF
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("https://example.com/Pay%20Per%20Click%20Digital%20Marketing.gif", 
             width=400, 
             caption="Customer Churn Prediction Analysis")

# App title with custom styling
st.markdown('<h1 class="title-text">✨ Churn Prediction Wizard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Predict customer churn with machine learning precision</p>', unsafe_allow_html=True)

# Input form with enhanced styling
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

    # Animated result display
    if prediction == 1:
        st.markdown(f"""
            <div class="danger-box">
                <h2 style='color: white; text-align: center;'>🚨 High Churn Risk</h2>
                <p style='color: white; text-align: center; font-size: 1.5rem;'>
                    Probability: {prediction_proba:.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
            <div class="success-box">
                <h2 style='color: white; text-align: center;'>✅ Loyal Customer</h2>
                <p style='color: white; text-align: center; font-size: 1.5rem;'>
                    Retention Probability: {(1 - prediction_proba):.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.snow()

    # Interactive pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Will Stay', 'Will Churn'],
        values=[1 - prediction_proba, prediction_proba],
        marker_colors=['#38ef7d', '#f12711'],
        hole=0.5,
        pull=[0.1 if prediction == 1 else 0, 0.1 if prediction == 0 else 0],
        textinfo='percent+label',
        hoverinfo='label+percent'
    )])
    
    fig.update_layout(
        title="Churn Probability Breakdown",
        width=500,
        height=500,
        showlegend=False,
        annotations=[dict(text=f"{prediction_proba:.0%}", x=0.5, y=0.5, font_size=40, showarrow=False)]
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Log prediction to MLflow
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Churn_Prediction_App")
        
        with mlflow.start_run(run_name="User_Prediction"):
            mlflow.log_params(encoded_input_df.to_dict(orient="records")[0])
            mlflow.log_metric("prediction_proba", float(prediction_proba))
            mlflow.log_metric("prediction_class", int(prediction))
    except:
        st.warning("Could not connect to MLflow tracking server")

# Add some decorative elements
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6a11cb; font-family: 'Poppins', sans-serif;">
        <p>🔮 Predict customer behavior with 90%+ accuracy</p>
        <p>📊 Get actionable insights to reduce churn</p>
        <p>💡 Powered by machine learning</p>
    </div>
""", unsafe_allow_html=True)
