import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import mlflow
import time
from datetime import datetime
import streamlit as st

# MUST be the first command
st.set_page_config(
    page_title="My App",
    layout="centered",
    initial_sidebar_state="auto"
)

# Only after page config you can do other stuff
st.title("My App")


# 2. تحميل النموذج
@st.cache_resource
def load_model():
    try:
        with open("final_stacked_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data["model"], model_data["threshold"]
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

model, threshold = load_model()

# 3. تنسيق واجهة المستخدم
def setup_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        :root {
            --primary: #6a11cb;
            --secondary: #2575fc;
            --success: #00b09b;
            --danger: #ff416c;
            --light-bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #2d3748;
            --border: #e2e8f0;
        }
        
        body, .stApp {
            font-family: 'Poppins', sans-serif;
            background-color: var(--light-bg);
            color: var(--text);
        }
        
        .title-text {
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--primary);
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .subtitle-text {
            font-size: 1.1rem;
            color: #4a5568;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .input-container {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 500 !important;
        }
        
        .success-box {
            background: linear-gradient(45deg, var(--success), #96c93d) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
        }
        
        .danger-box {
            background: linear-gradient(45deg, var(--danger), #ff4b2b) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.markdown('<h1 class="title-text">✨ Churn Prediction Wizard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle-text">Predict customer churn with machine learning precision</p>', unsafe_allow_html=True)

# 4. نموذج الإدخال
def get_user_input():
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            SeniorCitizen = st.selectbox("Is the customer a senior citizen?", [0, 1])
            Partner = st.selectbox("Has a partner?", ["Yes", "No"])
            Dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            
        with col2:
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, format="%.2f")
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, format="%.2f")
        
        st.markdown('</div>', unsafe_allow_html=True)

    return pd.DataFrame({
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    })

# 5. التنبؤ والتسجيل
def predict_and_log(input_df):
    # Encode data
    le = LabelEncoder()
    categorical_cols = ['Partner', 'Dependents', 'InternetService', 
                       'OnlineSecurity', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod']
    
    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])
    
    # Predict
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0
    
    # Display results
    if prediction == 1:
        st.markdown(f"""
            <div class="danger-box">
                <h2 style='text-align:center;margin-bottom:0.5rem'>🚨 High Churn Risk</h2>
                <p style='text-align:center;font-size:1.25rem;margin-bottom:0'>
                    Probability: {prediction_proba:.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="success-box">
                <h2 style='text-align:center;margin-bottom:0.5rem'>✅ Loyal Customer</h2>
                <p style='text-align:center;font-size:1.25rem;margin-bottom:0'>
                    Retention Probability: {(1-prediction_proba):.2%}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    fig = go.Figure(data=[go.Pie(
        labels=['Will Stay', 'Will Churn'],
        values=[1-prediction_proba, prediction_proba],
        marker_colors=['#00b09b', '#ff416c'],
        hole=0.5,
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        showlegend=False,
        margin=dict(t=30, b=0),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # MLflow logging
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        mlflow.set_experiment("Churn_Prediction_App")
        
        with mlflow.start_run(run_name=f"Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(input_df.to_dict(orient="records")[0])
            mlflow.log_metrics({
                "probability": float(prediction_proba),
                "prediction": int(prediction)
            })
    except Exception as e:
        st.warning(f"MLflow logging failed: {str(e)}")

# 6. الواجهة الرئيسية
def main():
    setup_ui()
    
    input_df = get_user_input()
    
    if st.button("✨ Predict Churn Probability"):
        if model is None:
            st.error("Model not loaded - cannot make predictions")
            return
            
        with st.spinner('Analyzing customer data...'):
            time.sleep(1)
            predict_and_log(input_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align:center;color:#4a5568;font-size:0.9rem'>
            <p>🔮 Predictive Analytics | 📊 Customer Insights | 🤖 ML Powered</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
