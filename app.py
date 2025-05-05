import streamlit as st
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# تسجيل النموذج باستخدام MLflow
def log_model_to_mlflow(model, threshold):
    mlflow.start_run()
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("threshold", threshold)
    mlflow.end_run()

# تحميل النموذج
with open("final_stacked_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
threshold = model_data["threshold"]

# تسجيل النموذج في MLflow
log_model_to_mlflow(model, threshold)

# واجهة المستخدم في Streamlit
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("🔮 Churn Prediction App")
st.markdown("Enter customer information to predict the likelihood of churn.")

# إدخال البيانات
def user_input():
    # ... (إدخال البيانات هنا كما في كودك السابق)
    pass

input_df = user_input()

# التنبؤ
if st.button("🔍 Predict Now"):
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0
    st.subheader("📊 Result:")
    if prediction == 1:
        st.error(f"🚨 The customer is likely to churn with a probability of {prediction_proba:.2%}")
    else:
        st.success(f"✅ The customer is likely to stay with a probability of {(1 - prediction_proba):.2%}")

    # رسم بياني للنتيجة
    fig = go.Figure(data=[go.Pie(
        labels=['No Churn', 'Churn'],
        values=[1 - prediction_proba, prediction_proba],
        marker_colors=['green', 'red'],
        hole=0.4
    )])
    fig.update_layout(title="Churn Probability", width=500, height=400)
    st.plotly_chart(fig)

# مراقبة الأداء
if st.button("🔍 Monitor Performance"):
    plot_model_performance()

# إعادة التدريب
if st.button("🔄 Retrain Model"):
    st.write("Re-training the model...")
    time.sleep(2)  # محاكاة تحميل البيانات
    data = pd.read_csv("new_data.csv")  # استبدل هذا بمصدر بياناتك الفعلي
    accuracy = retrain_model(data)
    st.write(f"Model re-trained! Accuracy: {accuracy:.2%}")
