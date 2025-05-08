import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

# ----------------- تحميل النموذج -----------------
import pickle
global model
# محاولة فتح النموذج وتحديد ما إذا كان ملف pickle فارغًا
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except EOFError:
    print("The model file is empty or corrupted.")
except Exception as e:
    print(f"An error occurred: {e}")


# ----------------- دالة لتسجيل الأداء -----------------
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

# ----------------- دالة لتحميل المدخلات -----------------
# ----------------- قائمة الأعمدة التي يتوقعها النموذج -----------------
expected_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MonthlyCharges']

def get_user_input():
    # توجيه المستخدم لإدخال البيانات
    SeniorCitizen = st.selectbox("Is the customer a senior citizen?", [0, 1])
    Partner = st.selectbox("Has a partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, format="%.2f")
    
    # تحويل البيانات المدخلة إلى DataFrame
    input_data = pd.DataFrame({
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges]
    })
    
    # التأكد من تطابق الأعمدة
    input_data = input_data[expected_columns]
    
    return input_data


# ----------------- دالة للتنبؤ -----------------
def make_prediction(input_df):
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if prediction_proba >= threshold else 0
    return prediction_proba, prediction

# ----------------- الكود الرئيسي -----------------
def main():
    monitor = ModelMonitor()

    input_df = get_user_input()

    if st.button("✨ Predict Churn"):
        prediction_proba, prediction = make_prediction(input_df)

        if prediction == 1:
            st.markdown(f"🚨 High Churn Risk! Probability: {prediction_proba:.2%}")
        else:
            st.markdown(f"✅ Loyal Customer! Retention Probability: {1 - prediction_proba:.2%}")

        # نموذج النتائج
        ground_truth = st.radio("What was the actual outcome?", ["Stayed", "Churned"], index=0)
        ground_truth_binary = 0 if ground_truth == "Stayed" else 1

        # تسجيل الأداء
        monitor.log_performance([ground_truth_binary], [prediction])

        # عرض النتائج
        if len(monitor.performance_history) > 0:
            last_performance = monitor.performance_history[-1]
            st.write(f"Accuracy: {last_performance['accuracy']:.2%}")
            st.write(f"F1 Score: {last_performance['f1_score']:.2%}")

if __name__ == "__main__":
    main()
