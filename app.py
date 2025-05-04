import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# تحميل النموذج
model = joblib.load("final_stacked_model.pkl")

# عنوان الصفحة
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("🔮 Churn Prediction App")
st.markdown("أدخل بيانات العميل لتوقع احتمالية ترك الخدمة.")

# إدخال البيانات
def user_input():
    SeniorCitizen = st.selectbox("هل العميل كبير سن؟", [0, 1])
    Partner = st.selectbox("هل لديه شريك؟", ["Yes", "No"])
    Dependents = st.selectbox("هل لديه معالون؟", ["Yes", "No"])
    tenure = st.slider("مدة الاشتراك (بالأشهر)", 0, 72, 12)
    InternetService = st.selectbox("نوع الإنترنت", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("الأمان على الإنترنت", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("نسخ احتياطي", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("حماية الأجهزة", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("دعم تقني", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("خدمة بث تلفزيوني", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("خدمة أفلام", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("نوع العقد", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("فاتورة إلكترونية", ["Yes", "No"])
    PaymentMethod = st.selectbox("طريقة الدفع", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("الرسوم الشهرية", min_value=0.0)
    TotalCharges = st.number_input("إجمالي الرسوم", min_value=0.0)
    TotalServices = st.slider("عدد الخدمات", 0, 10, 5)

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

input_df = user_input()

# التوقع
if st.button("🔍 توقع الآن"):
    prediction_proba = model.predict_proba(input_df)[0][1]  # احتمالية churn
    prediction = model.predict(input_df)[0]

    st.subheader("📊 النتيجة:")
    if prediction == 1:
        st.error(f"🚨 العميل مرشح لترك الخدمة (Churn) بنسبة {prediction_proba:.2%}")
    else:
        st.success(f"✅ العميل سيبقى (No Churn) بنسبة {(1 - prediction_proba):.2%}")

    # رسم بياني دائري بالاحتمالات
    fig = go.Figure(data=[go.Pie(labels=['No Churn', 'Churn'],
                                 values=[1 - prediction_proba, prediction_proba],
                                 marker_colors=['green', 'red'],
                                 hole=0.4)])
    fig.update_layout(title="احتمالية ترك الخدمة", width=500, height=400)
    st.plotly_chart(fig)
