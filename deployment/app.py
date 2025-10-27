import streamlit as st
import pandas as pd
import joblib

# تحميل الموديل
model = joblib.load(r"C:\Users\nice\Desktop\m\models\churn_model.pkl")

# إعداد الصفحة
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Telecom Customer Churn Prediction App")
st.write("تطبيق للتنبؤ باحتمالية مغادرة العميل (Churn) بناءً على بياناته وخدماته.")

# تقسيم الواجهة لأعمدة
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Has Partner", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col3:
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
    MonthlyCharges = st.number_input("Monthly Charges (EGP)", min_value=0.0, step=10.0)
    TotalCharges = st.number_input("Total Charges (EGP)", min_value=0.0, step=100.0)

# عند الضغط على زر التنبؤ
if st.button("🔍 Predict"):
    # حساب الـ service_plus
    service_plus = 1 if any([
        OnlineSecurity == "Yes",
        OnlineBackup == "Yes",
        DeviceProtection == "Yes",
        TechSupport == "Yes",
        StreamingTV == "Yes",
        StreamingMovies == "Yes"
    ]) else 0

    # تجهيز البيانات في DataFrame بنفس شكل التدريب
    df_input = pd.DataFrame({
        'gender': [1 if gender == "Male" else 0],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [1 if Partner == "Yes" else 0],
        'Dependents': [1 if Dependents == "Yes" else 0],
        'tenure': [tenure],
        'PhoneService': [1 if PhoneService == "Yes" else 0],
        'MultipleLines': [1 if MultipleLines == "Yes" else 0],
        'InternetService': [0 if InternetService == "DSL" else (1 if InternetService == "Fiber optic" else 2)],
        'OnlineSecurity': [1 if OnlineSecurity == "Yes" else 0],
        'OnlineBackup': [1 if OnlineBackup == "Yes" else 0],
        'DeviceProtection': [1 if DeviceProtection == "Yes" else 0],
        'TechSupport': [1 if TechSupport == "Yes" else 0],
        'StreamingTV': [1 if StreamingTV == "Yes" else 0],
        'StreamingMovies': [1 if StreamingMovies == "Yes" else 0],
        'Contract': [0 if Contract == "Month-to-month" else (1 if Contract == "One year" else 2)],
        'PaperlessBilling': [1 if PaperlessBilling == "Yes" else 0],
        'PaymentMethod': [
            0 if PaymentMethod == "Electronic check" else
            1 if PaymentMethod == "Mailed check" else
            2 if PaymentMethod == "Bank transfer (automatic)" else 3
        ],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'service_plus': [service_plus]
    })

    # التنبؤ
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    # عرض النتيجة
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ العميل **من المحتمل أن يغادر**.\nاحتمال المغادرة: {probability:.2%}")
    else:
        st.success(f"✅ العميل **من غير المحتمل أن يغادر**.\nاحتمال البقاء: {(1 - probability):.2%}")
