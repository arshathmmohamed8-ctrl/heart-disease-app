import streamlit as st
import joblib
import numpy as np

st.title("🧡 Heart Disease Prediction System")

# Load Model (Pipeline: Scaler + Logistic Regression)
model = joblib.load("model.pkl")  # your model file

st.subheader("Enter Patient Clinical Data:")

# UI Input — Must Match EXACT CSV Feature Order
age = st.number_input("Age", 0,150, 45)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80,200,120)
chol = st.number_input("Cholesterol Level", 100,600,240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=Yes,0=No)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Maximum Heart Rate", 60,220,150)
exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [0,1])
oldpeak = st.number_input("ST Depression", 0.0,10.0,1.0,step=0.1)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels Colored (0-3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (0=Normal,1=Fixed,2=Reversible,3=Others)", [0,1,2,3])

if st.button("Predict"):

    proba = model.predict_proba(input_data)[0][1]  # probability of disease

    if proba > 0.65:
        st.error(f"🔴 HIGH RISK — Model Confidence: {proba:.2f}")
    elif proba < 0.35:
        st.success(f"🟢 LOW RISK — Model Confidence: {proba:.2f}")
    else:
        st.warning(f"🟡 MEDIUM RISK — Model Confidence: {proba:.2f}")
