import streamlit as st
import numpy as np
import joblib

bundle = joblib.load("heart_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

st.title("❤️ Heart Disease Risk Prediction")

age = st.number_input("Age", 20, 90, 45)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
cp = st.slider("Chest Pain Type (0-3)", 0,3,1)
trestbps = st.number_input("Resting Blood Pressure",80,200,120)
chol = st.number_input("Cholesterol",100,400,200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (1=Yes,0=No)", [0,1])
restecg = st.selectbox("Rest ECG Results (0-2)", [0,1,2])
thalach = st.number_input("Maximum Heart Rate",60,220,150)
exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [0,1])
oldpeak = st.number_input("ST Depression",0.0,6.0,1.0,0.1)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels Colored (0-3)",[0,1,2,3])
thal = st.selectbox("Thalassemia (0=Normal,1=Fixed,2=Reversible)",[0,1,2])

data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
scaled = scaler.transform(data)
pred = model.predict(scaled)[0]

if pred == 1:
    st.error("🔴 HIGH RISK — Symptoms indicate possible heart disease.")
else:
    st.success("🟢 LOW RISK — No significant signs detected.")
