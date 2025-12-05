import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")    # <--- uses Logistic Regression pipeline

st.title("❤️ Heart Disease Prediction (Logistic Regression Model)")

age = st.number_input("Age", 20, 100, 40)
sex = st.selectbox("Sex (1=Male,0=Female)", [1,0])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol Level", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True,0=False)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 210, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [0,1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels Colored (0-3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (1-3)", [1,2,3])

if st.button("Predict"):
    data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    result = model.predict(data)[0]

    if result == 1:
        st.error("🔴 HIGH RISK — Possible heart disease")
    else:
        st.success("🟢 LOW RISK — Normal condition")
