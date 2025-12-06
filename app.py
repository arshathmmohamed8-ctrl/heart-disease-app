import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart Risk Predictor", layout="centered")

model = joblib.load("model.pkl")

st.title("❤️ Heart Disease Risk Prediction")

def get_user_input():
    age = st.number_input("Age", 20, 90, 50)
    sex = st.selectbox("Sex (1=M,0=F)", [0, 1])
    cp = st.selectbox("Chest Pain (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting BP", 90, 200, 120)
    chol = st.number_input("Cholesterol", 120, 600, 200)
    fbs = st.selectbox("Fasting Sugar > 120 mg/dl (1=Yes,0=No)", [0,1])
    restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [0,1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.5, 1.0)
    slope = st.selectbox("Slope (0-2)", [0,1,2])
    ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
    thal = st.selectbox("Thal (1-3)", [1,2,3])

    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

if st.button("Predict"):
    input_data = get_user_input()
    proba = model.predict_proba(input_data)[0][1]

    if proba >= 0.70:
        st.error(f"🔴 HIGH RISK — {proba:.2f}")
    elif proba >= 0.40:
        st.warning(f"🟡 MEDIUM RISK — {proba:.2f}")
    else:
        st.success(f"🟢 LOW RISK — {proba:.2f}")
