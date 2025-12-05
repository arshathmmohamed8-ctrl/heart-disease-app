import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Risk Predictor")


# ================= UI INPUTS =================

age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol", 100, 600, 220)
fbs = st.selectbox("Fasting Blood Sugar (1=Yes,0=No)", [1, 0])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0=Normal,1=Fixed,2=Reversible)", [0, 1, 2])


# =============== PREDICTION BUTTON ===============

if st.button("🔍 Predict Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]], dtype=float)

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]

    if pred == 1:
        st.error("🔴 HIGH RISK — Symptoms suggest possible heart disease.")
    else:
        st.success("🟢 LOW RISK — No major symptoms detected.")
