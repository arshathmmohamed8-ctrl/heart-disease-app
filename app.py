import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL ---------------- #
model, scaler = joblib.load("final.pkl")  # Must match the file you saved

# ---------------- APP UI ---------------- #
st.title("❤️ Heart Disease Risk Prediction")

st.subheader("Enter Patient Details")

age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (1=Male,0=Female)", [0,1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting BP", 50, 250)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar >120 (1=True, 0=False)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 50, 250)
exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [0,1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, format="%.2f")
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels Colored (0-3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (1-3)", [1,2,3])

# ---------------- PREDICT ---------------- #
if st.button("Predict"):

    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    scaled = scaler.transform(user_input)              # <--- Here your scaler is applied
    proba = model.predict_proba(scaled)[0][1]          # <--- Your line is HERE
    pred = model.predict(scaled)[0]

    st.subheader("🔍 Result:")
    if pred == 1:
        st.error(f"🚨 HIGH RISK — Probability: {proba:.2f}")
    else:
        st.success(f"🟢 LOW RISK — Probability: {proba:.2f}")
