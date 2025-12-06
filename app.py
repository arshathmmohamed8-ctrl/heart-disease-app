import streamlit as st
import joblib
import numpy as np

st.title("Heart Disease Risk Prediction")

# Load trained pipeline model (contains scaler inside)
model = joblib.load("final.pkl")

age = st.number_input("Age", 1, 120, 40)
sex = st.selectbox("Sex (1=Male,0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting BP", 80, 200, 130)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar (1=True,0=False)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes,0=No)", [0,1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (1-3)", [1,2,3])

if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0][1]

    st.subheader("Result:")
    if prediction == 1:
        st.error(f"🚨 HIGH RISK — Probability: {proba:.2f}")
    else:
        st.success(f"🟢 LOW RISK — Probability: {proba:.2f}")
