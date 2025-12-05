import streamlit as st
import joblib
import numpy as np

# Load model (ONLY ONE FILE USED)
model = joblib.load("heart_model.pkl")

st.title("Heart Disease Prediction System")

st.write("Enter details below and click Predict")

# === INPUT FIELDS ===
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 (1=Yes,0=No)", [0,1])
restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Angina (1=Yes,0=No)", [0,1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels Colored (0-3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (0=Normal,1=Fixed,2=Reversible)", [0,1,2])

# Combine values
input_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

# === PREDICT BUTTON ===
if st.button("Predict"):
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("🔴 HIGH RISK — Possible Heart Disease.")
    else:
        st.success("🟢 LOW RISK — No major symptoms detected.")
