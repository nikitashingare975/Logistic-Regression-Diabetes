import streamlit as st
import pickle
import numpy as np

# Load saved artifacts
model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Diabetes Prediction App")

# User inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 40.0, 300.0, 120.0)
bp = st.number_input("Blood Pressure", 20.0, 150.0, 70.0)
skin = st.number_input("Skin Thickness", 5.0, 100.0, 20.0)
insulin = st.number_input("Insulin", 10.0, 900.0, 80.0)
bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.05, 2.5, 0.5)
age = st.number_input("Age", 10, 120, 30)

# Predict
if st.button("Predict"):
    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(data)
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    if pred == 1:
        st.error(f"Diabetic (Probability: {prob:.2%})")
    else:
        st.success(f"Not Diabetic (Probability: {prob:.2%})")
