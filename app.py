import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and model features
model = joblib.load("stroke_predictor_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Stroke Prediction App", layout="centered")

st.title("🧠 Stroke Prediction App")
st.write("Fill in the details to predict stroke risk.")

# User input fields
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

age = st.number_input("Age", min_value=1.0, max_value=120.0, value=30.0)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=28.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Map inputs to numerical
data = {
    "gender": 1 if gender == "Male" else (2 if gender == "Other" else 0),
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "ever_married": 1 if ever_married == "Yes" else 0,
    "Residence_type": 1 if residence_type == "Urban" else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": 0 if smoking_status == "never smoked" else
                      1 if smoking_status == "formerly smoked" else
                      2 if smoking_status == "smokes" else 3
}

# One-hot encode work_type
work_type_encoded = {
    'work_type_Private': 0,
    'work_type_Self-employed': 0,
    'work_type_children': 0,
    'work_type_Never_worked': 0
}
selected_key = f"work_type_{work_type}"
if selected_key in work_type_encoded:
    work_type_encoded[selected_key] = 1

# Combine all inputs
input_data = {**data, **work_type_encoded}
input_df = pd.DataFrame([input_data])

# Ensure all model features are present
for feature in model_features:
    if feature not in input_df.columns:
        input_df[feature] = 0
input_df = input_df[model_features]

# Prediction
if st.button("🔍 Predict Stroke Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    st.write(f" **Stroke Probability: {probability:.2f}%**")

    if probability >= 50:
        st.error("🔴 High Risk of Stroke Detected!")
    elif probability >= 25:
        st.warning("🟠 Moderate Risk of Stroke.")
    else:
        st.success("✅ Low Risk of Stroke Detected.")
    
