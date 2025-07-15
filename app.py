import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature list
model = joblib.load("stroke_predictor_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Stroke Prediction App")

st.markdown("### Enter patient health details to assess stroke risk")

# User Inputs
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", 1, 100)
hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
heart_disease = st.selectbox("Heart Disease", ['No', 'Yes'])
ever_married = st.selectbox("Ever Married", ['No', 'Yes'])
work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
avg_glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, step=0.1)
bmi = st.slider("BMI", 10.0, 60.0, step=0.1)
smoking_status = st.selectbox("Smoking Status", ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

# Encode inputs
data = {
    'gender': 1 if gender == 'Male' else 0,
    'age': age,
    'hypertension': 1 if hypertension == 'Yes' else 0,
    'heart_disease': 1 if heart_disease == 'Yes' else 0,
    'ever_married': 1 if ever_married == 'Yes' else 0,
    'Residence_type': 1 if residence_type == 'Urban' else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': {
        'never smoked': 0,
        'formerly smoked': 1,
        'smokes': 2,
        'Unknown': 3
    }[smoking_status]
}

# One-hot encode work_type
work_type_encoded = {
    'work_type_Private': 0,
    'work_type_Self-employed': 0,
    'work_type_children': 0,
    'work_type_Govt_job': 0,
    'work_type_Never_worked': 0
}
selected_col = f"work_type_{work_type}"
if selected_col in work_type_encoded:
    work_type_encoded[selected_col] = 1

# Combine all inputs
input_data = {**data, **work_type_encoded}
input_df = pd.DataFrame([input_data])

# Align with training features
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# Predict
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f" High risk of stroke detected! (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ No stroke predicted. (Confidence: {1 - probability:.2f})")
