import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("stroke_predictor_model.pkl")

st.set_page_config(page_title="Stroke Predictor", layout="centered")
st.title("🧠 Stroke Risk Prediction")
st.markdown("Enter patient details below to check the likelihood of stroke.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=1, max_value=120)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Encoding inputs
def encode_inputs():
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    married_map = {"Yes": 1, "No": 0}
    work_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 4, "Never_worked": 1}
    residence_map = {"Urban": 1, "Rural": 0}
    smoke_map = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}

    return pd.DataFrame([{
        'gender': gender_map[gender],
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'ever_married': married_map[ever_married],
        'work_type': work_map[work_type],
        'Residence_type': residence_map[residence_type],
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'smoking_status': smoke_map[smoking_status]
    }])

# Prediction
if st.button("Predict Stroke Risk"):
    input_df = encode_inputs()
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"High risk of stroke! (Probability: {probability:.2f})")

    else:
        st.success(f"✅ No stroke risk detected (Probability: {probability:.2f})")
