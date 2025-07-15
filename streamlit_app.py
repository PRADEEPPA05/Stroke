
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

import zipfile
import joblib

with zipfile.ZipFile('stroke_predictor_model.zip', 'r') as zip_ref:
    zip_ref.extractall()  # Extracts stroke_predictor_model.pkl in current folder

model = joblib.load('stroke_predictor_model.pkl')

# Load the trained model and feature list

model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="🧠 Stroke Prediction App", layout="centered")
st.title("🧠 Stroke Prediction App")
st.subheader("Fill in the details to predict stroke risk.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

if st.button("🔍 Predict Stroke Risk"):

    # Prepare input dictionary matching training features
    input_dict = {
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Male': 1 if gender == "Male" else 0,
        'gender_Other': 1 if gender == "Other" else 0,
        'ever_married_Yes': 1 if ever_married == "Yes" else 0,
        'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
        'work_type_Private': 1 if work_type == "Private" else 0,
        'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
        'work_type_children': 1 if work_type == "children" else 0,
        'Residence_type_Urban': 1 if residence_type == "Urban" else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
        'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
        'smoking_status_smokes': 1 if smoking_status == "smokes" else 0,
    }

    # Make sure all model features are present in the input
    for feature in model_features:
        if feature not in input_dict:
            input_dict[feature] = 0

    # Create DataFrame in correct feature order
    input_df = pd.DataFrame([input_dict])[model_features]

    st.write("📊 Input DataFrame Preview:")
    st.dataframe(input_df)
    
    st.write("🧠 Model Type:", type(model))


    # Predict probability
    stroke_prob = model.predict_proba(input_df)[0][1]

    # Show probability
    st.subheader(f"🔢 Stroke Probability: {round(stroke_prob * 100, 2)}%")

    # Threshold logic
    if stroke_prob >= 0.4:
        st.error("🔴 High Risk of Stroke.")
    elif stroke_prob >= 0.25:
        st.warning("🟠 Moderate Risk of Stroke.")
    else:
        st.success("✅ Low Risk of Stroke Detected.")

# SHAP explainability using LightGBM base model
    st.subheader("🔍 Feature Contribution (SHAP)")
    
    try:
        import lightgbm as lgb
        # Extract LightGBM model from VotingClassifier
        lgb_model = model.named_estimators_["lgbm"]
    
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(input_df)
    
        # Plot SHAP summary for one instance
        fig, ax = plt.subplots()
        shap.plots.bar(shap.Explanation(values=shap_values[1], base_values=explainer.expected_value[1], data=input_df), show=False)
        st.pyplot(fig)
    
    except Exception as e:
        st.warning(f"⚠️ SHAP explainability not available.\n\n{e}")



