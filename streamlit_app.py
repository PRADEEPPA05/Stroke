import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model and feature list
model = joblib.load("stroke_predictor_model_new.pkl")
model_features = joblib.load("model_features_new.pkl")

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

    # Predict probability
    stroke_prob = model.predict_proba(input_df)[0][1]

    # Use threshold 0.25 to determine stroke risk
    threshold = 0.25
    stroke_pred = 1 if stroke_prob >= threshold else 0

    st.subheader(f"🔢 Stroke Probability: {round(stroke_prob * 100, 2)}%")

    if stroke_pred == 1:
        st.error("🔴 High Risk of Stroke.")
    else:
        st.success("✅ Low Risk of Stroke Detected.")

    # SHAP explainability
    st.subheader("🔍 Feature Contribution (SHAP)")

    try:
        explainer = shap.Explainer(model.predict_proba, input_df)
        shap_values = explainer(input_df)

        # Plot SHAP bar chart for class 1 (stroke)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ SHAP explainability not available.\n\n{e}")
