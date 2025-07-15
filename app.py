import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and features
model = joblib.load("stroke_predictor_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.set_page_config(page_title="🧠 Stroke Prediction App")
st.title("🧠 Stroke Prediction App")
st.markdown("Fill in the details to predict stroke risk.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=60.0)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# Convert inputs to dataframe
input_dict = {
    "gender": 1 if gender == "Male" else 0,
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "ever_married": 1 if ever_married == "Yes" else 0,
    "Residence_type": 1 if residence_type == "Urban" else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": 0 if smoking_status == "never smoked" else 1 if smoking_status == "formerly smoked" else 2,
    "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
    "work_type_Private": 1 if work_type == "Private" else 0,
    "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
    "work_type_children": 1 if work_type == "children" else 0,
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_names, fill_value=0)

if st.button("Predict Stroke Risk"):
    prob = model.predict_proba(input_df)[0][1]
    probability = round(prob * 100, 2)

    st.markdown(f"### 🔢 Stroke Probability: {probability}%")

    if probability >= 50:
        st.error("🔴 High Risk of Stroke Detected!")
    elif probability >= 25:
        st.warning("🟠 Moderate Risk of Stroke.")
    else:
        st.success("✅ Low Risk of Stroke Detected.")

    # SHAP explanation
    st.subheader("🔍 Feature Contribution (SHAP)")
    explainer = shap.TreeExplainer(model)
   # Compute SHAP values
    shap_values = explainer.shap_values(input_df)
    
    # Get the correct SHAP value array
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        shap_vals = shap_values[0]
    
    # Ensure it's a 1D array
    shap_vals = np.array(shap_vals).flatten()
    
    # Plot
    fig, ax = plt.subplots()
    shap.bar_plot(shap_vals, feature_names=feature_names, max_display=10)
    st.pyplot(fig)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load model and model features
# model = joblib.load("stroke_predictor_model.pkl")
# model_features = joblib.load("model_features.pkl")

# st.set_page_config(page_title="Stroke Prediction App", layout="centered")

# st.title("🧠 Stroke Prediction App")
# st.write("Fill in the details to predict stroke risk.")

# # User input fields
# gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# age = st.number_input("Age", min_value=1.0, max_value=120.0, value=30.0)

# hypertension = st.selectbox("Hypertension", ["No", "Yes"])
# heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
# ever_married = st.selectbox("Ever Married", ["No", "Yes"])
# work_type = st.selectbox("Work Type", ["Private", "Self-employed", "children", "Never_worked"])
# residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
# avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
# bmi = st.number_input("BMI", min_value=0.0, value=28.0)
# smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# # Map inputs to numerical
# data = {
#     "gender": 1 if gender == "Male" else (2 if gender == "Other" else 0),
#     "age": age,
#     "hypertension": 1 if hypertension == "Yes" else 0,
#     "heart_disease": 1 if heart_disease == "Yes" else 0,
#     "ever_married": 1 if ever_married == "Yes" else 0,
#     "Residence_type": 1 if residence_type == "Urban" else 0,
#     "avg_glucose_level": avg_glucose_level,
#     "bmi": bmi,
#     "smoking_status": 0 if smoking_status == "never smoked" else
#                       1 if smoking_status == "formerly smoked" else
#                       2 if smoking_status == "smokes" else 3
# }

# # One-hot encode work_type
# work_type_encoded = {
#     'work_type_Private': 0,
#     'work_type_Self-employed': 0,
#     'work_type_children': 0,
#     'work_type_Never_worked': 0
# }
# selected_key = f"work_type_{work_type}"
# if selected_key in work_type_encoded:
#     work_type_encoded[selected_key] = 1

# # Combine all inputs
# input_data = {**data, **work_type_encoded}
# input_df = pd.DataFrame([input_data])

# # Ensure all model features are present
# for feature in model_features:
#     if feature not in input_df.columns:
#         input_df[feature] = 0
# input_df = input_df[model_features]

# # Prediction
# if st.button("🔍 Predict Stroke Risk"):
#     prediction = model.predict(input_df)[0]
#     probability = model.predict_proba(input_df)[0][1] * 100

#     st.write(f" **Stroke Probability: {probability:.2f}%**")

#     if probability >= 50:
#         st.error("🔴 High Risk of Stroke Detected!")
#     elif probability >= 25:
#         st.warning("🟠 Moderate Risk of Stroke.")
#     else:
#         st.success("✅ Low Risk of Stroke Detected.")
    
