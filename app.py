import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Load model and feature list
model = joblib.load("stroke_predictor_model.pkl")
model_features = joblib.load("model_features.pkl")
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="🧠 Stroke Prediction App")
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
    # Encode inputs
    input_dict = {
        "gender": 1 if gender == "Male" else (2 if gender == "Other" else 0),
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": 1 if ever_married == "Yes" else 0,
        "Residence_type": 1 if residence_type == "Urban" else 0,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}[smoking_status]
    }

    # One-hot encode work_type
    work_type_cols = ['work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children']
    for col in work_type_cols:
        input_dict[col] = 1 if col.split("_")[1] == work_type else 0

    input_df = pd.DataFrame([input_dict])
    
    # Reorder columns to match model
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns
    input_df = input_df[model_features]

    # Predict
    stroke_prob = model.predict_proba(input_df)[0][1]
    stroke_percent = round(stroke_prob * 100, 1)

    st.subheader(f"🔢 Stroke Probability: {stroke_percent}%")
    
    # Display risk level
    if stroke_percent >= 70:
        st.error("🔴 High Risk of Stroke.")
    elif stroke_percent >= 30:
        st.warning("🟠 Moderate Risk of Stroke.")
    else:
        st.success("✅ Low Risk of Stroke Detected.")

    # # SHAP Explanation
    # st.subheader("🔍 Feature Contribution (SHAP)")
    # try:
    #     shap_input = input_df.astype(float)  # Ensure numerical
    #     shap_values = explainer.shap_values(shap_input)
    #     #st.set_option('deprecation.showPyplotGlobalUse', False)
    #     shap.initjs()
    #     st.pyplot(shap.plots.bar(shap_values[1][0], show=False))
    # except Exception as e:
    #     st.warning(f"⚠️ SHAP explainability not available for this input.\n\n{e}")
    # 🔍 SHAP Explainability
    st.subheader("🔍 Feature Contribution (SHAP)")
    try:
        # Create explainer
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(input_df)
    
        # Plot SHAP values
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.plots.bar(shap_values[0], max_display=10))
    except Exception as e:
        st.warning(f"⚠️ SHAP explainability not available for this input.\n\n{e}")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import shap
# import matplotlib.pyplot as plt

# # Page config
# st.set_page_config(page_title="🧠 Stroke Prediction App", layout="centered")

# st.title("🧠 Stroke Prediction App")
# st.markdown("Fill in the details to predict stroke risk.")

# # Load model and feature list
# model = joblib.load("stroke_predictor_model.pkl")
# model_features = joblib.load("model_features.pkl")

# # User inputs
# gender = st.selectbox("Gender", ["Male", "Female", "Other"])
# age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
# hypertension = st.selectbox("Hypertension", ["No", "Yes"])
# heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
# ever_married = st.selectbox("Ever Married", ["No", "Yes"])
# work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
# residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
# avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=1.0)
# bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=28.0, step=0.1)
# smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# # On submit
# if st.button("🔍 Predict Stroke Risk"):
#     input_data = {
#         "gender": [1 if gender == "Male" else (0 if gender == "Female" else 2)],
#         "age": [age],
#         "hypertension": [1 if hypertension == "Yes" else 0],
#         "heart_disease": [1 if heart_disease == "Yes" else 0],
#         "ever_married": [1 if ever_married == "Yes" else 0],
#         "Residence_type": [1 if residence_type == "Urban" else 0],
#         "avg_glucose_level": [avg_glucose_level],
#         "bmi": [bmi],
#         "smoking_status": [0 if smoking_status == "never smoked" else (1 if smoking_status == "formerly smoked" else (2 if smoking_status == "smokes" else 3))],
#         "work_type_Never_worked": [1 if work_type == "Never_worked" else 0],
#         "work_type_Private": [1 if work_type == "Private" else 0],
#         "work_type_Self-employed": [1 if work_type == "Self-employed" else 0],
#         "work_type_children": [1 if work_type == "children" else 0],
#     }

#     input_df = pd.DataFrame(input_data)

#     # Reorder columns to match training
#     input_df = input_df.reindex(columns=model_features, fill_value=0)

#     # Predict
#     probability = model.predict_proba(input_df)[0][1]
#     percent = round(probability * 100, 2)

#     st.subheader(f"🔢 Stroke Probability: {percent}%")

#     # Risk category
#     if percent >= 60:
#         st.error("🔴 High Risk of Stroke.")
#     elif percent >= 20:
#         st.warning("🟠 Moderate Risk of Stroke.")
#     else:
#         st.success("✅ Low Risk of Stroke Detected.")

#     # SHAP explainability
#     try:
#         st.subheader("🔍 Feature Contribution (SHAP)")
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(input_df)

#         # Classification: use class 1 SHAP values
#         if isinstance(shap_values, list):
#             shap_vals = shap_values[1][0]
#         else:
#             shap_vals = shap_values[0]

#         # Plot
#         plt.figure()
#         shap.bar_plot(shap_vals, feature_names=model_features, max_display=10)
#         st.pyplot(plt)
#     except Exception as e:
#         st.warning("⚠️ SHAP explainability not available for this input.")
#         st.text(str(e))




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
    
