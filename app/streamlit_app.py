import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# =============================================
# Path Setup for Streamlit Cloud
# =============================================
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from config import *

st.set_page_config(
    page_title="Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Diabetic Hospital Readmission Predictor")
st.markdown("**30-Day Early Readmission Risk Assessment (<30 Days)**")

# =============================================
# Load Model & Features
# =============================================
@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODELS_DIR / "best_readmission_model.pkl")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    return model, X_train.columns.tolist()

model, feature_columns = load_model_and_features()

# =============================================
# Sidebar - Patient Input
# =============================================
st.sidebar.header("🧑‍⚕️ Patient Clinical Information")

age_numeric = st.sidebar.slider("Age Group", 0, 9, 5)
time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 132, 45)
num_medications = st.sidebar.slider("Number of Medications", 1, 81, 15)
number_inpatient = st.sidebar.slider("Previous Inpatient Visits", 0, 21, 1)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 8)
discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition ID", [1, 2, 3, 6, 7, 20, 28])

# =============================================
# Prediction
# =============================================
if st.sidebar.button("🚀 Predict Readmission Risk", type="primary"):
    with st.spinner("Calculating Risk..."):
        # Prepare input data
        input_dict = {
            'age_numeric': [age_numeric],
            'time_in_hospital': [time_in_hospital],
            'num_lab_procedures': [num_lab_procedures],
            'num_medications': [num_medications],
            'number_inpatient': [number_inpatient],
            'number_diagnoses': [number_diagnoses],
            'discharge_disposition_id': [discharge_disposition_id],
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Add missing features
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[feature_columns]
        
        # Make Prediction
        prob = model.predict_proba(input_df)[0][1]
        
        # Display Results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("**Readmission Probability**", f"{prob:.2%}")
        
        with col2:
            if prob >= 0.15:
                st.error("🔴 **HIGH RISK** - Early Readmission Likely")
            else:
                st.success("🟢 **Low Risk** - Unlikely to be Readmitted Early")
        
        # SHAP Explanation
        st.subheader("🔍 Feature Contribution (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        fig = plt.figure(figsize=(12, 7))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0]
        ))
        st.pyplot(fig)

st.caption("Healthcare Data Science Portfolio Project | Built with LightGBM + SHAP")