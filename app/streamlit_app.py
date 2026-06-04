import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Fixed Import
sys.path.append(str(Path(__file__).parent.parent))
from config import *

st.set_page_config(page_title="Readmission Predictor", layout="wide")

st.title("🏥 Diabetic Hospital Readmission Predictor")
st.markdown("**30-Day Early Readmission (<30 Days) Risk Assessment**")

# ====================== LOAD EVERYTHING ======================
@st.cache_resource
def load_all():
    model = joblib.load(MODELS_DIR / "best_readmission_model.pkl")
    # Load training columns
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    return model, X_train.columns.tolist()

model, feature_columns = load_all()

# Sidebar
st.sidebar.header("🧑‍⚕️ Patient Information")

age_numeric = st.sidebar.slider("Age Group (0-9)", 0, 9, 5)
time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 132, 45)
num_medications = st.sidebar.slider("Medications", 1, 81, 15)
number_inpatient = st.sidebar.slider("Previous Inpatient Visits", 0, 21, 1)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 8)
discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition ID", [1, 2, 3, 6, 7, 20, 28])

if st.sidebar.button("🚀 Predict Risk", type="primary"):
    # Prepare Input
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
    
    # Add all missing columns with 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns exactly as training
    input_df = input_df[feature_columns]
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("**Readmission Probability**", f"{prob:.2%}")
    
    with col2:
        if prob >= 0.15:
            st.error("🔴 **HIGH RISK** - Early Readmission Likely")
        else:
            st.success("🟢 **Low Risk**")

    # SHAP Explanation
    st.subheader("🔍 Feature Contribution (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0]
    ))
    st.pyplot(fig)

st.caption("Healthcare Data Science Portfolio Project | Asif")