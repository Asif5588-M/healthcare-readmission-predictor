import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import os

st.set_page_config(
    page_title="Diabetic Readmission Risk",
    page_icon="🏥",
    layout="wide"
)

API_URL = "http://localhost:8000"

# ── Profile Header ────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
col_img, col_info = st.columns([1, 8])

with col_img:
    profile_path = os.path.join(APP_DIR, "profile.png")
    if os.path.exists(profile_path):
        st.image(profile_path, width=80)

with col_info:
    st.markdown("### Asif Nawaz")
    st.markdown(
        "🏥 Healthcare Data Scientist &nbsp;|&nbsp; "
        "MPhil Economics &nbsp;|&nbsp; "
        "Published Researcher &nbsp;|&nbsp; "
        "13+ Years Clinical Experience"
    )

st.divider()

# ── Header ────────────────────────────────────────────────────
st.title("🏥 Diabetic Patient Readmission Risk Predictor")
st.markdown(
    "**ML-powered early readmission prediction for diabetic patients**  \n"
    "Model: LightGBM | AUC: 0.68 | "
    "Dataset: 101,763 real hospital encounters"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🧑‍⚕️ Patient Information")

age_map = {
    '[0-10)' :0,'[10-20)':1,'[20-30)':2,'[30-40)':3,
    '[40-50)':4,'[50-60)':5,'[60-70)':6,'[70-80)':7,
    '[80-90)':8,'[90-100)':9
}
age_label   = st.sidebar.selectbox(
    "Age Group", list(age_map.keys()), index=6)
age_numeric = age_map[age_label]

gender     = st.sidebar.selectbox("Gender", ["Female","Male"])
gender_val = 0 if gender == "Female" else 1

race_map = {
    "Caucasian":0, "AfricanAmerican":1,
    "Hispanic" :2, "Asian":3,
    "Other"    :4, "Unknown":5
}
race_label = st.sidebar.selectbox("Race", list(race_map.keys()))
race_val   = race_map[race_label]

st.sidebar.markdown("---")
st.sidebar.subheader("🏨 Admission Details")

time_in_hospital = st.sidebar.slider(
    "Days in Hospital", 1, 14, 4)

admission_type_id = st.sidebar.selectbox(
    "Admission Type",
    options=[1,2,3,4,5,6,7,8],
    format_func=lambda x: {
        1:"Emergency",2:"Urgent",3:"Elective",
        4:"Newborn",5:"Not Available",6:"NULL",
        7:"Trauma Center",8:"Not Mapped"
    }[x])

discharge_disposition_id = st.sidebar.selectbox(
    "Discharge Disposition", list(range(1,27)), index=0)

admission_source_id = st.sidebar.selectbox(
    "Admission Source", list(range(1,26)), index=6)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Clinical Indicators")

num_lab_procedures = st.sidebar.slider(
    "Lab Procedures", 1, 132, 43)
num_procedures     = st.sidebar.slider(
    "Procedures", 0, 6, 1)
num_medications    = st.sidebar.slider(
    "Medications", 1, 81, 16)
number_diagnoses   = st.sidebar.slider(
    "Number of Diagnoses", 1, 16, 7)
number_inpatient   = st.sidebar.slider(
    "Prior Inpatient Visits", 0, 21, 0)
number_emergency   = st.sidebar.slider(
    "Prior Emergency Visits", 0, 76, 0)
number_outpatient  = st.sidebar.slider(
    "Prior Outpatient Visits", 0, 42, 0)
num_drugs_active   = st.sidebar.slider(
    "Active Diabetes Drugs", 0, 14, 4)
num_drugs_changed  = st.sidebar.slider(
    "Drugs with Dose Change", 0, 10, 1)

change = st.sidebar.selectbox(
    "Medication Change", [0, 1],
    format_func=lambda x: "Yes" if x else "No")
diabetesMed = st.sidebar.selectbox(
    "On Diabetes Medication", [1, 0],
    format_func=lambda x: "Yes" if x else "No")

# ── Top Metrics ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Days Hospitalized", time_in_hospital,
          delta="High" if time_in_hospital > 7 else "Normal")
c2.metric("Lab Procedures",    num_lab_procedures)
c3.metric("Medications",       num_medications,
          delta="High" if num_medications > 20 else "Normal")
c4.metric("Prior Emergencies", number_emergency,
          delta="Risk" if number_emergency > 0 else "None")

st.divider()

# ── Predict Button ────────────────────────────────────────────
if st.button("🔍 Predict Readmission Risk",
             type="primary", use_container_width=True):

    payload = {
        "race"                    : race_val,
        "gender"                  : gender_val,
        "age_numeric"             : age_numeric,
        "admission_type_id"       : admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id"     : admission_source_id,
        "time_in_hospital"        : time_in_hospital,
        "num_lab_procedures"      : num_lab_procedures,
        "num_procedures"          : num_procedures,
        "num_medications"         : num_medications,
        "number_outpatient"       : number_outpatient,
        "number_emergency"        : number_emergency,
        "number_inpatient"        : number_inpatient,
        "number_diagnoses"        : number_diagnoses,
        "change"                  : change,
        "diabetesMed"             : diabetesMed,
        "num_drugs_changed"       : num_drugs_changed,
        "num_drugs_active"        : num_drugs_active
    }

    with st.spinner("Analyzing patient data..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload, timeout=10)
            result = response.json()

            risk_level = result.get('risk_level', 'LOW')
            prob       = result.get('readmission_prob', 0)
            rec        = result.get('recommendation', '')
            threshold  = result.get('threshold_used', 0)
            model_name = result.get('model', '')

            risk_icons = {
                "CRITICAL": "🔴",
                "HIGH"    : "🟠",
                "MODERATE": "🟡",
                "LOW"     : "🟢"
            }
            icon = risk_icons.get(risk_level, "⚪")

            # ── Risk Banner ───────────────────────────────
            if risk_level in ["CRITICAL", "HIGH"]:
                st.error(
                    f"{icon} {risk_level} RISK — "
                    f"Early Readmission Likely")
            elif risk_level == "MODERATE":
                st.warning(
                    f"{icon} MODERATE RISK — "
                    f"Monitor Closely")
            else:
                st.success(
                    f"{icon} LOW RISK — "
                    f"Early Readmission Unlikely")

            # ── Results ───────────────────────────────────
            r1, r2 = st.columns(2)

            with r1:
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number+delta",
                    value = round(prob * 100, 1),
                    title = {'text': "Readmission Probability (%)"},
                    delta = {'reference': 11.2,
                             'increasing': {'color': '#E74C3C'},
                             'decreasing': {'color': '#2ECC71'}},
                    gauge = {
                        'axis' : {'range': [0, 100]},
                        'bar'  : {'color': '#E74C3C'},
                        'steps': [
                            {'range': [0,  30], 'color': '#2ECC71'},
                            {'range': [30, 50], 'color': '#F39C12'},
                            {'range': [50, 70], 'color': '#E67E22'},
                            {'range': [70,100], 'color': '#E74C3C'}
                        ],
                        'threshold': {
                            'line' : {'color': 'black', 'width': 3},
                            'value': round(threshold * 100, 0)
                        }
                    }
                ))
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

            with r2:
                st.subheader("📋 Clinical Summary")
                st.write(f"**Risk Level:** {icon} {risk_level}")
                st.write(
                    f"**Readmission Probability:** "
                    f"{prob * 100:.1f}%")
                st.write(f"**Population Baseline:** 11.2%")
                st.write(
                    f"**Decision Threshold:** "
                    f"{threshold * 100:.0f}%")
                st.write(f"**Model:** {model_name}")
                st.info(f"📌 **Recommendation:**  \n{rec}")

                # Risk Factor Table
                st.markdown("**Risk Factor Summary:**")
                risk_factors = {
                    "Days Hospitalized": (
                        time_in_hospital,
                        "⚠️ High" if time_in_hospital > 7
                        else "✅ Normal"),
                    "Prior Emergencies": (
                        number_emergency,
                        "⚠️ Risk" if number_emergency > 0
                        else "✅ None"),
                    "Prior Inpatient"  : (
                        number_inpatient,
                        "⚠️ Risk" if number_inpatient > 0
                        else "✅ None"),
                    "Medications"      : (
                        num_medications,
                        "⚠️ High" if num_medications > 20
                        else "✅ Normal"),
                    "Drug Changes"     : (
                        num_drugs_changed,
                        "⚠️ Active" if num_drugs_changed > 0
                        else "✅ None"),
                }
                rf_df = pd.DataFrame(
                    [(k, v[0], v[1])
                     for k, v in risk_factors.items()],
                    columns=["Factor", "Value", "Status"])
                st.dataframe(
                    rf_df, hide_index=True,
                    use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ API offline. Start it with:  \n"
                "`uvicorn api.main:app --reload --port 8000`")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ── About Section ─────────────────────────────────────────────
st.divider()
st.subheader("📊 About This Model")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Data",  "101,763 patients")
col2.metric("AUC-ROC",        "0.68")
col3.metric("Recall @0.15",   "93.35%")
col4.metric("Features Used",  "35")

with st.expander("📖 Research Background & Methodology"):
    st.markdown("""
    **Clinical Context**
    Early hospital readmission (<30 days) in diabetic 
    patients is a key quality indicator. CMS penalizes 
    hospitals for excess readmissions under HRRP.

    **Dataset**
    Diabetes 130-US Hospitals (1999–2008) — 101,763 real
    patient encounters from 130 US hospitals.
    Source: UCI ML Repository.

    **Methodology**
    - LightGBM with Optuna hyperparameter tuning
    - Class imbalance handled with scale_pos_weight
    - Clinical threshold 0.15 for high sensitivity
    - SHAP explainability

    **Published Research**
    *"ML for Sustainable Healthcare: Identifying 
    High-Cost Utilizers and Diagnostic Waste in Pakistan"*
    — HEC Y-Category Journal
    """)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "👨‍💻 Asif Nawaz | MPhil Economics | "
    "Arid Agriculture University Rawalpindi | "
    "📄 Published: HEC Y-Category Journal | "
    "🏥 13+ Years Healthcare Experience"
)