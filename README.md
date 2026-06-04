# 🏥 Healthcare Readmission Predictor

**30-Day Hospital Readmission Risk Prediction System**  
*Diabetic Patients ke liye Early Readmission (<30 Days) Prediction*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-orange)

---

## 🎯 Project Goal

Hospitals mein **readmission rates** kam karne ke liye Machine Learning based predictive system. High-risk patients ko identify karke clinical intervention possible banata hai.

---

## 📊 Dataset
- **UCI Diabetic Hospital Readmission Dataset** (101,766 records)
- Binary Target: Early Readmission (`<30` days)

---

## 🛠️ Tech Stack
- **Modeling**: LightGBM (Optuna tuned)
- **Explainability**: SHAP
- **Deployment**: Streamlit Dashboard
- **Hyperparameter Tuning**: Optuna
- **Environment**: Anaconda (ds-env)

---

## 📈 Model Performance

- **Best Model**: LightGBM
- **CV AUC**: 0.6776
- **Test Recall @0.15 threshold**: **0.9335** (High sensitivity for high-risk patients)
- **Clinical Threshold**: 0.15

---

## 🚀 Features
- Advanced Feature Engineering
- Class imbalance handling (`scale_pos_weight`)
- Hyperparameter tuning with Optuna
- **Interactive Streamlit Dashboard** with SHAP explanations
- Modular code structure

---

## 🧪 How to Run

```bash
cd D:\Projects\healthcare-readmission-predictor

# Activate environment
conda activate D:\Projects\ds-env

# Install dependencies
pip install -r requirements.txt

# Run Dashboard
streamlit run app/streamlit_app.py
📁 Project Structure
healthcare-readmission-predictor/
├── data/                  # raw + processed
├── notebooks/             # 01 to 04
├── app/                   # Streamlit Dashboard
├── models/                # best model + scaler
├── src/ & config.py
├── outputs/
└── README.md