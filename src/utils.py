import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR


def load_artifacts():
    model    = joblib.load(MODELS_DIR / 'readmission_model.pkl')
    scaler   = joblib.load(MODELS_DIR / 'scaler.pkl')
    metadata = joblib.load(MODELS_DIR / 'model_metadata.pkl')
    return model, scaler, metadata


def get_risk_level(prob: float) -> tuple:
    if prob >= 0.70:
        return "CRITICAL", "Immediate discharge planning + follow-up within 7 days"
    elif prob >= 0.50:
        return "HIGH", "Enhanced discharge planning + follow-up within 14 days"
    elif prob >= 0.30:
        return "MODERATE", "Standard discharge + follow-up within 30 days"
    else:
        return "LOW", "Routine discharge planning"


def predict_readmission(data: dict) -> dict:
    model, scaler, metadata = load_artifacts()

    threshold    = float(metadata['threshold'])
    feature_cols = metadata['features']

    # Keep only required features in correct order
    df = pd.DataFrame([data])

    # Add missing columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    X         = scaler.transform(df)
    prob      = float(model.predict_proba(X)[0][1])
    prediction = int(prob >= threshold)

    risk_level, recommendation = get_risk_level(prob)

    return {
        'prediction'      : prediction,
        'readmission_prob': round(prob, 4),
        'risk_level'      : risk_level,
        'recommendation'  : recommendation,
        'threshold_used'  : round(threshold, 2),
        'model'           : str(metadata['model_name'])
    }