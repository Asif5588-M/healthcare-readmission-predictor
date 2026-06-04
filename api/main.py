from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import predict_readmission, load_artifacts

app = FastAPI(
    title="Diabetic Patient Readmission Predictor",
    description="""
    Predicts early hospital readmission (<30 days) 
    for diabetic patients using ML.
    
    Built by Muhammad Asif | Healthcare Data Scientist
    Published Research: ML for Sustainable Healthcare (HEC Y-Category)
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class PatientData(BaseModel):
    race                    : int = Field(..., ge=0, le=5)
    gender                  : int = Field(..., ge=0, le=1)
    age_numeric             : int = Field(..., ge=0, le=9)
    admission_type_id       : int = Field(..., ge=1, le=8)
    discharge_disposition_id: int = Field(..., ge=1, le=30)
    admission_source_id     : int = Field(..., ge=1, le=25)
    time_in_hospital        : int = Field(..., ge=1, le=14)
    num_lab_procedures      : int = Field(..., ge=1, le=132)
    num_procedures          : int = Field(..., ge=0, le=6)
    num_medications         : int = Field(..., ge=1, le=81)
    number_outpatient       : int = Field(..., ge=0, le=42)
    number_emergency        : int = Field(..., ge=0, le=76)
    number_inpatient        : int = Field(..., ge=0, le=21)
    number_diagnoses        : int = Field(..., ge=1, le=16)
    change                  : int = Field(..., ge=0, le=1)
    diabetesMed             : int = Field(..., ge=0, le=1)
    num_drugs_changed       : int = Field(..., ge=0, le=10)
    num_drugs_active        : int = Field(..., ge=0, le=14)

    class Config:
        json_schema_extra = {"example": {
            "race": 0, "gender": 0,
            "age_numeric": 6,
            "admission_type_id": 1,
            "discharge_disposition_id": 1,
            "admission_source_id": 7,
            "time_in_hospital": 5,
            "num_lab_procedures": 45,
            "num_procedures": 2,
            "num_medications": 18,
            "number_outpatient": 0,
            "number_emergency": 1,
            "number_inpatient": 1,
            "number_diagnoses": 8,
            "change": 1,
            "diabetesMed": 1,
            "num_drugs_changed": 2,
            "num_drugs_active": 6
        }}

class PredictionResponse(BaseModel):
    prediction      : int
    readmission_prob: float
    risk_level      : str
    recommendation  : str
    threshold_used  : float
    model           : str


@app.get("/")
def root():
    return {
        "title"  : "Diabetic Patient Readmission API",
        "version": "1.0.0",
        "author" : "Muhammad Asif — Healthcare Data Scientist",
        "docs"   : "/docs"
    }

@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "healthy", "models": "loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    _, _, metadata = load_artifacts()
    return {
        "model_name"  : metadata['model_name'],
        "auc_roc"     : metadata['auc'],
        "threshold"   : metadata['threshold'],
        "total_features": len(metadata['features']),
        "features"    : metadata['features']
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    try:
        result = predict_readmission(patient.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))