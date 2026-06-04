# config.py
from pathlib import Path

# Directories
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / 'data'
RAW_DIR       = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR    = ROOT_DIR / 'models'
FIGURES_DIR   = ROOT_DIR / 'outputs' / 'figures'
REPORTS_DIR   = ROOT_DIR / 'outputs' / 'reports'

# Create if not exist
for d in [PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data
RAW_DATA_PATH = RAW_DIR / 'diabetic_data.csv'

# Model
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
VAL_SIZE      = 0.10
TARGET_COL    = 'readmitted'

# Features
CATEGORICAL_COLS = [
    'race', 'gender', 'age', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id',
    'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
    'max_glu_serum', 'A1Cresult', 'metformin',
    'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide',
    'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed'
]

NUMERICAL_COLS = [
    'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

# API
API_HOST = "0.0.0.0"
API_PORT = 8000