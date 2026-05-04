# -----------------------------------------------------------------------------
# File:    src/utils.py
# Project: Caltech CTME Healthcare Capstone
# Purpose: Shared utility functions used across multiple modules.
# -----------------------------------------------------------------------------
import logging

def add_finding(findings, category, finding, value):
    """
    Appends a finding to the findings list.
    :param findings: list of findings dictionaries
    :param category: category of the finding e.g. 'EDA', 'Factor Analysis', 'Model'
    :param finding: short label describing the finding
    :param value: full insight or value
    """
    findings.append({'category': category, 'finding': finding, 'value': value})


# Dictionary mapping column names to descriptive axis labels with units
COLUMN_LABELS = {
    'Pregnancies': 'Pregnancies (count)',
    'Glucose': 'Plasma Glucose Concentration (mg/dL)',
    'BloodPressure': 'Diastolic Blood Pressure (mm Hg)',
    'SkinThickness': 'Triceps Skin Fold Thickness (mm)',
    'Insulin': '2-Hour Serum Insulin (mu U/ml)',
    'BMI': 'Body Mass Index (kg/m²)',
    'DiabetesPedigreeFunction': 'Diabetes Pedigree Function (score)',
    'Age': 'Age (years)',
    'Outcome': 'Outcome (0 = Non-Diabetic, 1 = Diabetic)'
}

CATEGORICAL_COLS = ['Outcome', 'Pregnancies']

# Descriptive labels for clinical age bins used in diabetes risk analysis
AGE_BIN_LABELS = {
    '21-30': 'Young Adults',
    '31-40': 'Early Middle Age',
    '41-50': 'Middle Age',
    '51-60': 'Older Adults',
    '61+':   'Senior'
}
# Unit labels for measures used in age bin analysis
MEASURE_UNITS = {
    'Glucose': 'Glucose (mg/dL)',
    'BMI': 'BMI (kg/m²)',
    'Insulin': 'Insulin (mu U/ml)'
}

# -----------------------------------------------------------------------------
# Logging configuration — used across all modules for pipeline status messages
# Outputs to both console and pipeline.log file in the project root
# Format: timestamp [module_name] message
# Usage: from utils import logger
#        logger.info('Your message here')
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(module)s] %(message)s',
    handlers=[
        logging.StreamHandler(),           # console output
        logging.FileHandler('pipeline.log') # saves to file
    ]
)
# Suppress matplotlib and PIL informational messages from appearing in pipeline log
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Dictionary mapping for model comparison bar chart
EVAL_MEASURE_LABELS = {
    'sensitivity': 'Sensitivity (Recall) — Catches True Diabetics',
    'specificity': 'Specificity — Clears True Non-Diabetics',
    'precision': 'Precision — Accuracy of Diabetic Predictions',
    'f1': 'F1 Score — Balance of Precision & Recall',
    'auc_roc': 'AUC-ROC — Overall Separation Power'
}