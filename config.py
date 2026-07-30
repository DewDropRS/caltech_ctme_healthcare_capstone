from pathlib import Path

# pathlib.Path is the modern Python (3.4+) approach to file paths.
# It is platform-agnostic, using the correct slash direction
# for Windows (\) vs Mac/Linux (/). This is safer than hardcoding paths.

# Path(__file__) = the full path to this config.py file
# .resolve()     = converts to absolute path (no relative ../ references)
# .parent        = steps one level up to the project root directory
BASE_DIR = Path(__file__).resolve().parent

# --- Directory Paths ---
# The / operator here is a joiner operation
# BASE_DIR / "data" / "raw" is equivalent to os.path.join(BASE_DIR, "data", "raw")
DATA_RAW_DIR        = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR  = BASE_DIR / "data" / "processed"
OUTPUTS_FIGURES_DIR = BASE_DIR / "outputs" / "figures"
OUTPUTS_REPORTS_DIR = BASE_DIR / "outputs" / "reports"

# --- EDA Raw outputs ---
EDA_RAW_HISTOGRAMS      = OUTPUTS_FIGURES_DIR / 'eda_raw_histograms.png'
EDA_RAW_DTYPE_CHART     = OUTPUTS_FIGURES_DIR / 'eda_raw_dtype_distribution.png'
EDA_RAW_CLASS_BALANCE   = OUTPUTS_FIGURES_DIR / 'eda_raw_class_balance.png'

# --- EDA Clean outputs ---
EDA_CLEAN_HISTOGRAMS    = OUTPUTS_FIGURES_DIR / 'eda_clean_histograms.png'
EDA_CLEAN_PAIRPLOT      = OUTPUTS_FIGURES_DIR / 'eda_clean_pairplot.png'
EDA_CLEAN_HEATMAP       = OUTPUTS_FIGURES_DIR / 'eda_clean_heatmap.png'
EDA_CLEAN_AGE_BINS      = OUTPUTS_FIGURES_DIR / 'eda_clean_age_bins.png'

# --- EDA Pairplot ---
# Maps numeric Outcome to descriptive labels
OUTCOME_LABELS = {0: 'Non-Diabetic', 1: 'Diabetic'}

# --- Consolidated Findings Report ---
FINDINGS_REPORT     = OUTPUTS_REPORTS_DIR / 'project_findings.csv'
CORRELATION_MATRIX  = OUTPUTS_REPORTS_DIR / 'eda_clean_correlation_matrix.csv'

# --- Input File ---
RAW_DATA_FILE = DATA_RAW_DIR / "health_care_diabetes.csv"

# --- Processed File ---
# This is the cleaned dataset saved after missing value treatment in data_loader.py
CLEAN_DATA_FILE = DATA_PROCESSED_DIR / "diabetes_clean.csv"

# --- Target Variable ---
TARGET_COL = "Outcome"

# --- Columns with biologically impossible zero values (treated as missing) ---
# Defined here in config so any module (data_loader, eda_raw, etc.) can
# reference the same list without duplicating or hardcoding it elsewhere.
# If we ever add or remove a column, we fix it in one place only.
ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# --- Random state for reproducibility ---
# Setting a fixed random state ensures that train/test splits and model
# results are identical every time the pipeline is run. 42 is conventional
# but any integer works — what matters is consistency.
RANDOM_STATE = 42

# --- Age Bin Settings ---
AGE_BINS   = [21, 31, 41, 51, 61, 120]
AGE_LABELS = ['21-30', '31-40', '41-50', '51-60', '61+']

# --- Model Feature Columns ---
# Age and age_cat excluded — Age is replaced by age_cat_code, age_cat is text
FEATURE_COLS = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'age_cat_code'
]

# --- Model outputs ---
MODEL_ROC_CURVE         = OUTPUTS_FIGURES_DIR / 'model_roc_curve.png'
MODEL_CONFUSION_MATRIX  = OUTPUTS_FIGURES_DIR / 'model_confusion_matrix.png'
MODEL_RESULTS           = OUTPUTS_REPORTS_DIR / 'model_results.csv'
MODEL_RESULTS_HBAR       = OUTPUTS_FIGURES_DIR / 'model_results_hbar.png'
MODEL_RF_FEATURE_IMPORTANCE_HBAR = OUTPUTS_FIGURES_DIR / 'model_rf_feature_importance_hbar.png'

# --- Saved Models ---
MODEL_KNN       = OUTPUTS_REPORTS_DIR / 'model_knn.joblib'
MODEL_LR        = OUTPUTS_REPORTS_DIR / 'model_logistic.joblib'
MODEL_RF        = OUTPUTS_REPORTS_DIR / 'model_random_forest.joblib'
MODEL_SCALER    = OUTPUTS_REPORTS_DIR / 'model_scaler.joblib'