# -----------------------------------------------------------------------------
# File:    main.py
# Project: Caltech CTME Healthcare Capstone
# Purpose: Main pipeline entry point for the Diabetes Classification project.
#          Orchestrates the full pipeline in the following order:
#
#          1. EDA — Raw Data    : Documents zero-as-missing data quality issues
#          2. EDA — Clean Data  : Explores cleaned data, pairplots, correlation
#                                 heatmap, and age bin analysis
#          3. Feature Engineering: Ordinal age bin encoding and feature matrix
#                                 construction
#          4. Modeling          : Trains and evaluates three classification models:
#                                 - K-Nearest Neighbors (KNN)
#                                 - Logistic Regression
#                                 - Random Forest
#          5. Findings          : Saves consolidated findings report to outputs/
#
#          All outputs saved to outputs/figures/ and outputs/reports/
#          Trained models saved to outputs/reports/ for future predictions
# -----------------------------------------------------------------------------
# Standard library — Python built-ins
import sys
import os

# Add src/ to path so modules can find each other
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Local — project modules
from utils import logger
from eda_raw import run_eda_raw
from eda_clean import run_eda_clean
from features import build_features
from model import build_models
from config import RAW_DATA_FILE, FINDINGS_REPORT


def main():
    """
    Executes the full diabetes classification pipeline end-to-end.
    """
    import pandas as pd
    findings = []

    # EDA
    run_eda_raw(RAW_DATA_FILE, findings)
    run_eda_clean(RAW_DATA_FILE, findings)

    # Feature Engineering
    X, y = build_features(RAW_DATA_FILE)

    # Modeling
    build_models(X, y, findings)

    # Save consolidated findings list
    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(FINDINGS_REPORT, index=False)
    logger.info(f'All findings saved to {FINDINGS_REPORT.name}')
    logger.info('Pipeline complete')

if __name__ == "__main__":
    main()