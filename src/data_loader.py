# -----------------------------------------------------------------------------
# File:    src/data_loader.py
# Project: Caltech CTME Healthcare Capstone
# Purpose: Load raw diabetes dataset, treat biologically impossible zero values
#          as missing data, impute using median by Outcome group, and save the
#          cleaned dataset to data/processed/
# -----------------------------------------------------------------------------
# Third party — installed packages
import pandas as pd
import numpy as np

# Local — project modules
from utils import logger
from config import (
    RAW_DATA_FILE, 
    CLEAN_DATA_FILE, 
    ZERO_AS_MISSING_COLS, 
    TARGET_COL
)

def load_raw(filepath):
    """
    Loads raw CSV file without any cleaning for use in raw data exploration.
    Results are used to justify cleaning decisions made in load_and_clean().

    :param filepath: path to the CSV file to load
    :return: raw uncleaned dataframe (pd.DataFrame)
    """

    # Load raw data as-is
    df = pd.read_csv(filepath)
    logger.info(f'Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    return df


def load_and_clean(filepath):
    """
    Loads and cleans a CSV file — removes duplicates, drops unusable columns,
    excludes rows missing the target variable, and sets UID as the index.

    :param filepath: path to the CSV file to load
    :return: cleaned dataframe (pd.DataFrame)
    """

    # Load data
    df = load_raw(filepath)
    
    # Replace all zeros with NaN
    df[ZERO_AS_MISSING_COLS] = df[ZERO_AS_MISSING_COLS].replace(0, np.nan)
    
    # Fill NANs with measure's median by outcome group
    for measure in ZERO_AS_MISSING_COLS:
        df[measure] = df[measure].fillna(
            df.groupby(TARGET_COL)[measure].transform("median")
        )

    # Drop rows where the target column is missing
    df = df.dropna(subset=[TARGET_COL])

    # Save cleaned data to processed folder
    df.to_csv(CLEAN_DATA_FILE, index=False)
    logger.info(f'Cleaned data saved to: {CLEAN_DATA_FILE.name}')

    # Return cleaned dataframe
    return df