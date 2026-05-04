# -----------------------------------------------------------------------------
# File:    src/features.py
# Project: Caltech CTME Healthcare Capstone
# Purpose: Feature engineering — creates age bin categories 
# -----------------------------------------------------------------------------

# Third party — installed packages
import pandas as pd

# Local — project modules
from data_loader import load_and_clean
from utils import logger
from config import (
    RAW_DATA_FILE,
    TARGET_COL,
    FEATURE_COLS,
    AGE_BINS,
    AGE_LABELS
)


def encode_features(df):
    """
    Encodes categorical features for modeling.
    Creates ordinal age bin categories from the continuous Age column.

    :param df: cleaned dataframe (pd.DataFrame)
    :return: dataframe with encoded features added (pd.DataFrame)
    """

    # Create age bins and encode
    df['age_cat'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype('category')
    # .cat.codes converts ordered categories to integers — preserves order
    # 21-30 → 0, 31-40 → 1, 41-50 → 2, 51-60 → 3, 61+ → 4
    df['age_cat_code'] = df['age_cat'].cat.codes
    
    logger.info('Features encoded successfully')
    
    return df


def build_features(filepath):
    """
    Loads cleaned data and encodes categorical features for modeling

    :param filepath: path to raw data file (Path)
    :return: feature matrix X (pd.DataFrame), target vector y (pd.Series)
    """

    df = load_and_clean(filepath)
    df = encode_features(df)
    X = df[FEATURE_COLS] # noqa N806
    y = df[TARGET_COL]
    
    logger.info('Features built successfully')
    
    return X, y

