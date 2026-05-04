# -----------------------------------------------------------------------------
# File:    src/eda_raw.py
# Project: Caltech CTME Healthcare Capstone
# Purpose: Exploratory data analysis on the raw data.
# -----------------------------------------------------------------------------

# Third party — installed packages
import matplotlib.pyplot as plt
import seaborn as sns

# Local — project modules
from data_loader import load_raw
from utils import add_finding, COLUMN_LABELS, CATEGORICAL_COLS, logger
from config import (
    RAW_DATA_FILE,
    EDA_RAW_HISTOGRAMS,
    EDA_RAW_DTYPE_CHART,
    EDA_RAW_CLASS_BALANCE,
    ZERO_AS_MISSING_COLS
)

import warnings
warnings.filterwarnings('ignore', message='Using categorical units to plot a list of strings')

def run_eda_raw(filepath, findings):
    """
    Pre-cleaning data exploration of the raw data.
    Findings are appended to the shared findings list for export in main.py.

    :param filepath: path of raw data file
    :param findings: shared findings list
    """
    df = load_raw(filepath)
    # Count biologically impossible zero values of the measures saved in ZERO_AS_MISSING_COLS
    # axis=0 sums down the rows to give a zero count per column
    zero_counts = (df[ZERO_AS_MISSING_COLS] == 0).sum(axis=0)

    for col, count in zero_counts.items():
        add_finding(findings, 'EDA - Raw', 'Zero Count', f'{col}: {count} zeros found')

    # Data type distribution
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        add_finding(findings, 'EDA - Raw', 'Data Type Count', f'{dtype}: {count} columns')

    # Explore missing values
    # Get count of nulls at the column level
    # Returns a Series with each column name and its total null count across all rows.
    null_counts = df.isnull().sum()
    add_finding(findings, 'EDA - Raw', 'Null Count', f'Total nulls in raw data: {null_counts.sum()}')

    # Class balance check — how many diabetic vs non-diabetic patients (65/35 split)
    outcome_counts = df['Outcome'].value_counts()
    total = outcome_counts.sum()

    for outcome, count in outcome_counts.items():
        pct = round((count / total) * 100, 1)
        label = 'Diabetic' if outcome == 1 else 'Non-Diabetic'
        add_finding(findings, 'EDA - Raw', 'Class Balance', f'{label}: {count} patients ({pct}%)')

    # Class imbalance detected — stratified sampling required for modeling
    add_finding(findings, 'EDA - Raw', 'Class Imbalance',
                'Dataset is imbalanced — 65% non-diabetic, 35% diabetic. Stratified sampling required for modeling.')

    # Histograms of all columns in raw data
    logger.info('Plotting histograms')
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    # flatten axes array into a 1D list.
    axes = axes.flatten()
    # loop over all columns in the dataframe and plot a histogram for each
    for i, col in enumerate(df.columns):
        if col in CATEGORICAL_COLS:
            temp_df = df[[col]].copy()
            temp_df[col] = temp_df[col].astype(str)
            sns.countplot(x=col, data=temp_df, ax=axes[i])
        else:
            sns.histplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(COLUMN_LABELS.get(col, col))

    fig.suptitle('Distributions of Raw Measures and Outcome Variables', fontsize=16)
    plt.tight_layout()
    plt.savefig(EDA_RAW_HISTOGRAMS, dpi=150, bbox_inches='tight')
    plt.close()

    # Data type distribution bar chart
    logger.info('Plotting dtype distribution chart')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    # reset_index() promotes the Series index (data types) to a column,
    # converting the Series to a DataFrame that seaborn can plot
    dtype_df = dtype_counts.reset_index()
    dtype_df.columns = ['Data Type', 'Count']
    dtype_df['Data Type'] = dtype_df['Data Type'].astype(str)
    sns.barplot(data=dtype_df, x='Data Type', y='Count')
    ax.set_title('Data Type Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_RAW_DTYPE_CHART, dpi=150, bbox_inches='tight')
    plt.close()

    # Class balance count plot for the outcome variable
    logger.info('Plotting class balance')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
    outcome_counts.index = ['Non-Diabetic', 'Diabetic']
    outcome_df = outcome_counts.reset_index()
    outcome_df.columns = ['Outcome', 'Count']
    sns.barplot(data=outcome_df, x='Outcome', y='Count')
    ax.set_title('Outcome Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_RAW_CLASS_BALANCE, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("EDA Raw complete")