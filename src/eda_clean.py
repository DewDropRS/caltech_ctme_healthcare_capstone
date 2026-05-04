# -----------------------------------------------------------------------------
# File:    src/eda_clean.py
# Project: Caltech CTME Healthcare Capstone
# Purpose: Exploratory data analysis on the cleaned data
# -----------------------------------------------------------------------------

# Third party — installed packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local — project modules
from data_loader import load_and_clean
from utils import (
    add_finding,
    COLUMN_LABELS,
    CATEGORICAL_COLS,
    AGE_BIN_LABELS,
    MEASURE_UNITS,
    logger
)
from config import (
    RAW_DATA_FILE,
    ZERO_AS_MISSING_COLS,
    TARGET_COL,
    EDA_CLEAN_HISTOGRAMS,
    EDA_CLEAN_PAIRPLOT,
    EDA_CLEAN_HEATMAP,
    EDA_CLEAN_AGE_BINS,
    CORRELATION_MATRIX,
    AGE_BINS,
    AGE_LABELS
)

import warnings
warnings.filterwarnings('ignore', module='matplotlib')


def run_eda_clean(filepath, findings):
    """
    Exploratory analysis of the cleaned data.

    :param filepath: path of cleaned data file
    :param findings: shared findings list

    """

    # load and clean patient data
    df = load_and_clean(filepath)

    # Histograms of all columns in cleaned data
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

    fig.suptitle('Distributions of Measures and Outcome Variables', fontsize=16)
    plt.tight_layout()
    plt.savefig(EDA_CLEAN_HISTOGRAMS, dpi=150, bbox_inches='tight')
    plt.close()

    # Scatter pairplots — relationships between feature pairs, colored by Outcome
    # Rename columns to shorter display labels for pairplot readability
    logger.info('Scatter pairplots')
    df_plot = df.rename(columns={
        'BloodPressure': 'BP (mm Hg)',
        'SkinThickness': 'Skin (mm)',
        'DiabetesPedigreeFunction': 'DPF Score',
        'BMI': 'BMI (kg/m²)',
        'Glucose': 'Glucose (mg/dL)',
        'Insulin': 'Insulin (mu U/ml)',
        'Age': 'Age(years)'
    })
    # seaborn default kernel is gaussian
    sns.pairplot(df_plot, hue='Outcome', diag_kind='kde', kind='reg')
    plt.suptitle('Feature Pairplots by Diabetes Outcome (KDE Diagonal)', fontsize=16)
    plt.tight_layout()
    plt.savefig(EDA_CLEAN_PAIRPLOT, dpi=150, bbox_inches='tight')
    plt.close()

    # Record findings
    add_finding(findings, 'EDA - Clean', 'Pairplot Observation',
                'Glucose KDE diagonal shows the strongest separation between diabetic and non-diabetic patients — \
                 diabetic peak shifted right toward higher glucose values')

    add_finding(findings, 'EDA - Clean', 'Pairplot Observation',
                'BMI KDE diagonal shows modest separation — diabetic patients peak slightly higher but distributions \
                overlap considerably')

    add_finding(findings, 'EDA - Clean', 'Pairplot Observation',
                'Age distribution in the KDE diagonal is more spread out for diabetic patients — older patients more \
                likely to be diabetic')

    # Correlation heatmap
    correlation_df = df.corr(numeric_only=True)
    correlation_df.to_csv(CORRELATION_MATRIX)
    logger.info(f'Correlation Matrix saved to {CORRELATION_MATRIX.name}')

    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    sns.heatmap(data = correlation_df,
                ax = ax,
                annot= True,
                cmap ='coolwarm',
                fmt='.2f',
                annot_kws={'size': 7},
                vmin = -1,
                vmax = 1,
                center = 0)
    ax.set_title('Correlation Heatmap of Key Features', fontsize = 16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize = 10)
    plt.figtext(0.5, -0.02,
                'Correlation ranges: 0.0-0.4 weak | 0.4-0.7 moderate | 0.7-1.0 strong | \
                Negative values mirror the same scale',
                ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    plt.savefig(EDA_CLEAN_HEATMAP
                , dpi=150
                , bbox_inches='tight'
                )
    plt.close()
    add_finding(findings, 'EDA - Clean', 'Correlation with Outcome',
                'Glucose has the strongest correlation with Outcome at 0.50 — primary predictor of diabetes')

    add_finding(findings, 'EDA - Clean', 'Correlation with Outcome',
                'BMI, Insulin, and SkinThickness show moderate correlation with Outcome ranging from 0.30 to 0.38')

    add_finding(findings, 'EDA - Clean', 'Feature Correlation',
                'BMI and SkinThickness show the strongest inter-feature correlation at 0.57 — makes clinical sense \
                as both measure body composition')

    add_finding(findings, 'EDA - Clean', 'Feature Correlation',
                'Pregnancies and Age correlated at 0.54 — older patients naturally have had more pregnancies')

    add_finding(findings, 'EDA - Clean', 'Feature Correlation',
                'Glucose and Insulin correlated at 0.49 — confirms expected physiological relationship')

    # Age bins
    # right=False means each bin includes the left boundary and excludes the right
    # e.g. [20-25) includes age 20 but not 25, which falls into the next bin
    df['age_bin'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    # Average clinical measures by age bin and outcome
    age_group = df.groupby(['age_bin', 'Outcome'])[['Glucose', 'BMI', 'Insulin']].mean().reset_index()
    age_df = age_group.melt(
        id_vars=['age_bin','Outcome'],
        value_vars=['Glucose', 'BMI', 'Insulin'],
        var_name='Measure',
        value_name='Mean')
    # Map Outcome values to descriptive labels for legend readability
    age_df['Outcome'] = age_df['Outcome'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
    # cast to categorical type
    age_df['age_bin'] = pd.Categorical(age_df['age_bin'], categories=AGE_LABELS, ordered=True)
    measures = ['Glucose', 'BMI', 'Insulin']
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))
    for i, measure in enumerate(measures):
        measure_df = age_df[age_df['Measure'] == measure]
        sns.barplot(data=measure_df, x='age_bin', y='Mean', hue='Outcome', ax=axes[i])
        axes[i].set_title(f'Average {measure} by Age Category')
        axes[i].set_xlabel('Age Category')
        axes[i].set_ylabel(MEASURE_UNITS.get(measure, measure))

    plt.suptitle('Clinical Measures by Age Category and Diabetes Outcome', fontsize=16)
    plt.tight_layout()
    plt.savefig(EDA_CLEAN_AGE_BINS
                , dpi=150
                , bbox_inches='tight'
                )
    plt.close()

    # Add finding
    add_finding(findings, 'EDA - Clean', 'Age Bin Analysis',
                'Diabetic patients show higher mean Glucose, BMI, and Insulin levels across all age categories — consistent\
                 with clinical expectations for diabetes risk factors')

    logger.info('EDA Clean complete')
