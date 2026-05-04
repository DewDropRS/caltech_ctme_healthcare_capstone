# Caltech CTME Data Science Capstone — Healthcare Industry
## Diabetes Prediction Using Classification Models

---

## Overview

This project is part of the Caltech CTME Data Science Bootcamp capstone series. Using a clinical dataset originally 
sourced from the National Institute of Diabetes and Digestive and Kidney Diseases, the goal is to build and evaluate 
classification models that predict whether a patient has diabetes based on diagnostic measurements.

All patients in this dataset are females at least 21 years old of Pima Indian heritage.

---

## Problem Statement

Build a model to accurately predict whether a patient has diabetes based on clinical diagnostic features, and evaluate 
model performance using classification metrics including sensitivity, specificity, and AUC-ROC.

---

## Dataset Description

**Source:** National Institute of Diabetes and Digestive and Kidney Diseases  
**File:** `health_care_diabetes.csv`  
**Rows:** 768  
**Columns:** 9 (8 features + 1 target)

| Column | Type | Description |
|---|---|---|
| Pregnancies | int | Number of times pregnant |
| Glucose | int | Plasma glucose concentration at 2 hours in an oral glucose tolerance test |
| BloodPressure | int | Diastolic blood pressure (mm Hg) |
| SkinThickness | int | Triceps skin fold thickness (mm) |
| Insulin | int | 2-hour serum insulin (mu U/ml) |
| BMI | float | Body mass index (weight in kg / height in m²) |
| DiabetesPedigreeFunction | float | Diabetes pedigree function (genetic risk score) |
| Age | int | Age in years |
| Outcome | int | Target variable — 1 = diabetic, 0 = non-diabetic |

**Note:** Five columns — Glucose, BloodPressure, SkinThickness, Insulin, and BMI — contain zero values that are 
biologically impossible and treated as missing data. See the Methodology section for how these were handled.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations and array handling |
| scikit-learn | Machine learning models and evaluation metrics |
| matplotlib | Static visualizations |
| seaborn | Statistical visualizations |
| pathlib | Modern Python path handling (platform-agnostic) |
| Jupyter / PyCharm | Development environment |

---

## Configuration

All file paths, column constants, and model settings are centralized in `config.py` at the project root. No paths or
magic values are hardcoded inside individual modules — every script imports what it needs from config.

**Key concepts used:**

`pathlib.Path` — the modern Python (3.4+) approach to file paths. Unlike the older `os.path` style, `pathlib` treats 
paths as objects and uses the `/` operator to join segments, making code cleaner and platform-agnostic.

```python
# os.path style (older, still valid)
os.path.join(BASE_DIR, "data", "raw")

# pathlib style (modern)
BASE_DIR / "data" / "raw"
```

`RANDOM_STATE = 42` — a fixed integer passed to models and train/test splits to ensure results are identical every 
time the pipeline runs. Any integer works; what matters is consistency.

---

## Data Loading & Cleaning

**Module:** `src/data_loader.py`

The raw dataset contains no actual null values — instead, five columns use `0` 
as a placeholder for missing data. This is a common pattern in older clinical 
datasets. The following columns were treated this way:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

**Missing value treatment — median by Outcome group:**

Rather than using a single global median to impute missing values, we calculated 
the median separately for diabetic patients (Outcome=1) and non-diabetic patients 
(Outcome=0). This preserves the natural difference in clinical measurements 
between the two groups.

```python
# Replace impossible zeros with NaN
df[ZERO_AS_MISSING_COLS] = df[ZERO_AS_MISSING_COLS].replace(0, np.nan)

# Impute using median by Outcome group
for measure in ZERO_AS_MISSING_COLS:
    df[measure] = df[measure].fillna(
        df.groupby(TARGET_COL)[measure].transform("median")
    )
```

**Key concept — `.groupby()` + `.transform()`:**

`.transform()` differs from `.agg()` in that it returns a result with the same 
shape as the original dataframe — meaning each row gets the median value of its 
own group broadcast back to it. This makes it compatible with `.fillna()` for 
row-by-row imputation.

**DRY (Don't repeat yourself) principle in action:** The list of zero-as-missing columns is defined once 
in `config.py` as `ZERO_AS_MISSING_COLS` and imported wherever needed — 
`data_loader.py`, `eda_raw.py` — so there is no risk of inconsistency across modules.

---

## Exploratory Data Analysis — Raw Data

**Module:** `src/eda_raw.py`

Before any cleaning, the raw data was explored to document data quality issues 
and justify the cleaning decisions made in `data_loader.py`.

**Key findings:**

- Five columns — Glucose, BloodPressure, SkinThickness, Insulin, and BMI — 
  contained zero values that are biologically impossible, confirming they were 
  used as placeholders for missing data
- No actual null values exist in the raw dataset — zeros were the only form 
  of missing data
- Dataset is imbalanced — 65% non-diabetic (500 patients), 35% diabetic 
  (268 patients)
- 7 integer columns, 2 float columns

**Class imbalance and why it matters:**

When a dataset is imbalanced, a naive model could achieve 65% accuracy by 
simply predicting every patient as non-diabetic — without learning anything 
meaningful. This is why accuracy alone is not a sufficient metric for this 
problem. We use sensitivity, specificity, and AUC-ROC instead, and apply 
stratified sampling to ensure every train/test split maintains the same 
65/35 class ratio as the full dataset.

**New Python patterns used:**

`(df[cols] == 0).sum(axis=0)` — detects impossible zero values by creating 
a True/False dataframe and summing down the rows (`axis=0`) to get a zero 
count per column. Python treats `True` as 1 and `False` as 0, so the sum 
gives a count.

`plt.subplots(nrows, ncols)` + `axes.flatten()` — creates a grid of subplots 
and flattens the 2D axes array into a 1D list so it can be iterated over with 
a single index in a for loop.

`enumerate(df.columns)` — iterates over column names with a numeric index, 
giving both the position (`i`) for accessing the correct subplot and the column 
name (`col`) for plotting.

**PEP 8 — Python import ordering convention:**

Imports are ordered by convention following PEP 8 — Python's official style guide:
1. Standard library (`sys`, `os`)
2. Third party packages (`pandas`, `matplotlib`, `seaborn`)
3. Local project modules (`config`, `data_loader`, `utils`)

---

## Exploratory Data Analysis — Cleaned Data

**Module:** `src/eda_clean.py`

After missing value imputation, the cleaned dataset was explored to confirm 
the quality of the treatment and to identify relationships between features 
and the target variable.

**Key findings:**

- Glucose is the strongest predictor of diabetes with a correlation of 0.50 
  with Outcome
- BMI, Insulin, and SkinThickness show moderate correlation with Outcome 
  ranging from 0.30 to 0.38
- BMI and SkinThickness are the most correlated feature pair at 0.57 — 
  both measure body composition
- Diabetic patients show higher mean Glucose, BMI, and Insulin across all 
  age categories — consistent with clinical expectations
- Risk of diabetes increases with age — diabetic patients have a broader, 
  more spread out age distribution extending into older age groups

**How to read a KDE diagonal in a pairplot:**

The diagonal of a pairplot shows the distribution of each variable for each 
Outcome group. When reading these plots:
- Focus on the **x-axis only** — it shows the actual values of that variable
- The y-axis shows density (height of the curve) and is not meaningful for 
  comparison between groups
- Where each curve **peaks along the x-axis** tells you the most common value 
  for that group
- If the orange (diabetic) curve peaks further **right** than the blue 
  (non-diabetic) curve, diabetic patients tend to have higher values of 
  that variable

**Age bin analysis:**

Patients were grouped into clinically meaningful age categories rather than 
arbitrary equal intervals, reflecting how metabolic disease risk is typically 
stratified in clinical research and health plan analytics:

| Age Range | Clinical Description |
|---|---|
| 21-30 | Young Adults |
| 31-40 | Early Middle Age |
| 41-50 | Middle Age |
| 51-60 | Older Adults |
| 61+ | Senior |

**New Python patterns used:**

`df.corr(numeric_only=True)` — computes pairwise correlation coefficients 
between all numeric columns. Values range from -1.0 (perfect negative 
correlation) to 1.0 (perfect positive correlation). The diagonal is always 
1.0 since a variable is perfectly correlated with itself.

`pd.cut()` — bins continuous values into discrete intervals. `right=False` 
means each bin includes the left boundary and excludes the right, preventing 
any value from falling into two bins simultaneously.

`df.melt()` — reshapes a wide dataframe into a long format. `id_vars` are 
the anchor columns that stay as-is, `value_vars` are the columns to unpivot 
into rows, `var_name` names the new column holding variable names, and 
`value_name` names the column holding the actual values. Long format is 
required by seaborn for grouped plots.

`sns.pairplot()` with `hue='Outcome'`, `diag_kind='kde'`, `kind='reg'` — 
produces a matrix of scatter plots between every feature pair, colored by 
Outcome group, with KDE distributions on the diagonal and regression lines 
in the scatter plots.

---

## Feature Engineering

**Module:** `src/features.py`

Feature engineering transforms raw data into a form that is more useful for 
modeling. In this project the primary transformation is converting the 
continuous `Age` column into an ordinal categorical feature that reflects 
clinically meaningful age groupings for metabolic disease risk.

**Age bins — why clinically meaningful intervals instead of equal splits:**

Rather than using arbitrary equal 5-year intervals, age bins were defined to 
reflect how metabolic disease risk is actually stratified in clinical research 
and health plan analytics:

| Age Range | Clinical Description | Bin Code |
|---|---|---|
| 21-30 | Young Adults | 0 |
| 31-40 | Early Middle Age | 1 |
| 41-50 | Middle Age | 2 |
| 51-60 | Older Adults | 3 |
| 61+ | Senior | 4 |

**Encoding approaches — ordinal vs one-hot:**

**Ordinal Encoding** — converts ordered categories to integers, preserving 
the natural order. Used here for age bins because the ordering is meaningful — 
a patient aged 51-60 is genuinely older than 31-40 and that progression carries 
real clinical information for diabetes risk prediction. One-hot encoding would 
lose this ordering entirely.

```python
# Ordinal encoding — order is preserved
# 21-30 → 0, 31-40 → 1, 41-50 → 2, 51-60 → 3, 61+ → 4
df['age_cat_code'] = df['age_cat'].cat.codes
```

**One-Hot Encoding** — creates a separate binary column for each category. 
Used for truly unordered categories where no natural ordering exists and 
where assuming a numeric relationship between categories would mislead the 
model. For example, encoding a patient's blood type:

```python
# Hypothetical example — blood type has no natural order
# A, B, AB, O should not be encoded as 0, 1, 2, 3
# because that implies AB > B > A which is meaningless
df_encoded = pd.get_dummies(df['BloodType'], prefix='BloodType')

# Creates binary columns:
# BloodType_A | BloodType_B | BloodType_AB | BloodType_O
#      1      |      0      |      0       |      0
```

**Feature matrix `X` and target vector `y`:**

By convention in machine learning `X` is the feature matrix and `y` is 
the target variable

| In X | Excluded |
|---|---|
| Pregnancies | Age (replaced by age_cat_code) |
| Glucose | age_cat (text label) |
| BloodPressure | Outcome (this is y) |
| SkinThickness | |
| Insulin | |
| BMI | |
| DiabetesPedigreeFunction | |
| age_cat_code | |


---

---

## Modeling

**Module:** `src/model.py`

Three classification models were trained and evaluated using a consistent
pipeline: stratified train/test split, feature scaling, GridSearchCV
hyperparameter tuning, and a shared evaluation framework.

---

### Train/Test Split

The dataset was split 80/20 into training (614 rows) and test (154 rows) sets
using stratified sampling to preserve the 65/35 class ratio in both sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
```

`stratify=y` ensures each split maintains the same 65/35 diabetic/non-diabetic
ratio as the full dataset. Without stratification, random splits could
underrepresent diabetic patients in the test set, leading to unreliable
evaluation.

---

### Feature Scaling and Data Leakage

The training data and test data must be treated as completely separate.
Fitting the scaler on the full dataset before splitting causes data leakage —
the scaler learns the mean and standard deviation of the test set and
carries that information into the training process, giving the model an
unfair advantage on data it should never have seen.

The correct approach:
1. Split data into train and test sets first
2. Fit the scaler on training data only — `scaler.fit_transform(X_train)`
3. Apply the same scaler to test data — `scaler.transform(X_test)`

The fitted scaler is also saved for reuse — when the model is deployed
to make predictions on new patient data, the exact same scaling
transformation used during training must be applied to ensure consistent
and fair predictions.

**Why KNN requires scaling:**
KNN calculates the distance between data points to find the nearest neighbors.
Without scaling, features with larger numeric ranges (e.g. Glucose: 50-200)
dominate the distance calculation over features with smaller ranges
(e.g. BMI: 18-60), regardless of their actual predictive importance.
StandardScaler transforms every feature to have mean=0 and std=1,
ensuring equal contribution to distance calculations.

**Note:** Random Forest does not require scaling since it makes decisions
based on thresholds, not distances. However scaling is applied to all
models for consistency.

---

### Saving Models with joblib

Trained models and scalers exist only in memory — when the script finishes
they disappear. `joblib` serializes them to disk for reuse.

`joblib` is preferred over Python's built-in `pickle` for sklearn objects
because it is optimized for objects containing large numpy arrays,
which is exactly what trained sklearn models and scalers are.

```python
# Save
joblib.dump(scaler, 'scaler.joblib')

# Load and reuse on new data
scaler = joblib.load('scaler.joblib')
```

The scaler must be saved alongside the model — when predicting on new patient
data, the exact same scaling transformation used during training must be
applied first to ensure consistent and reliable predictions.

---

### GridSearchCV and Cross-Validation

All three models use `GridSearchCV` with 5-fold cross-validation to find
the optimal hyperparameters automatically.

**How it works:**
1. Define a `param_grid` of hyperparameter values to test
2. GridSearchCV tests every combination using 5-fold cross-validation
3. Each combination is evaluated 5 times on different data splits
4. The combination with the best average AUC-ROC score across all folds wins
5. The best model is returned via `grid_search.best_estimator_`

**Why 5-fold cross-validation:**
A single train/validation split gives one score that could be lucky or unlucky.
Cross-validation runs the model 5 times on different splits and averages the
results — a much more reliable estimate of true model performance.
With only 768 rows, cv=5 balances reliability with computation time.

```
Fold 1: [--TEST--][--train--][--train--][--train--][--train--]
Fold 2: [--train--][--TEST--][--train--][--train--][--train--]
Fold 3: [--train--][--train--][--TEST--][--train--][--train--]
Fold 4: [--train--][--train--][--train--][--TEST--][--train--]
Fold 5: [--train--][--train--][--train--][--train--][--TEST--]
```

`scoring='roc_auc'` optimizes for AUC-ROC rather than accuracy, which is
more appropriate for imbalanced datasets.

---

### Model 1 — K-Nearest Neighbors (KNN)

KNN classifies a new patient by finding the K most similar patients in the
training data (nearest neighbors) and taking a majority vote among them.
It requires feature scaling because similarity is measured by distance.

**Hyperparameter grid:**

| Parameter | Values Tested | Justification |
|---|---|---|
| `n_neighbors` | 3, 5, 7, 9, 11, 13, 15 | Odd values prevent ties; range covers simple to smooth boundaries |
| `weights` | uniform, distance | Uniform = all neighbors vote equally; distance = closer neighbors get more weight |
| `metric` | euclidean, manhattan | Euclidean = straight-line distance; Manhattan = grid-like distance |

**Best parameters:** `{'metric': 'manhattan', 'n_neighbors': 15, 'weights': 'uniform'}`

---

### Model 2 — Logistic Regression

Logistic Regression models the probability of diabetes as a function of
the input features. It uses regularization to prevent overfitting by
penalizing large feature weights.

**Note on sklearn 1.8+ parameter changes:**
The `penalty` parameter was deprecated in sklearn 1.8. The updated approach
uses `l1_ratio` with `solver='saga'`:

- `l1_ratio=0` — equivalent to L2 (Ridge): shrinks all weights toward zero,
  keeps all features
- `l1_ratio=1` — equivalent to L1 (Lasso): can zero out weak features entirely,
  acts as automatic feature selection
- `l1_ratio=0.5` — ElasticNet: blend of both L1 and L2

**Hyperparameter grid:**

| Parameter | Values Tested | Justification |
|---|---|---|
| `C` | 0.01, 0.1, 1, 10, 100 | Inverse of regularization strength; small C = strong regularization |
| `l1_ratio` | 0, 0.5, 1 | Tests L2, ElasticNet blend, and L1 regularization |

**Regularization rule of thumb:**
- Small C (0.01) — strong regularization, simpler model, less risk of overfitting
- Large C (100) — weak regularization, more complex model, higher risk of overfitting

---

### Model 3 — Random Forest

Random Forest builds an ensemble of decision trees, each trained on a
random subset of rows (bootstrap sampling) and features. Final predictions
are determined by majority vote across all trees.

**Why it outperforms a single decision tree:**
A single tree memorizes training data and overfits. By averaging many trees
that each see different subsets, errors cancel out and the ensemble is
more robust and generalizable.

**Feature importance — Gini impurity:**
At each split, Random Forest measures how much that split reduced impurity
(how well it separated diabetics from non-diabetics). Features that
consistently produce clean splits across all trees receive higher importance
scores. This is measured using Gini impurity.

**Hyperparameter grid:**

| Parameter | Values Tested | Justification |
|---|---|---|
| `n_estimators` | 100, 200, 300 | More trees = more stable; beyond 300 rarely improves performance |
| `max_depth` | 10, 20 | Limits tree complexity to prevent overfitting |
| `min_samples_split` | 5, 10 | Higher values prevent splits on very small groups |
| `min_samples_leaf` | 2, 4 | Minimum samples at leaf nodes; smooths the model |
| `bootstrap` | True | Each tree trained on random sample with replacement — classic Random Forest |

`n_jobs=-1` was used to parallelize GridSearchCV across all available CPU cores.

---

### Model Results

| Model | Sensitivity | Specificity | Precision | F1 Score | AUC-ROC |
|---|---|---|---|---|---|
| KNN | 77.8% | 92.0% | 84.0% | 80.8% | 90.9% |
| Logistic Regression | 51.9% | 87.0% | 68.3% | 58.9% | 82.6% |
| Random Forest | 79.6% | 89.0% | 79.6% | 79.6% | 93.9% |

---

### Model Recommendation

**Random Forest is the recommended model** based on:
- Highest AUC-ROC (93.9%) — best overall separation between diabetic and
  non-diabetic patients
- Highest Sensitivity (79.6%) — best at catching true diabetics, the
  priority metric in a clinical setting
- Strong F1 Score (79.6%) — well balanced precision and recall

KNN is a strong alternative with higher Specificity (92.0%) and Precision
(84.0%) — better at clearing healthy patients and more accurate when it
predicts diabetes. The choice between KNN and Random Forest depends on
whether the clinical priority is catching more diabetics (Random Forest)
or avoiding false alarms (KNN).

Logistic Regression significantly underperforms on Sensitivity (51.9%) —
missing nearly half of all diabetic patients — making it unsuitable for
clinical deployment in this context.

---

### Note on Insulin Feature Importance

Random Forest assigned Insulin the highest feature importance at 42%.
This likely reflects imputation bias — 49% of Insulin values were missing
in the raw dataset and replaced with group medians, potentially creating
an artificially clean split pattern. Glucose, which had the strongest
correlation with Outcome in EDA (0.50), ranked second at 19%.

Despite this, Random Forest achieved strong overall performance (AUC = 93.9%).
A recommended next step is to revisit the Insulin imputation strategy —
either dropping the column, using KNN imputation, or creating a binary
missing indicator flag.

---

### Making Predictions on New Patient Data

To use the saved Random Forest model for predictions on new patients:

```python
import joblib
import pandas as pd
from config import MODEL_RF, MODEL_SCALER, FEATURE_COLS

# Load saved model and scaler
model = joblib.load(MODEL_RF)
scaler = joblib.load(MODEL_SCALER)

# New patient data — must match the same feature columns used in training
new_patient = pd.DataFrame([{
    'Pregnancies': 2,
    'Glucose': 138,
    'BloodPressure': 72,
    'SkinThickness': 28,
    'Insulin': 140,
    'BMI': 33.5,
    'DiabetesPedigreeFunction': 0.45,
    'age_cat_code': 2   # 41-50 age bin
}])

# Apply the same scaling used during training
new_patient_scaled = scaler.transform(new_patient[FEATURE_COLS])

# Predict
prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)[:, 1]

print(f'Prediction: {"Diabetic" if prediction[0] == 1 else "Non-Diabetic"}')
print(f'Probability of Diabetes: {probability[0]:.1%}')
```

**Important:** Always apply `scaler.transform()` (not `fit_transform()`) to
new data — use the same scaler fitted during training to ensure consistent
and reliable predictions.

---

## Classification Metrics Reference

### Confusion Matrix

A confusion matrix categorizes every prediction into one of four outcomes:

```
                     Predicted Non-Diabetic    Predicted Diabetic
Actual Non-Diabetic       TN (C[0,0])              FP (C[0,1])
Actual Diabetic           FN (C[1,0])              TP (C[1,1])
```

| Term | Full Name | Plain English |
|---|---|---|
| TN | True Negative | Correctly cleared as non-diabetic |
| FP | False Positive | False alarm — healthy flagged as diabetic |
| FN | False Negative | Missed diabetic — most dangerous error |
| TP | True Positive | Correctly identified as diabetic |

---

### Sensitivity (Recall)
**"Did we find the sick people?"**

Looks at the actual diabetic row — of all patients who truly have diabetes,
what percentage did the model correctly identify?

```
Sensitivity = TP / (TP + FN)
```

High sensitivity = few missed diabetics. In healthcare this is the priority
metric — missing a diabetic patient is more dangerous than a false alarm.

---

### Specificity
**"Did we clear the healthy people?"**

Looks at the actual non-diabetic row — of all patients who are truly healthy,
what percentage did the model correctly clear?

```
Specificity = TN / (TN + FP)
```

High specificity = few false alarms. Important because unnecessary treatment
carries its own costs and risks.

---

### Precision
**"Were our diabetic predictions accurate?"**

Of all patients the model predicted as diabetic, what percentage actually were?

```
Precision = TP / (TP + FP)
```

Looks down the predicted diabetic column rather than across the actual diabetic row.

---

### The Sensitivity/Specificity Tradeoff

You cannot always maximize both simultaneously. Lowering the classification
threshold catches more diabetics (higher sensitivity) but also flags more
healthy patients as diabetic (lower specificity). This is the core tension
in medical classification problems.

---

### F1 Score — Harmonic Mean of Precision and Recall
**"Are precision and recall in harmony?"**

In music, harmony means all notes working well together — no single note
dominates. F1 works the same way:

- Both precision and recall must be strong — one cannot carry the other
- If either is weak it pulls the whole score down, just like one off-key
  note ruins the harmony
- A truly harmonious model finds the balance between catching diabetics
  (recall) and being accurate about it (precision)

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The harmonic mean is used instead of a regular average because it is always
pulled toward the lower value — if either precision or recall is poor, F1
will be low regardless of how good the other is.

| Mean Type | Precision | Recall | Result |
|---|---|---|---|
| Regular average | 99% | 1% | 50% — misleadingly high |
| Harmonic mean (F1) | 99% | 1% | ~2% — honest, exposes the imbalance |

---

### Weighted Average vs Macro Average

Both appear in sklearn's `classification_report`:

- **Macro average** — simple average across both classes, treats them equally
- **Weighted average** — weights each class's metrics by its support (sample
  count). More representative for imbalanced datasets since it accounts for
  the 65/35 class split

---

### AUC-ROC
**"How well does the model separate diabetics from non-diabetics overall?"**

The ROC curve plots True Positive Rate (Sensitivity) on the y-axis against
False Positive Rate (1 - Specificity) on the x-axis at every possible
classification threshold. The AUC (Area Under the Curve) summarizes this
into a single number:

| AUC Score | Interpretation |
|---|---|
| 1.0 | Perfect model |
| 0.9 — 1.0 | Excellent |
| 0.8 — 0.9 | Good |
| 0.7 — 0.8 | Acceptable |
| 0.5 | No better than random guessing |
| Below 0.5 | Worse than random |

A curve hugging the top-left corner = excellent model.
A diagonal straight line = random guessing.

---

### Why Accuracy Alone Is Not Sufficient

With a 65/35 class imbalance a naive model that always predicts non-diabetic
would achieve 65% accuracy without learning anything meaningful. This is why
sensitivity, specificity, F1, and AUC-ROC are used instead of relying on
accuracy alone.

---

## Installation

### Prerequisites
- Python 3.14.0
- pip (included with Python)
- Git

### Clone the Repository

```bash
git clone https://github.com/DewDropRS/caltech_ctme_healthcare_capstone.git
cd caltech_ctme_healthcare_capstone
```

### Create and Activate a Virtual Environment

**Windows:**
```bash
python -m venv .venv --upgrade-deps
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv .venv --upgrade-deps
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add the Dataset

Place the raw dataset in the following location:

```
data/raw/health_care_diabetes.csv
```

The raw data directory is excluded from version control via `.gitignore`.
The dataset can be obtained from the
[UCI Machine Learning Repository — Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

---

## How to Run

Run the full pipeline from the project root:

```bash
python main.py
```

The pipeline executes the following steps in order:
1. EDA on raw data
2. EDA on cleaned data
3. Feature engineering
4. Model training and evaluation
5. Saves consolidated findings report

### Output Locations

| Output Type | Location |
|---|---|
| Figures and charts | `outputs/figures/` |
| Classification reports | `outputs/reports/` |
| Trained models | `outputs/reports/` |
| Consolidated findings | `outputs/reports/project_findings.csv` |
| Pipeline log | `pipeline.log` (project root) |

### Pipeline Log

All pipeline status messages are logged to both the console and `pipeline.log`
in the project root. Log format:

```
2026-05-02 17:43:49,888 [module_name] message
```

---

## Project Structure

```
caltech_ctme_healthcare_capstone/
│
├── data/
│   ├── raw/                        # Original dataset (not tracked in Git)
│   └── processed/                  # Cleaned dataset generated by pipeline
│
├── docs/                           # Case study instructions
│
├── outputs/
│   ├── figures/                    # All charts and visualizations (.png)
│   └── reports/                    # Classification reports, model results,
│                                   # trained models (.csv, .joblib)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Load raw data, treat zeros as missing,
│   │                               # impute using median by Outcome group
│   ├── eda_raw.py                  # EDA on raw data — documents data quality issues
│   ├── eda_clean.py                # EDA on cleaned data — pairplots, heatmap,
│   │                               # correlation analysis, age bin analysis
│   ├── features.py                 # Feature engineering — ordinal age bin encoding
│   ├── model.py                    # KNN, Logistic Regression, Random Forest —
│   │                               # training, tuning, evaluation, and saving
│   └── utils.py                    # Shared helpers, constants, logger configuration
│
├── config.py                       # Centralized paths, constants, and settings
├── main.py                         # Pipeline entry point
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes data/raw/, .venv/, __pycache__/
└── README.md
```

---

## What's Next

### Data & Feature Engineering
- Revisit Insulin imputation strategy — with 49% missing values, alternatives
  such as KNN imputation or a binary missing indicator flag (`insulin_was_missing`)
  would produce more reliable feature importance results
- Explore interaction features — e.g. Glucose × BMI as a combined metabolic
  risk indicator
- Investigate SMOTE (Synthetic Minority Oversampling Technique) to address
  class imbalance rather than relying solely on stratified sampling

### Modeling
- Tune classification threshold — the default 0.5 threshold may not be optimal
  for a clinical setting where sensitivity is prioritized. Evaluate performance
  at thresholds of 0.3-0.4 to catch more diabetics at the cost of more false alarms
- Add XGBoost as a fourth model for comparison
- Explore model stacking — combine KNN and Random Forest predictions as a
  meta-learner input
- Add decision tree visualization using graphviz to illustrate individual
  Random Forest trees

### Pipeline & Code Quality
- Replace print-based testing with a formal unit test suite using `pytest`
- Replace `sys.path.insert` with a proper package structure using `pyproject.toml`
- Replace `logging` print statements with structured logging to a rotating
  file handler for production use

### Deployment
- Build a simple prediction interface using Streamlit — allow a clinician to
  input patient data and receive a diabetes risk prediction with probability score
- Containerize the pipeline using Docker for reproducible deployment
- Explore SHAP (SHapley Additive exPlanations) values for model explainability —
  particularly important in a clinical context where predictions must be
  interpretable

---

## Author

**Rocio Segura**
Data Analytics Professional | Healthcare Analytics Specialist

- 15+ years of experience in healthcare analytics 
- Caltech CTME Data Science Bootcamp — Capstone Project
- Stanford Statistics for AI/ML

[![GitHub](https://img.shields.io/badge/GitHub-DewDropRS-181717?style=flat&logo=github)](https://github.com/DewDropRS)

---

*This project is part of the Caltech CTME Data Science Bootcamp capstone series.*