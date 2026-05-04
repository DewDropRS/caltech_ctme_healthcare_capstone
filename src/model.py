# -----------------------------------------------------------------------------
# File:    src/model.py
# Project: Caltech CTME Healthcare Capstone
# Purpose: Modeling — Compares and evaluates three models: KNN, Logistic Regression
# and Random Forest.
# -----------------------------------------------------------------------------

# Third party — installed packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Local — project modules
from utils import (
    add_finding,
    logger,
    EVAL_MEASURE_LABELS,
    COLUMN_LABELS
)

from config import (
    RANDOM_STATE,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_REPORTS_DIR,
    MODEL_ROC_CURVE,
    MODEL_CONFUSION_MATRIX,
    MODEL_RESULTS,
    MODEL_SCALER,
    MODEL_KNN,
    MODEL_LR,
    MODEL_RF,
    MODEL_RESULTS_HBAR,
    MODEL_RF_FEATURE_IMPORTANCE_HBAR

)


def split_and_scale(X, y, findings):
    """
    Performs 80/20 train/test split, scales train features, transforms the test features, and saves the scaler.
    
    :param X: feature matrix X (pd.DataFrame)
    :param y: target vector y (pd.Series)
    :param findings: findings list
    :return: X_train_scaled, X_test_scaled, y_train, y_test (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
    """

    # Perform train/test split into numpy arrays
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # StandardScaler is required for KNN which uses distance-based calculations
    scaler = StandardScaler()

    # fit_transform() calculates the mean and standard deviation of each feature
    # then transforms all values to have mean=0 and std=1
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns) # noqa N806

    # apply scaling learned from training to avoid data leakage using transform method
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X.columns) # noqa N806
    
    # Save scaler
    joblib.dump(scaler, MODEL_SCALER)
    logger.info(f'Scaler saved to {MODEL_SCALER.name}')

    add_finding(findings, 'Model', 'Train/Test Split',
                '80/20 stratified split — 614 training rows, 154 test rows')

    logger.info(f'Train/Test split and scaling complete')
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def run_knn(X_train, X_test, y_train, y_test, findings):
    """
    Trains a K-Nearest Neighbors classifier using GridSearchCV to find the
    optimal hyperparameters via 5-fold cross-validation. Evaluates the best
    model using confusion matrix, classification report, and AUC-ROC score.
    Returns FPR, TPR arrays for ROC curve plotting and a results dictionary
    of key evaluation metrics.

    :param X_train: scaled training feature matrix (pd.DataFrame)
    :param X_test: scaled test feature matrix (pd.DataFrame)
    :param y_train: training target vector (pd.Series)
    :param y_test: test target vector (pd.Series)
    :param findings: shared findings list
    :return: fpr (np array), tpr (np array), results (evaluation measures dictionary)
    """

    # Define parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Grid search with 5-fold CV
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='roc_auc')
    # Fit GridSearchCV on training data — tests all hyperparameter combinations
    # using 5-fold cross-validation and identifies the best performing model
    grid_search.fit(X_train, y_train)

    # Run the best fitted model
    best_knn = grid_search.best_estimator_

    # Run the predictions
    y_pred = best_knn.predict(X_test)
    y_prob = best_knn.predict_proba(X_test)[:, 1]

    # findings
    add_finding(findings, 'Model - KNN', 'Best Parameters', str(grid_search.best_params_))
    add_finding(findings, 'Model - KNN', 'Best CV Score', f'{grid_search.best_score_:.4f}')
    add_finding(findings, 'Model - KNN', 'Test Accuracy', f'{grid_search.score(X_test, y_test):.4f}')

    # Save the model
    joblib.dump(best_knn, MODEL_KNN)
    logger.info(f'KNN model trained and saved to {MODEL_KNN.name}')

    fpr, tpr, results = evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob, model_name='KNN', findings=findings)

    return fpr, tpr, results


def run_logistic(X_train, X_test, y_train, y_test, findings):
    """
    Trains a Logistic Regression classifier using GridSearchCV to find the
    optimal hyperparameters via 5-fold cross-validation. Evaluates the best
    model using confusion matrix, classification report, and AUC-ROC score.
    Returns FPR, TPR arrays for ROC curve plotting and a results dictionary
    of key evaluation metrics.

    :param X_train: scaled training feature matrix (pd.DataFrame)
    :param X_test: scaled test feature matrix (pd.DataFrame)
    :param y_train: training target vector (pd.Series)
    :param y_test: test target vector (pd.Series)
    :param findings: shared findings list
    :return: fpr (np array), tpr (np array), results (evaluation measures dictionary)
    """


    # GridSearchCV parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0, 0.5, 1]
    }

    # Grid search with 5-fold CV
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000, solver='saga'), param_grid, cv=5, scoring='roc_auc')
    # Fit GridSearchCV on training data — tests all hyperparameter combinations
    # using 5-fold cross-validation and identifies the best performing model
    grid_search.fit(X_train, y_train)

    # Run the best fitted model
    best_logreg = grid_search.best_estimator_

    # Run the predictions
    y_pred = best_logreg.predict(X_test)
    y_prob = best_logreg.predict_proba(X_test)[:, 1]

    # findings
    add_finding(findings, 'Model - Logistic Regression', 'Best Parameters', str(grid_search.best_params_))
    add_finding(findings, 'Model - Logistic Regression', 'Best CV Score', f'{grid_search.best_score_:.4f}')
    add_finding(findings, 'Model - Logistic Regression', 'Test Accuracy', f'{grid_search.score(X_test, y_test):.4f}')

    # Save the model
    joblib.dump(best_logreg, MODEL_LR)
    logger.info(f'Logistic Regression model trained and saved to {MODEL_LR.name}')

    fpr, tpr, results = evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob, model_name='Logistic_Regression'
                                       , findings=findings)

    return fpr, tpr, results


def run_random_forest(X_train, X_test, y_train, y_test, findings):
    """
    Trains a Random Forest classifier using GridSearchCV to find the
    optimal hyperparameters via 5-fold cross-validation. Evaluates the best
    model using confusion matrix, classification report, and AUC-ROC score.
    Returns FPR, TPR arrays for ROC curve plotting and a results dictionary
    of key evaluation metrics.

    :param X_train: scaled training feature matrix (pd.DataFrame)
    :param X_test: scaled test feature matrix (pd.DataFrame)
    :param y_train: training target vector (pd.Series)
    :param y_test: test target vector (pd.Series)
    :param findings: shared findings list
    :return: fpr (np array), tpr (np array), results (evaluation measures dictionary)
    """

    # GridSearchCV parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'bootstrap': [True]
    }

    # Grid search with 5-fold CV
    classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(estimator=classifier
                               , param_grid=param_grid
                               , cv=5
                               , scoring='roc_auc'
                               , n_jobs=-1)
    # Fit GridSearchCV on training data — tests all hyperparameter combinations
    # using 5-fold cross-validation and identifies the best performing model
    grid_search.fit(X_train, y_train)

    # Run the best fitted model
    best_rf = grid_search.best_estimator_
    # Run the predictions
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]

    # findings
    add_finding(findings, 'Model - Random Forest', 'Best Parameters', str(grid_search.best_params_))
    add_finding(findings, 'Model - Random Forest', 'Best CV Score', f'{grid_search.best_score_:.4f}')
    add_finding(findings, 'Model - Random Forest', 'Test Accuracy', f'{grid_search.score(X_test, y_test):.4f}')

    # Save the model
    joblib.dump(best_rf, MODEL_RF)
    logger.info(f'Random Forest model trained and saved to {MODEL_RF.name}')

    fpr, tpr, results = evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob, model_name='Random_Forest'
                                       , findings=findings)
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    importance_df['importance'] = (importance_df['importance'] * 100).round(1)
    importance_df['feature'] = importance_df['feature'].map(COLUMN_LABELS)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', color='steelblue', ax=ax, orient='h')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=12)
    ax.set_title('Random Forest - Feature Importance', fontsize=16)
    ax.set_xlabel('Importance (%)')
    ax.set_ylabel('')
    plt.figtext(0.1, -0.04,
                'Note: Insulin\'s high importance (42%) may reflect imputation bias — 49% of Insulin values were missing\n'
                'and replaced with group medians, potentially creating an artificially clean split pattern.\n'
                'Despite this, Random Forest achieved strong overall performance (AUC = 93.9%).',
                ha='left', fontsize=8, style='italic')
    plt.tight_layout()
    plt.savefig(MODEL_RF_FEATURE_IMPORTANCE_HBAR
                , dpi=150
                , bbox_inches='tight'
                )
    plt.close()
    logger.info(f'Random Forest - Feature Importance saved to {MODEL_RF_FEATURE_IMPORTANCE_HBAR.name}')

    return fpr, tpr, results


def evaluate_model(y_test, y_pred, y_prob, model_name, findings):
    """
    Evaluates a trained model using confusion matrix, classification report,
    and AUC-ROC score. Saves confusion matrix plot and classification report
    to outputs. Logs key metrics to findings.

    :param y_test: true target values (pd.Series)
    :param y_pred: predicted class labels (np.array)
    :param y_prob: predicted probabilities for positive class (np.array)
    :param model_name: name of the model for file naming adn findings (str)
    :param findings: shared findings list
    :return: fpr (np array), tpr (np array), measures_dict (dictionary)
    """

    cm = confusion_matrix(y_test, y_pred) # an ndarray
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    sensitivity = round((TP / (TP + FN)) * 100, 1)
    specificity = round((TN / (TN + FP)) * 100, 1)
    precision = round((TP/ (TP + FP)) * 100, 1)

    # Display the confusion matrix plot
    label_array = np.array([
                    [f'{TN} (TN)\nTrue Non-Diabetic\nSpecificity contribution', f'{FP} (FP)\nFalse Alarm'],
                    [f'{FN} (FN)\nMissed Diabetic', f'{TP} (TP)\nTrue Diabetic\nSensitivity contribution']
                   ])

    cm_path = OUTPUTS_FIGURES_DIR / f'model_{model_name}_confusion_matrix.png'
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    sns.heatmap(data = cm,
                ax = ax,
                annot=label_array,
                fmt='',
                cmap ='Blues',
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'],
                annot_kws={'size': 12}
                )
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    ax.set_xlabel('Predicted Outcome', fontsize=12)
    ax.set_ylabel('True Outcome', fontsize=12)
    plt.figtext(0.1, -0.04,
                f'Sensitivity/recall (TP/(TP+FN)): {sensitivity}%\nSpecificity (TN/(TN+FP)): {specificity}%\nPrecision (TP/(TP+FP)): {precision}%',
                ha='left', fontsize=8, style='italic')
    plt.tight_layout()
    plt.savefig(cm_path
                , dpi=150
                , bbox_inches='tight'
                )
    plt.close()

    # Classification report
    cr_path = OUTPUTS_REPORTS_DIR / f'model_{model_name}_classification_report.csv'
    # by default will return a string unless you specify output_dict = True
    class_rep = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(class_rep).transpose().to_csv(cr_path)
    logger.info(f'{model_name} Classification report saved to {cr_path.name}')

    # F1 score
    f1 = round(class_rep['1']['f1-score'] * 100, 1)
    # AUC-ROC
    auc_score = round(roc_auc_score(y_test, y_prob) * 100, 1)

    # False Positive Rate (FPR) = FP / (FP + TN) = 1 - Specificity
    # True Positive Rate (TPR) = TP / (TP + FN) = Sensitivity
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Dictionary
    measures_dict = {
        'model': model_name,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc_roc': auc_score
    }

    # Add findings
    add_finding(findings, f'Model - {model_name}', 'Sensitivity', f'{sensitivity}%')
    add_finding(findings, f'Model - {model_name}', 'Specificity', f'{specificity}%')
    add_finding(findings, f'Model - {model_name}', 'Precision', f'{precision}%')
    add_finding(findings, f'Model - {model_name}', 'F1 Score', f'{f1}%')
    add_finding(findings, f'Model - {model_name}', 'AUC-ROC', f'{auc_score}%')

    logger.info(f'{model_name} Evaluation Complete')

    return fpr, tpr, measures_dict


def build_models(X, y, findings):
    """
    Splits and scales the feature engineered data then runs three models for comparison: KNN, Logistic Regression,
    and Random Forest.

    :param X: a feature engineered matrix
    :param y: a target vector
    :param findings: shared findings list
    :return:
    """

    X_train, X_test, y_train, y_test = split_and_scale(X, y, findings)
    fpr_knn, tpr_knn, results_knn = run_knn(X_train, X_test, y_train, y_test, findings)
    fpr_lr, tpr_lr, results_lr = run_logistic(X_train, X_test, y_train, y_test, findings)
    fpr_rf, tpr_rf, results_rf = run_random_forest(X_train, X_test, y_train, y_test, findings)

    # Combine all results into a list and create DataFrame
    results_df = pd.DataFrame([results_knn, results_lr, results_rf])
    results_df.to_csv(MODEL_RESULTS, index=False)
    logger.info(f'Combined classification reports saved to {MODEL_RESULTS.name}')

    knn_auc = results_df.loc[results_df['model'] == 'KNN', 'auc_roc'].values[0]
    lr_auc = results_df.loc[results_df['model'] == 'Logistic_Regression', 'auc_roc'].values[0]
    rf_auc = results_df.loc[results_df['model'] == 'Random_Forest', 'auc_roc'].values[0]

    # AUC-ROC curve
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.plot(fpr_knn, tpr_knn, color='blue', label=f'KNN (AUC = {knn_auc}%)')
    ax.plot(fpr_lr, tpr_lr, color='orange', label=f'Logistic Regression (AUC = {lr_auc}%)')
    ax.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {rf_auc}%)')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing Baseline (AUC = 50%)')
    ax.legend()

    ax.set_title('ROC Curve Comparison — Diabetes Classification Models', fontsize = 16)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    plt.figtext(0.1, -0.04,
                'A curve closer to the top-left corner = better model. The diagonal = random guessing (AUC = 50%).\n'
                 'Higher AUC = better separation between diabetic and non-diabetic patients.',
                ha='left', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(MODEL_ROC_CURVE
                , dpi=150
                , bbox_inches='tight'
                )
    logger.info(f'AUC-ROC Curve saved to {MODEL_ROC_CURVE.name}')
    plt.close()

    # Model comparison bar chart
    measure_df = results_df.melt(
        id_vars=['model'],
        value_vars=['sensitivity', 'specificity', 'precision', 'f1', 'auc_roc'],
        var_name='metric',
        value_name='score')
    measure_df['metric'] = measure_df['metric'].map(EVAL_MEASURE_LABELS)
    measure_df['model'] = measure_df['model'].str.replace('_', ' ')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 8))
    sns.barplot(data=measure_df, x='score', y='metric', hue='model', ax=ax, orient='h')
    # show value annotations for each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    ax.set_title('Performance Comparison of Diabetes Classifier Models', fontsize=16)
    ax.set_xlabel('Score (%)')
    ax.set_ylabel('')
    ax.set_xlim(40, 105)
    plt.tight_layout()
    plt.savefig(MODEL_RESULTS_HBAR
                , dpi=150
                , bbox_inches='tight'
                )
    plt.close()
    
    logger.info(f'Performance Comparison saved to {MODEL_RESULTS_HBAR.name}')
    logger.info('All models saved to outputs/reports/ — load with joblib for future predictions')
    
    return None