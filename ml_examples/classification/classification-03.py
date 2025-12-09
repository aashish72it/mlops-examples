# ===========================================================================================
# Churn Prediction on Imbalanced Dataset (<10% churn)
# Full ML Pipeline: Baseline → SMOTE → Class Weights → RF → SHAP → Optuna → Threshold Tuning
# ===========================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import shap
import optuna
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. Input Values
# ============================================================
##data source - https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset?resource=download
SOURCE_FILE = "./datasets/Bank_Customer_Churn_Prediction.csv"
OPTUNA_TRIALS = 3
ENCODING_COLS = ["country", "gender"]
TARGET = "churn"
# ============================================================
# 1. Load Dataset
# ============================================================

def load_data(source, target_column):
    df = pd.read_csv(source)
    df = df.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# ============================================================
# 2. Preprocessing (One-Hot Encoding)
# ============================================================

def preprocess(features):
    X_processed = pd.get_dummies(features, columns=ENCODING_COLS, drop_first=True)
    print("Final feature shape:", X_processed.shape)
    print("Columns:", X_processed.columns.tolist())
    return X_processed

# ============================================================
# 3. Train/Test Split (Sampling technique: Stratified)
# ============================================================

def train_test_data(features, target, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, stratify=y, random_state=random_state
    )

    print("Train Churn Rate:", y_train.mean())
    print("Test Churn Rate:", y_test.mean())
    return X_train, X_test, y_train, y_test

# ============================================================
# 4. Generic Model Evaluation Function
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return model, y_pred, y_prob, cm, roc_auc

# ============================================================
# 5. SMOTE Oversampling
# ============================================================

def smote_evaluate(X_train, X_test, y_train, y_test):
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=200)
    return evaluate_model(model, X_train_sm, X_test, y_train_sm, y_test)

# ============================================================
# 6. Class Weights Logistic Regression
# ============================================================

def weighted_logistic_evaluate(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    return evaluate_model(model, X_train, X_test, y_train, y_test)

# ============================================================
# 7. Random Forest Evaluation
# ============================================================

def random_forest_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    return evaluate_model(model, X_train, X_test, y_train, y_test)

# ============================================================
# 8. SHAP Explainability
# ============================================================

def shap_explain(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, use shap_values[1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values

    print("\nGenerating SHAP summary plot...")
    shap.summary_plot(shap_to_plot, X_test)


# ============================================================
# 9. Optuna Hyperparameter Tuning for Random Forest
# ============================================================

def tune_random_forest_optuna(X_train, X_test, y_train, y_test, n_trials=OPTUNA_TRIALS):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "class_weight": "balanced",
            "random_state": 42
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Best Hyperparameters ===")
    print(study.best_params)

    best_model = RandomForestClassifier(
        **study.best_params, class_weight="balanced", random_state=42
    )
    best_model.fit(X_train, y_train)

    y_prob_best = best_model.predict_proba(X_test)[:, 1]

    return best_model, study.best_params, y_prob_best

# ============================================================
# 10. Threshold Tuning
# ============================================================

def threshold_tuning(y_test, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)

    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_prob >= best_threshold).astype(int)

    cm = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return best_threshold, y_pred, cm, roc_auc

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # Load + preprocess
    X, y = load_data(SOURCE_FILE, TARGET)
    X_p = preprocess(X)
    X_train, X_test, y_train, y_test = train_test_data(X_p, y)
    print("Data loaded & preprocessing completed...")

    # 1️ Baseline Logistic Regression
    model, y_pred, y_prob, cm, roc_auc = evaluate_model(
        LogisticRegression(max_iter=200),
        X_train, X_test, y_train, y_test
    )
    print("\n=== Baseline Logistic Regression ===")
    print(cm)
    print("ROC-AUC:", roc_auc)

    # 2️ SMOTE + Logistic Regression
    model, y_pred, y_prob, cm, roc_auc = smote_evaluate(
        X_train, X_test, y_train, y_test
    )
    print("\n=== After SMOTE ===")
    print(cm)
    print("ROC-AUC:", roc_auc)

    # 3️ Class Weights Logistic Regression
    model, y_pred, y_prob, cm, roc_auc = weighted_logistic_evaluate(
        X_train, X_test, y_train, y_test
    )
    print("\n=== After Class Weights ===")
    print(cm)
    print("ROC-AUC:", roc_auc)

    # 4️ Random Forest Evaluation
    rf_model, y_pred, y_prob, cm, roc_auc = random_forest_evaluate(
        X_train, X_test, y_train, y_test
    )
    print("\n=== Random Forest (Balanced) ===")
    print(cm)
    print("ROC-AUC:", roc_auc)

    # 5️ SHAP Explainability
    shap_explain(rf_model, X_train, X_test)

    # 6️ Optuna Hyperparameter Tuning
    best_rf, best_params, y_prob_best = tune_random_forest_optuna(
        X_train, X_test, y_train, y_test
    )
    print("====Optuna Results====")
    print(best_params)
    print(best_rf)

    # 7️ Threshold Tuning
    best_threshold, y_pred_best, cm_best, roc_auc_best = threshold_tuning(
        y_test, y_prob_best
    )
    print("\n=== Final Model (Threshold Tuned) ===")
    print("Best Threshold:", best_threshold)
    print(cm_best)
    print("ROC-AUC:", roc_auc_best)



# ##############################Output######################################################

# Final feature shape: (10000, 12)
# Columns: ['customer_id', 'credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'country_Germany', 'country_Spain', 'gender_Male']
# Train Churn Rate: 0.20375
# Test Churn Rate: 0.2035
# Data loaded & preprocessing completed...

# === Baseline Logistic Regression ===
#               precision    recall  f1-score   support

#            0       0.80      0.97      0.88      1593
#            1       0.29      0.05      0.08       407

#     accuracy                           0.78      2000
#    macro avg       0.54      0.51      0.48      2000
# weighted avg       0.70      0.78      0.71      2000

# ROC-AUC: 0.7393155867732139

# === After SMOTE ===
#               precision    recall  f1-score   support

#            0       0.90      0.71      0.80      1593
#            1       0.38      0.69      0.49       407

#     accuracy                           0.71      2000
#    macro avg       0.64      0.70      0.64      2000
# weighted avg       0.79      0.71      0.73      2000

# ROC-AUC: 0.7453802030073217

# === After Class Weights ===
#               precision    recall  f1-score   support

#            0       0.90      0.71      0.79      1593
#            1       0.37      0.68      0.48       407

#     accuracy                           0.70      2000
#    macro avg       0.64      0.69      0.64      2000
# weighted avg       0.79      0.70      0.73      2000

# ROC-AUC: 0.7443329307736087

# === Random Forest (Balanced) ===
#               precision    recall  f1-score   support

#            0       0.87      0.97      0.92      1593
#            1       0.79      0.43      0.56       407

#     accuracy                           0.86      2000
#    macro avg       0.83      0.70      0.74      2000
# weighted avg       0.85      0.86      0.84      2000

# ROC-AUC: 0.8538199216165319

# Generating SHAP summary plot...
# [I 2025-11-30 15:40:45,072] A new study created in memory with name: no-name-683a6bb4-8136-499d-83d3-8df07f5e01ff
# [I 2025-11-30 15:40:48,940] Trial 0 finished with value: 0.8610659966592171 and parameters: {'n_estimators': 342, 'max_depth': 13, 'min_samples_split': 15}. Best is trial 0 with value: 0.8610659966592171.
# [I 2025-11-30 15:40:52,633] Trial 1 finished with value: 0.8578856205974849 and parameters: {'n_estimators': 334, 'max_depth': 20, 'min_samples_split': 9}. Best is trial 0 with value: 0.8610659966592171.
# [I 2025-11-30 15:40:56,766] Trial 2 finished with value: 0.859475808628351 and parameters: {'n_estimators': 386, 'max_depth': 17, 'min_samples_split': 13}. Best is trial 0 with value: 0.8610659966592171.

# === Best Hyperparameters ===
# {'n_estimators': 342, 'max_depth': 13, 'min_samples_split': 15}
# ====Optuna Results====
# {'n_estimators': 342, 'max_depth': 13, 'min_samples_split': 15}
# RandomForestClassifier(class_weight='balanced', max_depth=13,
#                        min_samples_split=15, n_estimators=342, random_state=42)

# === Final Model (Threshold Tuned) ===
# Best Threshold: 0.42751943737622233
#               precision    recall  f1-score   support

#            0       0.92      0.85      0.89      1593
#            1       0.56      0.72      0.63       407

#     accuracy                           0.83      2000
#    macro avg       0.74      0.79      0.76      2000
# weighted avg       0.85      0.83      0.84      2000

# ROC-AUC: 0.8610659966592171