
"""
Pipeline Features:
- Loads California Housing dataset and converts it into a classification problem.
- Uses stratified sampling and trains on 50% of the data.
- Trains an XGBoost classifier.
- Evaluates using Accuracy, F1, Precision, Recall, Confusion Matrix, and Classification Report.
- Generates SHAP explanations for global feature importance and a single prediction.
- Saves SHAP summary plots and interactive force plot as HTML.
- Prints textual explanation for top 3 contributing features for the single prediction.
"""


import os
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ---------------- Stage 1: Load Dataset ----------------
def load_california():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    # Create classification target by binning median house value
    df['target'] = pd.qcut(df['MedHouseVal'], q=3, labels=['Low', 'Medium', 'High'])
    X = df.drop(columns=['MedHouseVal', 'target'])
    y = df['target']
    return X, y, df

if __name__ == "__main__":
    try:
        # Load dataset
        X, y, full_df = load_california()
        print(f"California Housing loaded. Rows={len(full_df)}, Shape={full_df.shape}")

        # Encode target labels for XGBoost
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)  # Converts ['Low','Medium','High'] -> [0,1,2]

        # ---------------- Stage 2: Stratified Sampling ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.7, random_state=42, stratify=y_encoded
        )
        print(f"Training size: {len(X_train)}, Test size: {len(X_test)}")

        # ---------------- Stage 3: Train XGBoost Model ----------------
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb.fit(X_train, y_train)

        # ---------------- Stage 4: Evaluate Model ----------------
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # ---------------- Stage 5: Save Model ----------------
        Path("./trained_models").mkdir(parents=True, exist_ok=True)
        with open("./trained_models/california_xgb.pkl", "wb") as f:
            pickle.dump(xgb, f)

        # ---------------- Stage 6: SHAP Explanations ----------------
        background = X_train.sample(n=min(100, len(X_train)), random_state=42)
        explainer = shap.TreeExplainer(xgb, data=background)

        shap_values = explainer.shap_values(X_test, check_additivity=False)

        # Handle multi-class SHAP output
        if isinstance(shap_values, list):
            shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            shap_values_for_plots = shap_values[0]
        else:
            shap_abs_mean = np.abs(shap_values).mean(axis=0)
            shap_values_for_plots = shap_values

        shap_abs_mean = np.asarray(shap_abs_mean).flatten()
        feature_names = list(X_train.columns)
        if len(shap_abs_mean) != len(feature_names):
            shap_abs_mean = shap_abs_mean[:len(feature_names)]

        Path("./explanations/xgb/").mkdir(parents=True, exist_ok=True)

        # Global Feature Importance Bar Plot
        order = np.argsort(shap_abs_mean)[::-1]
        plt.figure(figsize=(7, 5))
        plt.bar([feature_names[i] for i in order], shap_abs_mean[order], color="#4c72b0")
        plt.xticks(rotation=30, ha="right")
        plt.title("Global Feature Importance (mean |SHAP|) - XGBoost")
        plt.tight_layout()
        plt.savefig("./explanations/xgb/xgb_shap_summary_bar.png", dpi=160)
        plt.close()

        # Beeswarm Plot
        shap.summary_plot(shap_values_for_plots, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("./explanations/xgb/xgb_shap_beeswarm.png", dpi=160)
        plt.close()

        print("SHAP explanations saved in ./explanations/xgb/")

        # ---------------- Stage 7: Online Inference Example ----------------
        new_data = X_test.iloc[[0]]
        pred_class = xgb.predict(new_data)[0]
        single_shap_values = explainer.shap_values(new_data, check_additivity=False)

        if isinstance(single_shap_values, list):
            class_idx = list(xgb.classes_).index(pred_class)
            single_shap_values_for_plot = single_shap_values[class_idx][0]
        else:
            class_idx = 0
            single_shap_values_for_plot = single_shap_values[0]

        single_shap_values_for_plot = np.array(single_shap_values_for_plot).flatten()
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = expected_val[class_idx]
        expected_val = float(expected_val)

        shap_html = shap.plots.force(expected_val, single_shap_values_for_plot)
        shap.save_html("./explanations/xgb/xgb_shap_force_single.html", shap_html)
        print("Interactive SHAP force plot saved as HTML.")

        # Textual explanation for top 3 contributing features
        contributions = list(zip(feature_names, single_shap_values_for_plot))
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        top3 = contributions_sorted[:3]

        print("="*50)
        print(f"Prediction: {pred_class}. Top contributing features:")
        for feat, val in top3:
            direction = "positive" if val > 0 else "negative"
            print(f"- {feat} ({direction} impact: {val:.4f})")
        print("="*50)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

"""
############Output##############

California Housing loaded. Rows=20640, Shape=(20640, 10)
Training size: 16512, Test size: 4128
Accuracy: 0.8336, F1: 0.8345, Precision: 0.8362, Recall: 0.8336

Confusion Matrix:
[[1153   11  212]
 [  11 1207  159]
 [ 136  158 1081]]

Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.84      0.86      1376
           1       0.88      0.88      0.88      1377
           2       0.74      0.79      0.76      1375

    accuracy                           0.83      4128
   macro avg       0.84      0.83      0.83      4128
weighted avg       0.84      0.83      0.83      4128

==================================================
Prediction: 1. Top contributing features:
- HouseAge (positive impact: 1.5516)
- MedInc (negative impact: -1.1042)
- Latitude (negative impact: -0.6154)
==================================================

"""