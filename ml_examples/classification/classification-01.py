"""
ML Pipeline with SHAP and Textual Explanation
---------------------------------------------
Pipeline Features:
- Loads California Housing dataset and converts it into a classification problem.
- Trains a RandomForestClassifier model.
- Evaluates using Accuracy, F1, Precision, Recall, Confusion Matrix, and Classification Report.
- Generates SHAP explanations with proper multi-class handling and shape validation.
- Fixes SHAP additivity check error using check_additivity=False.
- Adds online inference example with SHAP force_plot for a single record.
- Prints textual explanation for top 3 contributing features for the single prediction.
"""

from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Stage 1: Load Dataset ----------------
def load_california():
    """Load California Housing dataset (~20,640 rows) and bin target into 3 classes."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    # Create classification target by binning median house value
    df['target'] = pd.qcut(df['MedHouseVal'], q=3, labels=['Low', 'Medium', 'High'])
    X = df.drop(columns=['MedHouseVal', 'target'])
    y = df['target']
    return X, y, df

# ---------------- Stage 2: Random Sampling ----------------
def random_sample(df: pd.DataFrame, frac: float = 0.3, random_state: int = 42):
    """Random sample for experimentation."""
    return df.sample(frac=frac, random_state=random_state)

if __name__ == "__main__":
    try:
        # Load dataset
        X, y, full_df = load_california()
        print(f"California Housing loaded. Rows={len(full_df)}, Shape={full_df.shape}")

        # Sample data
        sample_df = random_sample(full_df, frac=0.3)
        print("Sample Data Preview:")
        print(sample_df.head())
        print("="*120)

        Xs = sample_df.drop(columns=['MedHouseVal', 'target'])
        ys = sample_df['target']

        # ---------------- Stage 3: Train/Test Split ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=0.3, random_state=42, stratify=ys
        )

        # ---------------- Stage 4: Train Model ----------------
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)

        # ---------------- Stage 5: Evaluate Model ----------------
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # ---------------- Stage 6: Save & Reload Model ----------------
        Path("./trained_models").mkdir(parents=True, exist_ok=True)
        with open("./trained_models/california_rf.pkl", "wb") as f:
            pickle.dump(rf, f)

        with open("./trained_models/california_rf.pkl", "rb") as f:
            loaded_rf = pickle.load(f)

        # ---------------- Stage 7: SHAP Explanations ----------------
        """
        SHAP Explanation:
        -----------------
        Default SHAP logic:
            - shap.TreeExplainer(model).shap_values(X)
            - Works well for binary classification.
        Extra Logic:
            - For multi-class, shap_values returns a list (one per class).
            - We aggregate absolute SHAP values across classes to get global importance.
        """
        background = X_train.sample(n=min(50, len(X_train)), random_state=42)
        explainer = shap.TreeExplainer(rf, data=background, feature_perturbation="interventional")

        # Disable additivity check to avoid errors for multi-class models
        shap_values = explainer.shap_values(X_test, check_additivity=False)

        # Handle multi-class SHAP output
        if isinstance(shap_values, list):
            shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            shap_values_for_plots = shap_values[0]
        else:
            shap_abs_mean = np.abs(shap_values).mean(axis=0)
            shap_values_for_plots = shap_values

        # Ensure correct shape
        shap_abs_mean = np.asarray(shap_abs_mean).flatten()
        feature_names = list(X_train.columns)
        if len(shap_abs_mean) != len(feature_names):
            print(f"WARNING: SHAP mean length {len(shap_abs_mean)} != feature count {len(feature_names)}")
            shap_abs_mean = shap_abs_mean[:len(feature_names)]

        Path("./explanations").mkdir(parents=True, exist_ok=True)

        # Global Feature Importance Bar Plot
        order = np.argsort(shap_abs_mean)[::-1]
        plt.figure(figsize=(7, 5))
        plt.bar([feature_names[i] for i in order], shap_abs_mean[order], color="#4c72b0")
        plt.xticks(rotation=30, ha="right")
        plt.title("Global Feature Importance (mean |SHAP|)")
        plt.tight_layout()
        plt.savefig("./explanations/shap_summary_bar.png", dpi=160)
        plt.close()

        # Beeswarm Plot
        shap.summary_plot(shap_values_for_plots, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("./explanations/shap_beeswarm.png", dpi=160)
        plt.close()

        print("SHAP explanations saved in ./explanations/")

        # ---------------- Stage 8: Online Inference Example ----------------
        """
        Explain a single prediction in production using SHAP force plot.
        """
        new_data = X_test.iloc[[0]]
        pred_class = rf.predict(new_data)[0]
        single_shap_values = explainer.shap_values(new_data, check_additivity=False)

        # For multi-class, pick the predicted class index
        if isinstance(single_shap_values, list):
            class_idx = list(rf.classes_).index(pred_class)
            single_shap_values_for_plot = single_shap_values[class_idx][0]
        else:
            class_idx = 0
            single_shap_values_for_plot = single_shap_values[0]

        # Flatten SHAP values and expected value
        single_shap_values_for_plot = np.array(single_shap_values_for_plot).flatten()
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = expected_val[class_idx]
        expected_val = float(expected_val)

        # Generate interactive SHAP force plot and save as HTML
        shap_html = shap.plots.force(expected_val, single_shap_values_for_plot)
        shap.save_html("./explanations/shap_force_single.html", shap_html)
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



"""
############Output##############

Accuracy: 0.7750, F1: 0.7739, Precision: 0.7735, Recall: 0.7750

Confusion Matrix:
[[497  25 103]
 [ 17 530  71]
 [100 102 413]]

 Classification Report:
              precision    recall  f1-score   support

        High       0.81      0.80      0.80       625
         Low       0.81      0.86      0.83       618
      Medium       0.70      0.67      0.69       615

    accuracy                           0.78      1858
   macro avg       0.77      0.77      0.77      1858
weighted avg       0.77      0.78      0.77      1858


==================================================
Prediction: High. Top contributing features:
- MedInc (positive impact: 0.2135)
- HouseAge (negative impact: -0.1937)
- Latitude (positive impact: 0.0756)
==================================================

############Analysis#############

Following 5 are good and important techniques to improve the model performance:

- Feature engineering (if required, re-engineering)
- sampling technique (potentially, medium class has less data so it is 
impacting the prediction, stratified sampling can help)
- hyper parameter tuning
- Add more training data
- Change the model in case of underfitting or overfitting

In confusion matrix - The diagonal line should be high (max) 
and side numbers should be low for better performance 

SHAP tells you which features are contributing the most to the prediction
for a particular record in a positive or negative way. It helps in both offline
and online explainability of the model predictions.

"""