
import os
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Stage 1: Load Dataset ----------------
def load_california():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    return X, y, df

if __name__ == "__main__":
    try:
        # Load dataset
        X, y, full_df = load_california()
        print(f"California Housing loaded. Rows={len(full_df)}, Shape={full_df.shape}")

        # ---------------- Stage 2: Train-Test Split ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.7, random_state=42
        )
        print(f"Training size: {len(X_train)}, Test size: {len(X_test)}")

        # ---------------- Stage 3: Train Linear Regression Model ----------------
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # ---------------- Stage 4: Evaluate Model ----------------
        y_pred = lr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # ---------------- Stage 5: Save Model ----------------
        Path("./trained_models").mkdir(parents=True, exist_ok=True)
        with open("./trained_models/california_lr.pkl", "wb") as f:
            pickle.dump(lr, f)

        # ---------------- Stage 6: SHAP Explanations ----------------
        background = X_train.sample(n=min(100, len(X_train)), random_state=42)
        explainer = shap.Explainer(lr, background)
        shap_values = explainer(X_test)

        # Global Feature Importance
        shap_abs_mean = np.abs(shap_values.values).mean(axis=0)
        feature_names = list(X_train.columns)

        Path("./explanations/lr/").mkdir(parents=True, exist_ok=True)

        # Bar Plot
        order = np.argsort(shap_abs_mean)[::-1]
        plt.figure(figsize=(7, 5))
        plt.bar([feature_names[i] for i in order], shap_abs_mean[order], color="#4c72b0")
        plt.xticks(rotation=30, ha="right")
        plt.title("Global Feature Importance (mean |SHAP|) - Linear Regression")
        plt.tight_layout()
        plt.savefig("./explanations/lr/lr_shap_summary_bar.png", dpi=160)
        plt.close()

        # Beeswarm Plot
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("./explanations/lr/lr_shap_beeswarm.png", dpi=160)
        plt.close()

        print("SHAP explanations saved in ./explanations/lr/")

        # ---------------- Stage 7: Online Inference Example ----------------
        new_data = X_test.iloc[[0]]
        pred_value = lr.predict(new_data)[0]
        single_shap_values = explainer(new_data)

        shap_html = shap.plots.force(single_shap_values)
        shap.save_html("./explanations/lr/lr_shap_force_single.html", shap_html)
        print("Interactive SHAP force plot saved as HTML.")

        # Textual explanation for top 3 contributing features
        contributions = list(zip(feature_names, single_shap_values.values[0]))
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        top3 = contributions_sorted[:3]

        print("="*50)
        print(f"Prediction: {pred_value:.4f}. Top contributing features:")
        for feat, val in top3:
            direction = "positive" if val > 0 else "negative"
            print(f"- {feat} ({direction} impact: {val:.4f})")
        print("="*50)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


"""
California Housing loaded. Rows=20640, Shape=(20640, 9)
Training size: 6192, Test size: 14448
RMSE: 0.7263, MAE: 0.5297, R²: 0.6031

==================================================
Prediction: 0.7349. Top contributing features:
- MedInc (negative impact: -0.8342)
- Longitude (negative impact: -0.2011)
- Latitude (negative impact: -0.1860)
==================================================

"""