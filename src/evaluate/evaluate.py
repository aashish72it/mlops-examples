import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import mlflow

from src.utils.logger import get_logger
from src.core.config import Config

cfg = Config()
logger = get_logger(__name__)

class Evaluator:
    def __init__(self, run_name="evaluation"):
        """
        Evaluator handles model evaluation and logs artifacts to MLflow.
        """
        self.run_name = run_name

    def log_confusion_matrix(self, y_true, y_pred, labels=("Legit", "Fraud")):
        """
        Generate and log confusion matrix as an MLflow artifact.
        """
        logger.info("Generating confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        confusion_matrix_file = os.path.join(cfg.LOG_DIR,"confusion_matrix.png")
        cm_file_path = confusion_matrix_file
        plt.savefig(cm_file_path)
        plt.close()

        mlflow.log_artifact(cm_file_path)
        logger.info("Confusion matrix logged to MLflow.")

    def log_roc_curve(self, y_true, y_proba):
        """
        Generate and log ROC curve as an MLflow artifact.
        """
        if y_proba is None:
            logger.warning("ROC curve skipped: model does not provide probabilities.")
            return

        logger.info("Generating ROC curve...")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()

        roc_file = os.path.join(cfg.LOG_DIR,"roc_curve.png")
        roc_file_path = roc_file
        plt.savefig(roc_file_path)
        plt.close()

        mlflow.log_artifact(roc_file_path)
        logger.info("ROC curve logged to MLflow.")
