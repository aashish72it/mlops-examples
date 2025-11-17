import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

from src.preprocess.data_loader import DataLoader
from src.preprocess.preprocess import Preprocessor
from src.core.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = Config()

class Trainer:
    def __init__(self, model_type="logistic"):
        """
        model_type: 'logistic' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None

        # Set MLflow tracking URI from config
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    def _get_model(self):
        if self.model_type == "logistic":
            logger.info("Initializing Logistic Regression with RobustScaler...")
            return Pipeline([
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(max_iter=1000))
            ])
        elif self.model_type == "random_forest":
            logger.info("Initializing Random Forest Classifier...")
            return RandomForestClassifier(
                n_estimators=100,
                random_state=cfg.RANDOM_STATE
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self):
        logger.info("Loading and preprocessing data...")
        loader = DataLoader()
        df = loader.load_data()
        preprocessor = Preprocessor(encoding_type=cfg.ENCODING_TYPE)
        df = preprocessor.preprocess(df)

        X = df[cfg.FEATURES]
        y = df[cfg.TARGET]

        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
        )

        self.model = self._get_model()
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)

        logger.info("Generating predictions...")
        y_pred = self.model.predict(X_test)

        y_proba = None
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(
            f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}"
        )

        # MLflow logging
        logger.info("Logging metrics and model to MLflow...")
        with mlflow.start_run():
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("test_size", cfg.TEST_SIZE)
            mlflow.log_param("random_state", cfg.RANDOM_STATE)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)

            signature = infer_signature(X_train, self.model.predict(X_train))

            mlflow.sklearn.log_model(
                self.model,
                name=cfg.BINARY_CLASSIFICATION_MODEL_NAME,
                input_example=X_test[:5],
                signature=signature
            )

        logger.info("Training complete.")
        return self.model, X_test, y_test, y_pred, y_proba
