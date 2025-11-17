import mlflow
import os
from src.core.config import Config
from src.utils.logger import get_logger

cfg = Config()
logger = get_logger(__name__)

class ModelManager:
    def __init__(self, model_name=cfg.BINARY_CLASSIFICATION_MODEL_NAME):
        """
        Handles model registration, deployment, and serving.
        """
        self.model_name = model_name
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    def register(self, run_id: str):
        """
        Register a model from a given MLflow run ID.
        """
        model_uri = f"runs:/{run_id}/{self.model_name}"
        logger.info(f"Registering model from {model_uri}...")

        result = mlflow.register_model(model_uri, self.model_name)
        logger.info(f"Model registered: {result.name}, version {result.version}")
        return result

    def deploy(self, version: int, stage=cfg.MODEL_STAGE):
        """
        Transition a registered model version to a given stage (e.g., sqa, prod).
        """
        client = mlflow.tracking.MlflowClient()
        logger.info(f"Transitioning {self.model_name} v{version} to stage={stage}...")
        client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        logger.info(f"Model {self.model_name} v{version} is now in {stage} stage.")

    def serve(self, stage=cfg.MODEL_STAGE, port=cfg.MODEL_PORT):
        """
        Serve the model locally via MLflow's REST API.
        """
        logger.info(f"Serving {self.model_name} at stage={stage} on port={port}...")
        os.system(
            f"mlflow models serve -m 'models:/{self.model_name}/{stage}' -p {port}"
        )
