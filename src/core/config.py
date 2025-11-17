import os
from dotenv import load_dotenv
from src.utils.exceptions import ConfigError

class Config:
    def __init__(self):
        load_dotenv()
        try:
            self.DATA_PATH = os.getenv("DATA_PATH")
            self.ENCODING_TYPE = os.getenv("ENCODING_TYPE")  # onehot or label
            self.TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
            self.RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
            features = os.getenv("FEATURES")
            self.FEATURES = features.split(",") if features else []
            self.TARGET = os.getenv("TARGET")
            self.LOG_DIR = os.getenv("LOG_DIR")
            self.BINARY_CLASSIFICATION_MODEL_NAME = os.getenv("BINARY_CLASSIFICATION_MODEL_NAME")
            self.MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
            self.OPTUNA_STUDY = os.getenv("OPTUNA_STUDY")
            self.OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE")
            self.OPTUNA_PRUNING = os.getenv("OPTUNA_PRUNING", True)
            self.OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", 30))
            self.OPTUNA_JOBS = int(os.getenv("OPTUNA_JOBS", 4))
            self.OPTUNA_N_ESTIMATORS_MIN = int(os.getenv("OPTUNA_N_ESTIMATORS_MIN", 50))
            self.OPTUNA_N_ESTIMATORS_MAX = int(os.getenv("OPTUNA_N_ESTIMATORS_MAX", 300))
            self.OPTUNA_MAX_DEPTH_MIN = int(os.getenv("OPTUNA_MAX_DEPTH_MIN", 3))
            self.OPTUNA_MAX_DEPTH_MAX = int(os.getenv("OPTUNA_MAX_DEPTH_MAX", 20))
            self.OPTUNA_MIN_SAMPLES_SPLIT_MIN = int(os.getenv("OPTUNA_MIN_SAMPLES_SPLIT_MIN", 2))
            self.OPTUNA_MIN_SAMPLES_SPLIT_MAX = int(os.getenv("OPTUNA_MIN_SAMPLES_SPLIT_MAX", 20))
            self.OPTUNA_METRIC = os.getenv("OPTUNA_METRIC", "f1")  # options: f1, roc_auc, pr_auc
            self.MODEL_STAGE = os.getenv("MODEL_STAGE", "stg") #options: stg, prod
            self.MODEL_PORT = int(os.getenv("MODEL_PORT", 5000))

            if not self.DATA_PATH:
                raise ConfigError("DATA_PATH missing in environment.")

        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")