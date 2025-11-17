import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.exceptions import PreprocessError
from src.core.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

cfg = Config()

class Preprocessor:
    def __init__(self, encoding_type=cfg.ENCODING_TYPE):
        self.encoding_type = encoding_type
        self.label_encoder = LabelEncoder()

    def encode_type(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.encoding_type == "label":
                df["type"] = self.label_encoder.fit_transform(df["type"])
            elif self.encoding_type == "onehot":
                df = pd.get_dummies(df, columns=["type"])
            else:
                raise PreprocessError(f"Unsupported encoding type: {self.encoding_type}")
            return df
        except Exception as e:
            raise PreprocessError(f"Encoding failed: {e}")

    def drop_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.drop(columns=["nameOrig", "nameDest"], errors="ignore")
            return df
        except Exception as e:
            raise PreprocessError(f"Dropping identifiers failed: {e}")

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Add balance difference features
            df["deltaOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
            df["deltaDest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
            return df
        except Exception as e:
            raise PreprocessError(f"Feature engineering failed: {e}")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Dropping identifiers...")
        df = self.drop_identifiers(df)

        logger.info("Encoding transaction type...")
        df = self.encode_type(df)

        logger.info("Adding engineered features...")
        df = self.feature_engineering(df)

        logger.info("Creating final dataframe...")
        df = df[cfg.FEATURES + [cfg.TARGET]]

        logger.info("Preprocessing complete.")
        return df
