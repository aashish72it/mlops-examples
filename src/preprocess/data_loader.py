import pandas as pd
from src.core.config import Config
from src.utils.exceptions import DataLoadError

cfg = Config()

## Synthetic Financial Datasets For Fraud Detection
## Download link: https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download

class DataLoader:
    def __init__(self, path=cfg.DATA_PATH):
        self.path = path

    def load_data(self):
        try:
            df = pd.read_csv(self.path)
            return df
        except Exception as e:
            raise DataLoadError(f"Failed to load data: {e}")