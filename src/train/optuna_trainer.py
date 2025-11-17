import os
import optuna
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.preprocess.data_loader import DataLoader
from src.preprocess.preprocess import Preprocessor
from src.core.config import Config
from src.utils.logger import get_logger
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


logger = get_logger(__name__)
cfg = Config()

class OptunaTrainer:
    def __init__(self, n_trials=cfg.OPTUNA_TRIALS, study_name=cfg.OPTUNA_STUDY,
                 storage=cfg.OPTUNA_STORAGE, enable_pruning=cfg.OPTUNA_PRUNING):
        """
        OptunaTrainer runs hyperparameter optimization and trains the final model.
        Args:
            n_trials: number of optimization trials
            study_name: name of the Optuna study
            storage: backend storage (default: sqlite file)
            enable_pruning: whether to enable Optuna pruning (early stopping of bad trials)
        """
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.enable_pruning = enable_pruning
        self.best_params = None
        self.best_score = None
        self.model = None

    def _objective(self, trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", cfg.OPTUNA_N_ESTIMATORS_MIN, cfg.OPTUNA_N_ESTIMATORS_MAX)
        max_depth = trial.suggest_int("max_depth", cfg.OPTUNA_MAX_DEPTH_MIN, cfg.OPTUNA_MAX_DEPTH_MAX)
        min_samples_split = trial.suggest_int("min_samples_split", cfg.OPTUNA_MIN_SAMPLES_SPLIT_MIN, cfg.OPTUNA_MIN_SAMPLES_SPLIT_MAX)


        # Load and preprocess data
        loader = DataLoader()
        df = loader.load_data()
        preprocessor = Preprocessor(encoding_type=cfg.ENCODING_TYPE)
        df = preprocessor.preprocess(df)

        X = df[cfg.FEATURES]
        y = df[cfg.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=cfg.RANDOM_STATE
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metric to optimize
        if cfg.OPTUNA_METRIC == "f1":
            score = f1_score(y_test, y_pred)
        elif cfg.OPTUNA_METRIC == "roc_auc":
            score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        elif cfg.OPTUNA_METRIC == "pr_auc":
            score = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            raise ValueError(f"Unsupported metric: {cfg.OPTUNA_METRIC}")

        # Report intermediate score for pruning
        if self.enable_pruning:
            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return score

    def run(self):
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials...")
        pruner = optuna.pruners.MedianPruner() if self.enable_pruning else None

        study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            pruner=pruner
        )

        cpu_count = os.cpu_count()
        optuna_job_count = cfg.OPTUNA_JOBS
        if cpu_count >= optuna_job_count:
            logger.info(f"Optuna will run with n_jobs={cpu_count}")
        else:
            logger.info(f"Using configured Optuna job count: {optuna_job_count}")
            cpu_count = optuna_job_count


        study.optimize(self._objective,
                    n_trials=self.n_trials,
                    n_jobs=cpu_count)

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best {cfg.OPTUNA_METRIC} score: {self.best_score:.4f}")

        # 🔥 Log to MLflow
        if mlflow.active_run() is None:
            with mlflow.start_run(run_name=f"{self.study_name}_optuna"):
                mlflow.log_params(self.best_params)
                mlflow.log_metric(cfg.OPTUNA_METRIC, self.best_score)
        else:
            mlflow.log_params(self.best_params)
            mlflow.log_metric(cfg.OPTUNA_METRIC, self.best_score)



        # Train final model with best params
        loader = DataLoader()
        df = loader.load_data()
        preprocessor = Preprocessor(encoding_type=cfg.ENCODING_TYPE)
        df = preprocessor.preprocess(df)

        X = df[cfg.FEATURES]
        y = df[cfg.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
        )

        self.model = RandomForestClassifier(
            **self.best_params,
            random_state=cfg.RANDOM_STATE
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        if cfg.OPTUNA_METRIC == "f1":
            score = f1_score(y_test, y_pred)
        elif cfg.OPTUNA_METRIC == "roc_auc":
            score = roc_auc_score(y_test, y_proba)
        elif cfg.OPTUNA_METRIC == "pr_auc":
            score = average_precision_score(y_test, y_proba)
        else:
            raise ValueError(f"Unsupported metric: {cfg.OPTUNA_METRIC}")

        logger.info(f"Final {cfg.OPTUNA_METRIC} score on test set: {score:.4f}")
        return self.model, X_test, y_test, y_pred, y_proba, self.best_score
