from src.train.train import Trainer
#from src.train.optuna_trainer import OptunaTrainer
from src.evaluate.evaluate import Evaluator
from src.deploy.model_manager import ModelManager
import mlflow

def run_pipeline():
    # Phase 1: Train + Evaluate
    trainer = Trainer(model_type="logistic")  # or "random_forest"
    model, X_test, y_test, y_pred, y_proba = trainer.train()

    #optuna_trainer = OptunaTrainer()
    #model, X_test, y_test, y_pred, y_proba = optuna_trainer.run()

    evaluator = Evaluator()
    evaluator.log_confusion_matrix(y_test, y_pred)
    evaluator.log_roc_curve(y_test, y_proba)

    # Phase 2: Register + Deploy + Serve
    run_id = mlflow.last_active_run().info.run_id
    manager = ModelManager()

    result = manager.register(run_id)
    manager.deploy(version=result.version)
    manager.serve()

if __name__ == "__main__":
    run_pipeline()
