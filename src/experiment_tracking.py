import logging
import mlflow
import mlflow.pytorch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

from config import CFG


# LOGGER
# ─────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "[%(name)s] %(message)s"
)
logger = logging.getLogger("experiment_tracker")




class ExperimentTracker:
    def __init__(
        self,
        experiment_name : str  = "jane_street_forecasting",
        tracking_uri    : str  = None,
    ):
        if tracking_uri is None:
            # # Store in project root/mlruns/
            tracking_uri=str(CFG.paths.output_dir.parent/"mlruns")


        self.experiment_name=experiment_name
        self.tracking_uri=tracking_uri
        self.active_run=None
        mlflow.set_tracking_uri(f"file://{tracking_uri}")
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking URI : {tracking_uri}")
        logger.info(f"Experiment name     : {experiment_name}")
        logger.info(f"View UI at          : mlflow ui --port 5000")


    def start_run(self, run_name: str) -> None:
        self.active_run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run  : {run_name}")
        logger.info(f"Run ID              : {self.active_run.info.run_id}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info("MLflow run ended")


    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")


    def log_config(self) -> None:
        params = {
            # Data
            "data_start"      : CFG.data.data_start,
            "train_end"       : CFG.data.train_end,
            "val_start"       : CFG.data.val_start,
            "val_end"         : CFG.data.val_end,
            "test_start"      : CFG.data.test_start,
            "target"          : CFG.data.target,
            # Model
            "hidden_size"     : CFG.model.hidden_size,
            "dropout"         : CFG.model.dropout,
            "n_heads"         : CFG.model.n_heads,
            # Training
            "learning_rate"   : CFG.train.learning_rate,
            "max_epochs"      : CFG.train.max_epochs,
            "early_stop"      : CFG.train.early_stop,
            # Online learning
            "online_lr"       : CFG.online.learning_rate,
            # Features
            "n_top_features"  : CFG.features.n_top_features,
            "rolling_window"  : CFG.features.rolling_window,
        }
        self.log_params(params)


    def log_metric(
        self,
        name  : str,
        value : float,
        step  : int = None,
    ) -> None:
        mlflow.log_metric(name, value, step=step)


    def log_ensemble_score(self, score: float) -> None:
        """Log the final ensemble weighted R² score."""
        mlflow.log_metric("ensemble_test_r2", score)
        logger.info(f"Logged ensemble R²: {score:.6f}")

    def log_final_score(
        self,
        score     : float,
        split     : str = "test",
    ) -> None:
        """
        Log the final evaluation score.

        Args:
            score : Weighted R² score.
            split : Which split this score is from ("val" or "test").
        """
        mlflow.log_metric(f"final_{split}_r2", score)
        logger.info(f"Logged final {split} R²: {score:.6f}")


    def log_training_history(
        self,
        history : Dict,
    ) -> None:
        """
        Log full training history — loss and score per epoch.

        Args:
            history : Dict with keys "train_loss" and "val_score"
                      Each value is a list of floats (one per epoch)
        """
        for epoch, train_loss in enumerate(history["train_loss"]):
            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)

        for epoch, val_score in enumerate(history["val_score"]):
            mlflow.log_metric("val_score", val_score, step=epoch + 1)

        # Log best val score as a summary metric
        best_val   = max(history["val_score"])
        best_epoch = history["val_score"].index(best_val) + 1
        mlflow.log_metric("best_val_score", best_val)
        mlflow.log_metric("best_epoch", best_epoch)

        logger.info(f"Logged {len(history['train_loss'])} epochs of history")
        logger.info(f"Best val score: {best_val:.6f} at epoch {best_epoch}")
