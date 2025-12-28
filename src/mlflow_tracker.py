"""
MLflow Integration for Connect4 ML System

Handles experiment tracking, model versioning, and metrics logging.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Manages MLflow experiment tracking for Connect4 ML models.
    
    Features:
    - Experiment tracking
    - Model versioning
    - Metrics logging
    - Hyperparameter tracking
    - Model comparison
    """
    
    def __init__(
        self,
        tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = "connect4-policy-imitation",
        artifact_location: str = "/workspace/mlflow-artifacts"
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            artifact_location: Where to store artifacts
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        self.experiment = self._get_or_create_experiment()
        self.client = MlflowClient(tracking_uri)
        
        logger.info(f"MLflow tracker initialized: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name} (ID: {self.experiment.experiment_id})")
    
    def _get_or_create_experiment(self):
        """Get existing experiment or create new one"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location
            )
            experiment = mlflow.get_experiment(experiment_id)
            logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            logger.info(f"Using existing experiment: {self.experiment_name}")
        
        return experiment
    
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Additional tags
            
        Returns:
            Active MLflow run
        """
        default_tags = {
            "model_type": "policy-imitation",
            "timestamp": datetime.now().isoformat()
        }
        
        if tags:
            default_tags.update(tags)
        
        run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=run_name,
            tags=default_tags
        )
        
        logger.info(f"Started run: {run_name} (ID: {run.info.run_id})")
        return run
    
    def log_dataset_info(self, dataset_path: str, df: pd.DataFrame):
        """Log dataset information"""
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("num_features", len(df.columns))
        mlflow.log_param("num_games", df['gameId'].nunique() if 'gameId' in df.columns else 'unknown')
        
        # Log target distribution
        if 'action_taken' in df.columns:
            target_dist = df['action_taken'].value_counts().to_dict()
            mlflow.log_dict(target_dist, "target_distribution.json")
    
    def log_split_info(
        self,
        train_games: int,
        val_games: int,
        test_games: int,
        train_moves: int,
        val_moves: int,
        test_moves: int
    ):
        """Log train/val/test split information"""
        mlflow.log_param("train_games", train_games)
        mlflow.log_param("val_games", val_games)
        mlflow.log_param("test_games", test_games)
        mlflow.log_param("train_moves", train_moves)
        mlflow.log_param("val_moves", val_moves)
        mlflow.log_param("test_moves", test_moves)
        
        # Log split ratios
        total_games = train_games + val_games + test_games
        mlflow.log_param("train_ratio", train_games / total_games)
        mlflow.log_param("val_ratio", val_games / total_games)
        mlflow.log_param("test_ratio", test_games / total_games)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log model hyperparameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        logger.info(f"Logged {len(params)} hyperparameters")
    
    def log_training_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: Optional[int] = None
    ):
        """
        Log training and validation metrics.
        
        Args:
            train_metrics: Training metrics (accuracy, loss, etc.)
            val_metrics: Validation metrics
            epoch: Epoch number (for iterative training)
        """
        # Log training metrics
        for key, value in train_metrics.items():
            metric_name = f"train_{key}"
            if epoch is not None:
                mlflow.log_metric(metric_name, value, step=epoch)
            else:
                mlflow.log_metric(metric_name, value)
        
        # Log validation metrics
        for key, value in val_metrics.items():
            metric_name = f"val_{key}"
            if epoch is not None:
                mlflow.log_metric(metric_name, value, step=epoch)
            else:
                mlflow.log_metric(metric_name, value)
    
    def log_test_metrics(self, metrics: Dict[str, float]):
        """Log final test metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(f"test_{key}", value)
        
        # Log metrics to JSON file
        mlflow.log_dict(metrics, "test_metrics.json")
        
        logger.info(f"Logged test metrics: {metrics}")
    
    def log_per_column_performance(self, per_column_metrics: Dict[int, Dict[str, float]]):
        """Log per-column performance metrics"""
        # Log individual column metrics
        for column, metrics in per_column_metrics.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"column_{column}_{metric_name}", value)
        
        # Save to JSON
        mlflow.log_dict(per_column_metrics, "per_column_metrics.json")
    
    def log_confusion_matrix(self, confusion_matrix: np.ndarray):
        """Log confusion matrix as artifact"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(7),
            yticklabels=range(7),
            ax=ax
        )
        ax.set_xlabel('Predicted Column')
        ax.set_ylabel('True Column')
        ax.set_title('Confusion Matrix - Move Predictions')
        
        # Save and log
        fig.savefig('/tmp/confusion_matrix.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('/tmp/confusion_matrix.png')
        plt.close()
        
        logger.info("Logged confusion matrix")
    
    def log_feature_importance(
        self,
        feature_names: list,
        importances: np.ndarray,
        top_n: int = 20
    ):
        """Log feature importance plot"""
        import matplotlib.pyplot as plt
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(
            range(top_n),
            importances[indices],
            color='steelblue'
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.invert_yaxis()
        
        # Save and log
        fig.savefig('/tmp/feature_importance.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('/tmp/feature_importance.png')
        plt.close()
        
        # Also log as JSON
        importance_dict = {
            feature_names[i]: float(importances[i])
            for i in indices
        }
        mlflow.log_dict(importance_dict, "top_features.json")
        
        logger.info("Logged feature importance")
    
    def log_model(
        self,
        model,
        model_type: str = "xgboost",
        registered_model_name: Optional[str] = None
    ):
        """
        Log trained model to MLflow.
        
        Args:
            model: Trained model
            model_type: Type of model (xgboost, sklearn, etc.)
            registered_model_name: Name for model registry
        """
        if model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name=registered_model_name
            )
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=registered_model_name
            )
        else:
            # Generic Python model
            mlflow.pyfunc.log_model(
                "model",
                python_model=model,
                registered_model_name=registered_model_name
            )
        
        logger.info(f"Logged {model_type} model")
    
    def log_preprocessor(self, preprocessor_path: str):
        """Log preprocessing artifacts"""
        mlflow.log_artifact(preprocessor_path, "preprocessing")
        logger.info("Logged preprocessor")
    
    def log_learning_curves(
        self,
        train_scores: list,
        val_scores: list,
        metric_name: str = "accuracy"
    ):
        """Plot and log learning curves"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_scores) + 1)
        
        ax.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}', linewidth=2)
        ax.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'Learning Curves - {metric_name.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save and log
        fig.savefig('/tmp/learning_curves.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('/tmp/learning_curves.png')
        plt.close()
        
        logger.info("Logged learning curves")
    
    def compare_models(self, run_ids: list) -> pd.DataFrame:
        """
        Compare multiple model runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with comparison
        """
        runs_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            
            run_data = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unknown'),
                'start_time': run.info.start_time,
                **run.data.params,
                **run.data.metrics
            }
            runs_data.append(run_data)
        
        return pd.DataFrame(runs_data)
    
    def get_best_run(self, metric: str = "test_accuracy") -> Dict[str, Any]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize
            
        Returns:
            Best run info
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            logger.info(f"Best run: {best_run.info.run_id} ({metric}={best_run.data.metrics.get(metric)})")
            return {
                'run_id': best_run.info.run_id,
                'run_name': best_run.data.tags.get('mlflow.runName'),
                'metrics': best_run.data.metrics,
                'params': best_run.data.params
            }
        
        return None
    
    def log_continuous_learning_iteration(
        self,
        iteration: int,
        metrics: Dict[str, float],
        deployed: bool
    ):
        """Log continuous learning iteration metrics"""
        # Use iteration as step
        for key, value in metrics.items():
            mlflow.log_metric(f"continuous_{key}", value, step=iteration)
        
        mlflow.log_metric("deployed", 1 if deployed else 0, step=iteration)
        
        logger.info(f"Logged iteration {iteration} metrics")


def create_mlflow_tracker(config: Dict[str, Any]) -> MLflowTracker:
    """
    Create MLflow tracker from configuration.
    
    Args:
        config: Configuration dict
        
    Returns:
        Initialized MLflow tracker
    """
    return MLflowTracker(
        tracking_uri=config.get('tracking_uri', 'http://mlflow:5000'),
        experiment_name=config.get('experiment_name', 'connect4-policy-imitation'),
        artifact_location=config.get('artifact_location', '/workspace/mlflow-artifacts')
    )
