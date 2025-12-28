"""
Train Connect4 Policy Imitation Model

Train XGBoost/LightGBM/RandomForest to predict MCTS moves from game state.
Includes MLflow tracking for experiments.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import logging
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import Connect4Preprocessor

# MLflow integration
try:
    from mlflow_tracker import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available - experiment tracking disabled")

logger = logging.getLogger(__name__)


class PolicyImitationTrainer:
    """Train policy imitation models for Connect4"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Args:
            model_type: 'xgboost', 'lightgbm', or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        
    def create_model(self, **kwargs):
        """Create model based on type"""
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                objective='multi:softmax',
                num_class=7,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.1),
                num_leaves=kwargs.get('num_leaves', 31),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 12),
                min_samples_split=kwargs.get('min_samples_split', 10),
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **model_params
    ):
        """Train the model"""
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Train samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        # Create model
        self.model = self.create_model(**model_params)
        
        # Train with early stopping if validation set provided
        if X_val is not None and self.model_type in ['xgboost', 'lightgbm']:
            logger.info("Training with early stopping...")
            
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:  # lightgbm
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                )
        else:
            self.model.fit(X_train, y_train)
        
        logger.info("Training complete!")
        
        return self.model
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        top_k: int = 3
    ) -> dict:
        """
        Evaluate model performance.
        
        Metrics:
        - Accuracy
        - F1 (macro and weighted)
        - Top-k accuracy
        - Confusion matrix
        - Per-class metrics
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Top-k accuracy
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -top_k:]
        top_k_acc = np.mean([y_test[i] in top_k_preds[i] for i in range(len(y_test))])
        
        # Top-5 accuracy
        top_5_preds = np.argsort(y_pred_proba, axis=1)[:, -5:]
        top_5_acc = np.mean([y_test[i] in top_5_preds[i] for i in range(len(y_test))])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'top_3_accuracy': top_k_acc,
            'top_5_accuracy': top_5_acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Macro: {f1_macro:.4f}")
        logger.info(f"F1 Weighted: {f1_weighted:.4f}")
        logger.info(f"Top-3 Accuracy: {top_k_acc:.4f}")
        logger.info(f"Top-5 Accuracy: {top_5_acc:.4f}")
        
        return self.metrics
    
    def save_model(self, output_dir: str, version: str = None):
        """Save model and metrics"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_str = f"_{version}" if version else f"_{timestamp}"
        model_filename = f"{self.model_type}_model{version_str}.joblib"
        
        # Save model
        model_path = output_path / model_filename
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metrics
        metrics_path = output_path / f"metrics{version_str}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        return model_path, metrics_path


def train_connect4_model(
    dataset_path: str,
    model_type: str = 'xgboost',
    output_dir: str = 'models',
    version: str = None,
    **model_params
):
    """
    Complete training pipeline.
    
    Args:
        dataset_path: Path to parquet dataset
        model_type: 'xgboost', 'lightgbm', or 'random_forest'
        output_dir: Where to save models
        version: Model version string
        **model_params: Hyperparameters for model
    """
    logger.info("=" * 80)
    logger.info("CONNECT4 POLICY IMITATION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Output dir: {output_dir}")
    
    # Preprocess data
    preprocessor = Connect4Preprocessor()
    data = preprocessor.preprocess_pipeline(
        dataset_path=dataset_path,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        output_dir=f"{output_dir}/preprocessing"
    )
    
    # Train model
    trainer = PolicyImitationTrainer(model_type=model_type)
    trainer.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        **model_params
    )
    
    # Evaluate
    metrics = trainer.evaluate(
        X_test=data['X_test'],
        y_test=data['y_test']
    )
    
    # Save
    model_path, metrics_path = trainer.save_model(
        output_dir=f"{output_dir}/{model_type}",
        version=version
    )
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE ✅")
    logger.info("=" * 80)
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Metrics saved: {metrics_path}")
    
    return {
        'model': trainer.model,
        'metrics': metrics,
        'preprocessor': preprocessor,
        'model_path': model_path,
        'metrics_path': metrics_path
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train XGBoost model on dataset_v1
    results = train_connect4_model(
        dataset_path='/mnt/user-data/uploads/dataset_v1.parquet',
        model_type='xgboost',
        output_dir='models/v1',
        version='v1',
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Macro: {results['metrics']['f1_macro']:.4f}")
    print(f"Top-3 Accuracy: {results['metrics']['top_3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['metrics']['top_5_accuracy']:.4f}")
    print()
    print("Per-column performance:")
    report = results['metrics']['classification_report']
    for col in range(7):
        col_str = str(col)
        if col_str in report:
            print(f"  Column {col}: F1={report[col_str]['f1-score']:.3f}, "
                  f"Precision={report[col_str]['precision']:.3f}, "
                  f"Recall={report[col_str]['recall']:.3f}")
