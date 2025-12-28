"""
Integrated Training Script with MLflow and TensorBoard

Fully implements experiment tracking, monitoring, and visualization.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import monitoring tools
try:
    from src.mlflow_tracker import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available")

try:
    from src.tensorboard_logger import TensorBoardLogger
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available")


def train_with_full_monitoring(
    dataset_path: str,
    output_dir: str = '/workspace/models',
    version: str = 'v1',
    model_type: str = 'xgboost',
    use_mlflow: bool = True,
    use_tensorboard: bool = True,
    **hyperparameters
):
    """
    Train model with full MLflow and TensorBoard monitoring.
    
    Args:
        dataset_path: Path to training dataset
        output_dir: Directory for saving models
        version: Model version
        model_type: Type of model (xgboost, lightgbm, random_forest)
        use_mlflow: Enable MLflow tracking
        use_tensorboard: Enable TensorBoard logging
        **hyperparameters: Model hyperparameters
        
    Returns:
        Dictionary with training results
    """
    logger.info("=" * 80)
    logger.info("TRAINING WITH FULL MONITORING")
    logger.info("=" * 80)
    
    # Initialize monitoring
    mlflow_tracker = None
    tb_logger = None
    
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow_tracker = MLflowTracker(
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
            experiment_name='connect4-policy-imitation'
        )
        run = mlflow_tracker.start_run(
            run_name=f'training-{version}',
            tags={
                'model_type': model_type,
                'version': version
            }
        )
        logger.info(f"MLflow run started: {run.info.run_id}")
    
    if use_tensorboard and TENSORBOARD_AVAILABLE:
        tb_logger = TensorBoardLogger(
            log_dir=os.getenv('TENSORBOARD_LOG_DIR', '/workspace/tensorboard-logs'),
            experiment_name='connect4-ml'
        )
        logger.info(f"TensorBoard logs: {tb_logger.log_dir}")
    
    # 1. LOAD AND PREPROCESS DATA
    logger.info("Loading and preprocessing data...")
    from src.preprocessing import Connect4Preprocessor
    from src.eda import Connect4EDA
    
    # Run EDA first
    if use_tensorboard or use_mlflow:
        logger.info("Running Exploratory Data Analysis...")
        eda = Connect4EDA(output_dir=f"{output_dir}/{version}/eda")
        df_full = pd.read_parquet(dataset_path)
        eda_report = eda.generate_full_report(df_full, save_plots=True)
        logger.info(f"EDA complete - reports saved to: {output_dir}/{version}/eda/")
    
    # Preprocess
    preprocessor = Connect4Preprocessor()
    data = preprocessor.preprocess_pipeline(
        dataset_path=dataset_path,
        output_dir=f"{output_dir}/{version}/preprocessing"
    )
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Log dataset info
    df = pd.read_parquet(dataset_path)
    if mlflow_tracker:
        mlflow_tracker.log_dataset_info(dataset_path, df)
        mlflow_tracker.log_split_info(
            train_games=len(np.unique(df.iloc[:len(y_train)]['gameId'])) if 'gameId' in df.columns else 0,
            val_games=len(np.unique(df.iloc[len(y_train):len(y_train)+len(y_val)]['gameId'])) if 'gameId' in df.columns else 0,
            test_games=len(np.unique(df.iloc[-len(y_test):]['gameId'])) if 'gameId' in df.columns else 0,
            train_moves=len(y_train),
            val_moves=len(y_val),
            test_moves=len(y_test)
        )
    
    # 2. CREATE AND TRAIN MODEL
    logger.info(f"Training {model_type} model...")
    
    # Set hyperparameters
    default_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    default_params.update(hyperparameters)
    
    # Log hyperparameters
    if mlflow_tracker:
        mlflow_tracker.log_hyperparameters(default_params)
    
    if tb_logger:
        tb_logger.log_hyperparameters(
            hparams=default_params,
            metrics={'dataset_size': len(y_train)}
        )
    
    # Train model
    import xgboost as xgb
    
    model = xgb.XGBClassifier(
        n_estimators=default_params['n_estimators'],
        max_depth=default_params['max_depth'],
        learning_rate=default_params['learning_rate'],
        subsample=default_params['subsample'],
        colsample_bytree=default_params['colsample_bytree'],
        random_state=default_params['random_state'],
        eval_metric='mlogloss',
        early_stopping_rounds=20
    )
    
    # Train with validation
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Get training history
    results = model.evals_result()
    train_losses = results['validation_0']['mlogloss']
    val_losses = results['validation_1']['mlogloss']
    
    # Log training curves
    if tb_logger:
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            tb_logger.log_training_step(
                {'loss': train_loss},
                epoch,
                phase='train'
            )
            tb_logger.log_training_step(
                {'loss': val_loss},
                epoch,
                phase='val'
            )
    
    # 3. EVALUATE MODEL
    logger.info("Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'f1_macro': f1_score(y_train, y_train_pred, average='macro'),
        'f1_weighted': f1_score(y_train, y_train_pred, average='weighted')
    }
    
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1_macro': f1_score(y_val, y_val_pred, average='macro'),
        'f1_weighted': f1_score(y_val, y_val_pred, average='weighted')
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'f1_macro': f1_score(y_test, y_test_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_test_pred, average='weighted')
    }
    
    # Top-k accuracy
    y_test_proba = model.predict_proba(X_test)
    top_3_acc = np.mean([y_test[i] in np.argsort(y_test_proba[i])[-3:] for i in range(len(y_test))])
    top_5_acc = np.mean([y_test[i] in np.argsort(y_test_proba[i])[-5:] for i in range(len(y_test))])
    
    test_metrics['top_3_accuracy'] = top_3_acc
    test_metrics['top_5_accuracy'] = top_5_acc
    
    # Log metrics
    if mlflow_tracker:
        mlflow_tracker.log_training_metrics(train_metrics, val_metrics)
        mlflow_tracker.log_test_metrics(test_metrics)
    
    if tb_logger:
        tb_logger.log_training_step(train_metrics, 0, 'train')
        tb_logger.log_training_step(val_metrics, 0, 'val')
        tb_logger.log_test_metrics(test_metrics)
    
    # Per-column performance
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_test_pred,
        labels=range(7),
        zero_division=0
    )
    
    per_column_metrics = {}
    for col in range(7):
        per_column_metrics[col] = {
            'precision': float(precision[col]),
            'recall': float(recall[col]),
            'f1': float(f1[col])
        }
    
    if mlflow_tracker:
        mlflow_tracker.log_per_column_performance(per_column_metrics)
    
    if tb_logger:
        tb_logger.log_per_column_performance(per_column_metrics)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    if mlflow_tracker:
        mlflow_tracker.log_confusion_matrix(cm)
    
    if tb_logger:
        tb_logger.log_confusion_matrix(cm)
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = data['feature_names']
    
    if mlflow_tracker:
        mlflow_tracker.log_feature_importance(feature_names, feature_importance)
    
    if tb_logger:
        tb_logger.log_feature_importance(feature_names, feature_importance)
    
    # 4. SAVE MODEL
    logger.info("Saving model...")
    
    output_path = Path(output_dir) / version / model_type
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / f'{model_type}_model_{version}.joblib'
    import joblib
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = output_path / f'metrics_{version}.json'
    import json
    with open(metrics_path, 'w') as f:
        json.dump({
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'per_column': per_column_metrics
        }, f, indent=2)
    
    # Log model to MLflow
    if mlflow_tracker:
        mlflow_tracker.log_model(
            model,
            model_type='xgboost',
            registered_model_name='connect4-policy-model'
        )
        mlflow_tracker.log_preprocessor(str(output_path.parent / 'preprocessing' / 'preprocessor.joblib'))
    
    # 5. PRINT RESULTS
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"Top-3 Accuracy: {test_metrics['top_3_accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy: {test_metrics['top_5_accuracy']:.4f}")
    logger.info("=" * 80)
    
    # Close loggers
    if tb_logger:
        tb_logger.close()
    
    return {
        'model': model,
        'model_path': str(model_path),
        'metrics': test_metrics,
        'per_column_metrics': per_column_metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Connect4 ML model with full monitoring')
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--output-dir', default='/workspace/models', help='Output directory')
    parser.add_argument('--version', default='v1', help='Model version')
    parser.add_argument('--model-type', default='xgboost', help='Model type')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard')
    
    args = parser.parse_args()
    
    train_with_full_monitoring(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        version=args.version,
        model_type=args.model_type,
        use_mlflow=not args.no_mlflow,
        use_tensorboard=not args.no_tensorboard
    )
