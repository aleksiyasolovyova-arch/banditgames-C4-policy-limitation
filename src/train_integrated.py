import os
import joblib
import logging
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

from src.preprocessing import Connect4Preprocessor
from src.eda import Connect4EDA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_job(dataset_path: str, output_dir: str, version: str):
    logger.info(f" Starting training job for {version}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 0. Load Raw Data for EDA
    try:
        df_raw = pd.read_parquet(dataset_path)
        eda_output = out_path / "reports"
        eda = Connect4EDA(output_dir=str(eda_output))
        eda.generate_report(df_raw, version)
    except Exception as e:
        logger.warning(f"EDA Generation failed (skipping): {e}")

    # 1. Preprocess
    preprocessor = Connect4Preprocessor()
    try:
        data = preprocessor.preprocess_pipeline(dataset_path)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None

    # 2. Train (always)
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "num_class": 7,
    }

    logger.info("Training XGBoost model...")
    model = XGBClassifier(**params)
    model.fit(data["X_train"], data["y_train"])

    preds = model.predict(data["X_test"])
    acc = accuracy_score(data["y_test"], preds)
    logger.info(f" Accuracy: {acc:.4f}")

    # 3. Save Artifacts locally (always)
    joblib.dump(model, out_path / f"model_{version}.joblib")
    joblib.dump(preprocessor, out_path / f"preprocessor_{version}.joblib")
    logger.info(f" Model + preprocessor saved to {out_path}")

    # 4. Log to MLflow (best-effort)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "connect4_continuous_learning")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=version):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

        logger.info(f" Logged run to MLflow  {tracking_uri} (experiment: {experiment_name})")
    except Exception as e:
        logger.warning(f" MLflow logging skipped (server not reachable / blocked): {e}")

    return acc
