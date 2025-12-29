import time
import requests
import logging
import os
from pathlib import Path
from src.train_integrated import train_job

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Watcher")

# These paths must match the VOLUME mounts in docker-compose.yml
DATA_DIR = Path("/workspace/datasets")
MODELS_DIR = Path("/workspace/models")
API_URL = "http://connect4_ml_api:8001/deploy"


def main():
    logger.info(f" Watching {DATA_DIR} for new parquet files...")
    processed_files = set()

    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            # 1. Scan for files (sorted to handle v1 before v2)
            files = sorted(list(DATA_DIR.glob("*.parquet")))

            for file_path in files:
                if file_path.name in processed_files:
                    continue

                logger.info(f" Detected new dataset: {file_path.name}")

                # Extract version from filename (dataset_v1.parquet -> v1)
                try:
                    version = file_path.stem.split("_")[-1]
                except:
                    version = str(int(time.time()))

                # 2. Train (Includes EDA and Preprocessing)
                acc = train_job(str(file_path), str(MODELS_DIR), version)

                if acc is not None:
                    # 3. Signal API to Deploy
                    try:
                        requests.post(API_URL, json={"version": version}, timeout=5)
                        logger.info(f" Deployment signal sent for {version}")
                    except Exception as e:
                        logger.error(f" API Deployment failed (Is the API running?): {e}")

                processed_files.add(file_path.name)

            # Wait before next scan
            time.sleep(10)

        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()