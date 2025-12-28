# Connect4 ML Trainer

**Training and continuous learning microservice for Connect4 ML models.**

## 🎯 Purpose

Handles all ML training operations:

- **Model Training:** Train XGBoost models on gameplay data
- **Continuous Learning:** Automated retraining on new data
- **Experiment Tracking:** MLflow integration
- **Performance Monitoring:** TensorBoard integration

## 🚀 Quick Start

### Manual Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_integrated.py \
  --dataset /path/to/dataset.parquet \
  --output-dir ./models \
  --version v2

# Output:
# - models/v2/xgboost/model.joblib
# - models/v2/preprocessing/preprocessor.joblib
# - models/v2/metrics.json
```

### Continuous Learning

```bash
# Run continuous learning loop
python continuous_learning.py --config learning_config.yaml

# Or run once
python continuous_learning.py --once
```

### Docker

```bash
# Build
docker build -t connect4-ml-trainer:latest .

# Run training once
docker run --rm \
  -v /path/to/datasets:/workspace/datasets:ro \
  -v /path/to/models:/workspace/models \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  connect4-ml-trainer:latest \
  python train_integrated.py --dataset /workspace/datasets/dataset_v1.parquet

# Run continuous learning
docker run -d \
  -v /path/to/datasets:/workspace/datasets:ro \
  -v /path/to/models:/workspace/models \
  -e LOGGER_API_URL=http://logger:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  connect4-ml-trainer:latest
```

## 📊 Features

### 1. Game-Level Data Splitting
- Prevents data leakage
- Ensures true model generalization
- Validates no overlap between train/val/test

### 2. MLflow Integration
- Automatic experiment tracking
- Hyperparameter logging
- Metrics logging (train/val/test)
- Model versioning
- Artifact storage

### 3. TensorBoard Integration
- Training curves visualization
- Confusion matrices
- Feature importance plots
- Real-time monitoring

### 4. Continuous Learning
- Automated dataset export from logger
- Periodic retraining (configurable)
- Model evaluation before deployment
- Automatic deployment (optional)

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGGER_API_URL` | `http://localhost:8010` | Logger service URL |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `DATASET_DIR` | `/workspace/datasets` | Dataset directory |
| `MODEL_DIR` | `/workspace/models` | Model output directory |
| `LOG_DIR` | `/workspace/logs` | Logs directory |
| `TENSORBOARD_LOG_DIR` | `/workspace/tensorboard-logs` | TensorBoard logs |
| `MODEL_TYPE` | `xgboost` | Model type |
| `INTERVAL_HOURS` | `24` | Retraining interval |
| `AUTO_DEPLOY` | `false` | Auto-deploy if improved |
| `MIN_ACCURACY_IMPROVEMENT` | `0.01` | Min improvement to deploy |

### learning_config.yaml

```yaml
game_generation:
  games_per_iteration: 100

dataset_export:
  logger_api_url: http://connect4_logger:8000
  dataset_dir: /workspace/datasets
  min_new_games: 50

model_training:
  model_type: xgboost
  test_size: 0.2
  val_size: 0.1
  models_dir: /workspace/models
  min_accuracy_improvement: 0.01

deployment:
  auto_deploy: false
  backup_previous: true

scheduling:
  interval_hours: 24
  max_iterations: null

monitoring:
  log_dir: /workspace/logs
  metrics_file: learning_history.json
```

## 📈 Workflow

### Continuous Learning Loop

```
1. Export new dataset from logger
   ↓
2. Load dataset (game-level split)
   ↓
3. Train XGBoost model
   ↓
4. Evaluate on test set
   ↓
5. Log to MLflow & TensorBoard
   ↓
6. Compare with baseline
   ↓
7. Deploy if improved (optional)
   ↓
8. Wait interval_hours
   ↓
9. Repeat
```

### Training Metrics

**Logged to MLflow:**
- Hyperparameters
- Train/val/test accuracy
- F1 scores (macro, weighted, per-class)
- Top-k accuracy
- Confusion matrix
- Feature importance

**Logged to TensorBoard:**
- Training curves (loss, accuracy)
- Validation curves
- Confusion matrix heatmap
- Feature importance charts
- Learning curves

## 🧪 Testing

```bash
# Unit tests
pytest tests/

# Integration test - train on sample data
python train_integrated.py \
  --dataset tests/fixtures/sample_dataset.parquet \
  --output-dir ./test_output \
  --no-mlflow \
  --no-tensorboard
```

## 📊 Expected Performance

**Training Time:**
- 100 games: ~3 seconds
- 500 games: ~10 seconds
- 1000 games: ~20 seconds

**Model Performance:**
- 100 games: 94.15% accuracy
- 500 games: ~96% accuracy
- 1000 games: ~97% accuracy

## 🔗 Dependencies

**Required Services:**
- Logger API (for dataset export)
- MLflow server (for experiment tracking)
- TensorBoard (for visualization)
- PostgreSQL (optional, for direct DB access)

## 📦 Deployment

### Production Recommendations

- Run as scheduled job (cron/k8s CronJob)
- Use read-only mounts for datasets
- Store models in persistent volume
- Configure resource limits
- Monitor training metrics
- Set up alerts for failures

### Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: connect4-ml-trainer
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: your-registry/connect4-ml-trainer:latest
            args: ["python", "continuous_learning.py", "--once"]
            env:
            - name: LOGGER_API_URL
              value: http://connect4-logger:8000
            - name: MLFLOW_TRACKING_URI
              value: http://mlflow:5000
            volumeMounts:
            - name: datasets
              mountPath: /workspace/datasets
              readOnly: true
            - name: models
              mountPath: /workspace/models
```

## 📝 License

MIT
