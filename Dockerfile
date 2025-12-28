FROM python:3.11-slim

LABEL maintainer="your-team@example.com"
LABEL service="connect4-ml-trainer"
LABEL version="1.0.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY src/ ./src/
COPY train_integrated.py .
COPY continuous_learning.py .
COPY learning_config.yaml .

# Data directories (mounted as volumes)
RUN mkdir -p /workspace/datasets \
    /workspace/models \
    /workspace/logs \
    /workspace/tensorboard-logs \
    /workspace/mlflow-artifacts

VOLUME /workspace/datasets
VOLUME /workspace/models
VOLUME /workspace/logs

# Default command: continuous learning
CMD ["python", "continuous_learning.py", "--config", "learning_config.yaml"]
