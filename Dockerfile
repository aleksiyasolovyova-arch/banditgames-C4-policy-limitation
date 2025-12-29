FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ADDED git here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Source Code
COPY src/ /app/src/

# Copy the Orchestrator script to the root
COPY continuous_learning.py .

# Create directory structures for Docker Volumes
RUN mkdir -p /workspace/datasets /workspace/models /workspace/tensorboard-logs

# Default command: Run the watcher
CMD ["python", "continuous_learning.py"]
