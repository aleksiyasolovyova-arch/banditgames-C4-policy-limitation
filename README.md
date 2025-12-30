
# Connect4 Policy Imitation Model

This directory contains the policy imitation learning pipeline for Connect4.
The goal of this model is to learn move-selection behavior from historical gameplay data by imitating the actions taken in each board state.

The trained model predicts a probability distribution over the 7 possible columns, representing how likely each move is given the current board.

---

## Purpose

The policy imitation model is responsible for:
- Learning move preferences from existing games
- Predicting the next move as a probability distribution
- Serving as a fast, interpretable baseline policy
- Providing move proposals for downstream systems

This model does not reason about long-term outcomes directly.
It imitates observed behavior rather than optimizing for win probability.

---

## Target Definition

The training target is the column chosen by the player.

- Action space: integers 0–6
- One class per column
- Multiclass classification problem with 7 outputs

---

## Exploratory Data Analysis (EDA)

EDA is performed before training to validate data quality and behavioral patterns.

### Action Distribution
- Plots how often each column is selected
- Helps detect dataset imbalance or bias toward specific columns

### Board Occupancy Heatmap
- Visualizes average board usage across all positions
- Player pieces and opponent pieces are mapped to opposite signs
- Highlights strategic tendencies such as center-column preference

EDA artifacts are saved per dataset version:
```
models/
└── reports/
    ├── action_dist_<version>.png
    └── heatmap_<version>.png
```

---

## Preprocessing

### Board Encoding

Board states are represented as 42 features (6 rows × 7 columns):

- Empty: 0
- Current player: 1
- Opponent: 2

Boards may be provided as:
- A single string representation, or
- 42 flattened columns (board_before_r0c0 … board_before_r5c6)

All formats are normalized into a consistent 42-feature layout.

---

### Leakage Prevention

To avoid information leakage:
- Data is split by gameId, not by individual moves
- Entire games are assigned to either training or test sets

Default split:
- 80 percent training games
- 20 percent test games

If gameId is missing, a warning is logged and a fallback split is used.

---

## Feature Scaling

No feature scaling is applied.

Board encodings are discrete and tree-based models handle them naturally without normalization.

---

## Model Training

### Algorithm
- XGBoost Classifier
- Objective: multi:softprob
- Number of classes: 7

### Training Objective
The model learns to approximate:

P(action | board_state)

The output is a probability distribution over all columns.

---

## Evaluation

### Accuracy

Evaluation uses top-1 accuracy:
- Checks whether the most likely predicted column matches the true action

Accuracy is sufficient here because:
- The task is pure imitation
- Outputs are used to rank moves, not to estimate uncertainty

---

## Artifacts

Each training run produces:
```
models/
├── model_<version>.joblib
├── preprocessor_<version>.joblib
└── reports/
    ├── action_dist_<version>.png
    └── heatmap_<version>.png
```

---

## Experiment Tracking

MLflow logging is supported on a best-effort basis:
- Logs model parameters
- Logs accuracy
- Stores the trained model artifact

Training continues even if MLflow is unavailable.

---

## Automated Retraining

A watcher process continuously monitors a dataset directory:
- Detects new parquet files
- Triggers training automatically
- Saves versioned artifacts
- Notifies the inference API to deploy the new policy model

This enables continuous learning as new gameplay data becomes available.

---
The policy imitation model provides:
- Fast and stable move predictions
- Interpretable behavior learned from real games
- A reliable baseline policy for Connect4

It is designed to be simple, reproducible, and safe to deploy in automated pipelines.
