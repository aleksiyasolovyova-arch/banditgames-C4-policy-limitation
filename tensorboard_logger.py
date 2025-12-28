"""
TensorBoard Integration for Connect4 ML System

Real-time monitoring of:
- Training metrics
- AI player decisions
- Match outcomes
- Predictions
- Player comparisons
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    import torch
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard not available - install with: pip install tensorboard torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    TensorBoard logger for Connect4 ML system.
    
    Logs:
    - Training metrics (accuracy, loss, F1)
    - Model predictions
    - Player decisions
    - Game outcomes
    - Performance comparisons
    """
    
    def __init__(
        self,
        log_dir: str = "/workspace/tensorboard-logs",
        experiment_name: str = "connect4-ml"
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of the experiment
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available")
        
        self.log_dir = Path(log_dir) / experiment_name / datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create writers for different aspects
        self.train_writer = SummaryWriter(self.log_dir / 'train')
        self.val_writer = SummaryWriter(self.log_dir / 'val')
        self.test_writer = SummaryWriter(self.log_dir / 'test')
        self.game_writer = SummaryWriter(self.log_dir / 'games')
        self.comparison_writer = SummaryWriter(self.log_dir / 'comparison')
        
        logger.info(f"TensorBoard logs: {self.log_dir}")
        logger.info(f"Start TensorBoard: tensorboard --logdir {self.log_dir.parent.parent}")
    
    def log_training_step(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ):
        """
        Log training/validation metrics.
        
        Args:
            metrics: Dictionary of metrics (accuracy, loss, f1, etc.)
            step: Training step/epoch
            phase: 'train' or 'val'
        """
        writer = self.train_writer if phase == 'train' else self.val_writer
        
        for metric_name, value in metrics.items():
            writer.add_scalar(f'{metric_name}', value, step)
        
        logger.debug(f"Logged {phase} metrics at step {step}")
    
    def log_test_metrics(self, metrics: Dict[str, float]):
        """Log final test metrics"""
        for metric_name, value in metrics.items():
            self.test_writer.add_scalar(f'final/{metric_name}', value, 0)
    
    def log_per_column_performance(
        self,
        per_column_metrics: Dict[int, Dict[str, float]],
        step: int = 0
    ):
        """
        Log per-column performance metrics.
        
        Args:
            per_column_metrics: Metrics for each column (0-6)
            step: Step number
        """
        # Create bar chart data
        columns = sorted(per_column_metrics.keys())
        
        for metric_name in ['accuracy', 'f1', 'precision', 'recall']:
            values = [per_column_metrics[col].get(metric_name, 0) for col in columns]
            
            # Log as scalars
            for col, val in zip(columns, values):
                self.test_writer.add_scalar(f'column_{col}/{metric_name}', val, step)
    
    def log_confusion_matrix(
        self,
        cm: np.ndarray,
        step: int = 0,
        class_names: Optional[List[str]] = None
    ):
        """
        Log confusion matrix as heatmap.
        
        Args:
            cm: Confusion matrix (7x7 for Connect4)
            step: Step number
            class_names: Names for classes (columns 0-6)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if class_names is None:
            class_names = [f'Col {i}' for i in range(7)]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_xlabel('Predicted Column')
        ax.set_ylabel('True Column')
        ax.set_title('Confusion Matrix - Move Predictions')
        
        # Log to TensorBoard
        self.test_writer.add_figure('confusion_matrix', fig, step)
        plt.close()
    
    def log_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        step: int = 0,
        top_n: int = 20
    ):
        """
        Log feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            step: Step number
            top_n: Number of top features to show
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Log as bar chart
        import matplotlib.pyplot as plt
        
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
        
        self.test_writer.add_figure('feature_importance', fig, step)
        plt.close()
        
        # Also log as scalars
        for i, idx in enumerate(indices):
            self.test_writer.add_scalar(
                f'feature_importance/{feature_names[idx]}',
                importances[idx],
                step
            )
    
    def log_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = 'accuracy'
    ):
        """
        Log learning curves.
        
        Args:
            train_scores: Training scores per epoch
            val_scores: Validation scores per epoch
            metric_name: Name of the metric
        """
        for epoch, (train_score, val_score) in enumerate(zip(train_scores, val_scores)):
            self.train_writer.add_scalar(f'learning_curve/{metric_name}', train_score, epoch)
            self.val_writer.add_scalar(f'learning_curve/{metric_name}', val_score, epoch)
    
    def log_game_outcome(
        self,
        game_id: str,
        winner: str,
        player1_type: str,
        player2_type: str,
        num_moves: int,
        duration_sec: float,
        step: int
    ):
        """
        Log individual game outcome.
        
        Args:
            game_id: Game identifier
            winner: 'player1', 'player2', or 'draw'
            player1_type: Type of player 1 (MCTS, ML, Human)
            player2_type: Type of player 2
            num_moves: Number of moves in game
            duration_sec: Game duration in seconds
            step: Global step counter
        """
        # Log outcome
        outcome_value = {'player1': 1, 'player2': -1, 'draw': 0}.get(winner, 0)
        self.game_writer.add_scalar('outcome/winner', outcome_value, step)
        
        # Log game stats
        self.game_writer.add_scalar('stats/num_moves', num_moves, step)
        self.game_writer.add_scalar('stats/duration_sec', duration_sec, step)
        
        # Log player matchup
        matchup = f"{player1_type}_vs_{player2_type}"
        self.game_writer.add_text(
            'matchup',
            f"Game {game_id}: {matchup} - Winner: {winner}",
            step
        )
    
    def log_move_decision(
        self,
        game_id: str,
        move_number: int,
        player_type: str,
        board_state: np.ndarray,
        chosen_column: int,
        all_probabilities: Optional[np.ndarray] = None,
        inference_time_ms: Optional[float] = None,
        step: int = 0
    ):
        """
        Log AI player move decision.
        
        Args:
            game_id: Game identifier
            move_number: Move number in game
            player_type: Type of player (MCTS, ML)
            board_state: Current board state (6x7)
            chosen_column: Column chosen by AI
            all_probabilities: Probabilities for all columns
            inference_time_ms: Time taken for decision
            step: Global step counter
        """
        # Log chosen column
        self.game_writer.add_scalar(
            f'decisions/{player_type}/chosen_column',
            chosen_column,
            step
        )
        
        # Log inference time
        if inference_time_ms:
            self.game_writer.add_scalar(
                f'decisions/{player_type}/inference_time_ms',
                inference_time_ms,
                step
            )
        
        # Log probability distribution
        if all_probabilities is not None:
            # Create bar chart
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(7), all_probabilities, color='steelblue', alpha=0.7)
            ax.bar(chosen_column, all_probabilities[chosen_column], color='red', alpha=0.9)
            ax.set_xlabel('Column')
            ax.set_ylabel('Probability')
            ax.set_title(f'{player_type} - Move {move_number} Decision')
            ax.set_xticks(range(7))
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            self.game_writer.add_figure(
                f'decisions/{player_type}/probabilities',
                fig,
                step
            )
            plt.close()
    
    def log_player_comparison(
        self,
        player_types: List[str],
        win_rates: List[float],
        avg_inference_times: List[float],
        avg_game_lengths: List[float],
        step: int = 0
    ):
        """
        Log comparison between different player types.
        
        Args:
            player_types: List of player type names
            win_rates: Win rate for each player type
            avg_inference_times: Average inference time (ms)
            avg_game_lengths: Average game length (moves)
            step: Step number
        """
        # Log win rates
        for player_type, win_rate in zip(player_types, win_rates):
            self.comparison_writer.add_scalar(
                f'win_rate/{player_type}',
                win_rate,
                step
            )
        
        # Log inference times
        for player_type, inf_time in zip(player_types, avg_inference_times):
            self.comparison_writer.add_scalar(
                f'inference_time/{player_type}',
                inf_time,
                step
            )
        
        # Create comparison charts
        import matplotlib.pyplot as plt
        
        # Win rate comparison
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(range(len(player_types)), win_rates, color='steelblue')
        ax1.set_xlabel('Player Type')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Win Rate Comparison')
        ax1.set_xticks(range(len(player_types)))
        ax1.set_xticklabels(player_types, rotation=45, ha='right')
        ax1.set_ylim(0, 100)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        self.comparison_writer.add_figure('comparison/win_rates', fig1, step)
        plt.close()
        
        # Speed comparison
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        colors = ['green' if 'ML' in pt else 'blue' for pt in player_types]
        ax2.bar(range(len(player_types)), avg_inference_times, color=colors)
        ax2.set_xlabel('Player Type')
        ax2.set_ylabel('Avg Inference Time (ms)')
        ax2.set_title('Speed Comparison')
        ax2.set_xticks(range(len(player_types)))
        ax2.set_xticklabels(player_types, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        self.comparison_writer.add_figure('comparison/speed', fig2, step)
        plt.close()
    
    def log_skill_by_game_phase(
        self,
        player_type: str,
        early_accuracy: float,
        mid_accuracy: float,
        late_accuracy: float,
        step: int = 0
    ):
        """
        Log AI skill variation by game phase.
        
        Args:
            player_type: Type of player
            early_accuracy: Accuracy in early game (moves 1-10)
            mid_accuracy: Accuracy in mid game (moves 11-25)
            late_accuracy: Accuracy in late game (moves 26+)
            step: Step number
        """
        phases = ['Early (1-10)', 'Mid (11-25)', 'Late (26+)']
        accuracies = [early_accuracy, mid_accuracy, late_accuracy]
        
        # Log as scalars
        for phase, acc in zip(phases, accuracies):
            self.comparison_writer.add_scalar(
                f'skill_by_phase/{player_type}/{phase}',
                acc,
                step
            )
        
        # Create chart
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(phases, accuracies, marker='o', linewidth=2, markersize=10, color='steelblue')
        ax.set_xlabel('Game Phase')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{player_type} - Skill by Game Phase')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        self.comparison_writer.add_figure(
            f'skill_by_phase/{player_type}',
            fig,
            step
        )
        plt.close()
    
    def log_move_distribution_heatmap(
        self,
        player_type: str,
        move_counts: np.ndarray,
        step: int = 0
    ):
        """
        Log move distribution heatmap (6x7 board).
        
        Args:
            player_type: Type of player
            move_counts: 6x7 array of move counts
            step: Step number
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            move_counts,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Move Count'},
            ax=ax
        )
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(f'{player_type} - Move Distribution Heatmap')
        
        self.comparison_writer.add_figure(
            f'move_distribution/{player_type}',
            fig,
            step
        )
        plt.close()
    
    def log_continuous_learning_iteration(
        self,
        iteration: int,
        accuracy: float,
        f1_score: float,
        model_deployed: bool,
        num_games_trained: int
    ):
        """
        Log continuous learning iteration.
        
        Args:
            iteration: Iteration number
            accuracy: Model accuracy
            f1_score: Model F1 score
            model_deployed: Whether model was deployed
            num_games_trained: Number of games in training set
        """
        self.train_writer.add_scalar('continuous/accuracy', accuracy, iteration)
        self.train_writer.add_scalar('continuous/f1_score', f1_score, iteration)
        self.train_writer.add_scalar('continuous/deployed', int(model_deployed), iteration)
        self.train_writer.add_scalar('continuous/num_games', num_games_trained, iteration)
    
    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """
        Log hyperparameters with their results.
        
        Args:
            hparams: Hyperparameter dictionary
            metrics: Resulting metrics
        """
        # Convert all values to basic types
        hparams_clean = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                hparams_clean[k] = v
            else:
                hparams_clean[k] = str(v)
        
        self.train_writer.add_hparams(
            hparam_dict=hparams_clean,
            metric_dict=metrics
        )
    
    def close(self):
        """Close all writers"""
        self.train_writer.close()
        self.val_writer.close()
        self.test_writer.close()
        self.game_writer.close()
        self.comparison_writer.close()
        
        logger.info("TensorBoard writers closed")


def create_tensorboard_logger(config: Dict[str, Any]) -> Optional[TensorBoardLogger]:
    """
    Create TensorBoard logger from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TensorBoard logger or None if not available
    """
    if not TENSORBOARD_AVAILABLE:
        logger.warning("TensorBoard not available")
        return None
    
    try:
        return TensorBoardLogger(
            log_dir=config.get('log_dir', '/workspace/tensorboard-logs'),
            experiment_name=config.get('experiment_name', 'connect4-ml')
        )
    except Exception as e:
        logger.error(f"Failed to create TensorBoard logger: {e}")
        return None
