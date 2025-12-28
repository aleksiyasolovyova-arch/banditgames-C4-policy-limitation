"""
Data Preprocessing Pipeline for Connect4 Policy Imitation

Designed to work with dataset_v1.parquet format with actual column names:
- gameId, eventId, moveIndex
- board_before_r{0-5}c{0-6} (42 features)
- board_after_r{0-5}c{0-6} (42 features)
- legal_col{0-6} (7 features)
- mcts_visits_col{0-6}, mcts_qvalue_col{0-6}, mcts_prob_col{0-6} (21 features)
- action_taken (target)
- outcome, winner, current_player, game_progress, phase
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import joblib

logger = logging.getLogger(__name__)


class Connect4Preprocessor:
    """Preprocessor for Connect4 game data matching actual dataset format"""
    
    def __init__(self, feature_config: Dict = None):
        self.feature_config = feature_config or {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from parquet file"""
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_parquet(dataset_path)
        logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        logger.info(f"Unique games: {df['gameId'].nunique()}")
        logger.info(f"Moves per game (avg): {len(df) / df['gameId'].nunique():.1f}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate dataset.
        
        - Remove rows with invalid actions
        - Filter out illegal moves
        - Handle missing values
        """
        logger.info("Cleaning dataset...")
        initial_rows = len(df)
        
        # Remove rows where action is not in 0-6
        df = df[df['action_taken'].between(0, 6)].copy()
        
        # Remove rows where the action was illegal
        # Check if legal_col{action} == 1
        invalid_moves = []
        for idx, row in df.iterrows():
            action = row['action_taken']
            legal_col = f'legal_col{action}'
            if legal_col in df.columns and row[legal_col] == 0:
                invalid_moves.append(idx)
        
        if invalid_moves:
            logger.warning(f"Removing {len(invalid_moves)} illegal moves")
            df = df.drop(invalid_moves)
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Handling {missing_before} missing values")
            # Fill MCTS values with 0 (means column wasn't explored)
            mcts_cols = [col for col in df.columns if col.startswith(('mcts_visits_', 'mcts_qvalue_', 'mcts_prob_'))]
            df[mcts_cols] = df[mcts_cols].fillna(0)
            
            # Drop rows with other missing critical values
            df = df.dropna(subset=['action_taken', 'current_player'])
        
        logger.info(f"Cleaned: {initial_rows} → {len(df)} rows ({len(df)/initial_rows*100:.1f}% retained)")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw data.
        
        New features:
        - player1_pieces: Count of player 1 pieces on board
        - player2_pieces: Count of player 2 pieces on board
        - center_control: Pieces in center columns (2, 3, 4)
        - move_number: Alias for moveIndex
        - visit_entropy: Diversity of MCTS visit distribution
        - qvalue_range: Spread of Q-values
        - top_visit_ratio: Ratio of max visits to total visits
        """
        logger.info("Engineering features...")
        df = df.copy()
        
        # Get board columns
        board_cols = [col for col in df.columns if col.startswith('board_before_')]
        
        # Count pieces for each player (assuming 1=player1, 2=player2, 0=empty)
        board_matrix = df[board_cols].values
        df['player1_pieces'] = (board_matrix == 1).sum(axis=1)
        df['player2_pieces'] = (board_matrix == 2).sum(axis=1)
        
        # Center control (columns 2, 3, 4)
        center_cols = [f'board_before_r{r}c{c}' for r in range(6) for c in [2, 3, 4]]
        center_matrix = df[center_cols].values
        df['center_control_p1'] = (center_matrix == 1).sum(axis=1)
        df['center_control_p2'] = (center_matrix == 2).sum(axis=1)
        
        # Move number (already have moveIndex)
        df['move_number'] = df['moveIndex']
        
        # MCTS statistics
        visit_cols = [f'mcts_visits_col{i}' for i in range(7)]
        qvalue_cols = [f'mcts_qvalue_col{i}' for i in range(7)]
        
        # Visit entropy - how evenly are visits distributed
        visits = df[visit_cols].values + 1e-10  # Avoid log(0)
        total_visits = visits.sum(axis=1, keepdims=True)
        visit_probs = visits / total_visits
        df['visit_entropy'] = -(visit_probs * np.log(visit_probs + 1e-10)).sum(axis=1)
        
        # Q-value range
        qvalues = df[qvalue_cols].values
        df['qvalue_range'] = qvalues.max(axis=1) - qvalues.min(axis=1)
        df['qvalue_mean'] = qvalues.mean(axis=1)
        
        # Top visit concentration
        df['top_visit_ratio'] = df[visit_cols].max(axis=1) / (df[visit_cols].sum(axis=1) + 1e-10)
        
        # Game progress (already have game_progress)
        
        logger.info(f"Added {len([c for c in df.columns if c not in board_cols + visit_cols + qvalue_cols])} new features")
        return df
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Select features for training.
        
        Features:
        - Board state (42 features): board_before_r{0-5}c{0-6}
        - MCTS statistics (21 features): visits, qvalues, probs for 7 columns
        - Legal moves (7 features): legal_col{0-6}
        - Engineered features (~10 features)
        
        Target: action_taken
        
        Returns:
            X (features), y (target), game_ids (for proper splitting)
        """
        logger.info("Selecting features...")
        
        # Board state
        board_cols = [f'board_before_r{r}c{c}' for r in range(6) for c in range(7)]
        
        # MCTS features
        mcts_visit_cols = [f'mcts_visits_col{i}' for i in range(7)]
        mcts_qvalue_cols = [f'mcts_qvalue_col{i}' for i in range(7)]
        mcts_prob_cols = [f'mcts_prob_col{i}' for i in range(7)]
        
        # Legal moves
        legal_cols = [f'legal_col{i}' for i in range(7)]
        
        # Engineered features
        engineered_cols = [
            'player1_pieces', 'player2_pieces',
            'center_control_p1', 'center_control_p2',
            'move_number', 'visit_entropy', 'qvalue_range', 'qvalue_mean',
            'top_visit_ratio'
        ]
        
        # Combine all features
        feature_cols = (board_cols + mcts_visit_cols + mcts_qvalue_cols + 
                       mcts_prob_cols + legal_cols + engineered_cols)
        
        # Filter to columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        self.feature_columns = feature_cols
        logger.info(f"Selected {len(feature_cols)} features:")
        logger.info(f"  - Board state: {len(board_cols)}")
        logger.info(f"  - MCTS visits: {len(mcts_visit_cols)}")
        logger.info(f"  - MCTS Q-values: {len(mcts_qvalue_cols)}")
        logger.info(f"  - MCTS probs: {len(mcts_prob_cols)}")
        logger.info(f"  - Legal moves: {len(legal_cols)}")
        logger.info(f"  - Engineered: {len(engineered_cols)}")
        
        X = df[feature_cols]
        y = df['action_taken']
        game_ids = df['gameId']  # Extract game IDs for proper splitting
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        return X, y, game_ids
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        game_ids: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/validation/test sets by GAME, not by move.
        
        CRITICAL: Splits at game level to prevent data leakage.
        All moves from the same game stay in the same split.
        
        Args:
            X: Features
            y: Target
            game_ids: Game identifiers for each move
            test_size: Proportion of GAMES for test set
            val_size: Proportion of remaining GAMES for validation
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Splitting data by GAME (no data leakage)...")
        
        # Get unique game IDs
        unique_games = game_ids.unique()
        n_games = len(unique_games)
        
        logger.info(f"Total unique games: {n_games}")
        logger.info(f"Total moves: {len(X)}")
        
        # Split games (not moves!) into train+val and test
        n_test_games = int(n_games * test_size)
        n_val_games = int((n_games - n_test_games) * val_size)
        
        # Shuffle games
        np.random.seed(random_state)
        shuffled_games = np.random.permutation(unique_games)
        
        # Assign games to splits
        test_games = shuffled_games[:n_test_games]
        val_games = shuffled_games[n_test_games:n_test_games + n_val_games]
        train_games = shuffled_games[n_test_games + n_val_games:]
        
        # Create boolean masks for each split
        train_mask = game_ids.isin(train_games)
        val_mask = game_ids.isin(val_games)
        test_mask = game_ids.isin(test_games)
        
        # Split the data
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        
        logger.info(f"Train: {len(train_games)} games, {len(X_train)} moves")
        logger.info(f"Val: {len(val_games)} games, {len(X_val)} moves")
        logger.info(f"Test: {len(test_games)} games, {len(X_test)} moves")
        
        # Verify no game appears in multiple splits
        assert len(set(train_games) & set(val_games)) == 0, "Train/val game overlap!"
        assert len(set(train_games) & set(test_games)) == 0, "Train/test game overlap!"
        assert len(set(val_games) & set(test_games)) == 0, "Val/test game overlap!"
        logger.info("✓ Verified: No game overlap between splits (no data leakage)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using StandardScaler.
        
        Fit on training data, transform all sets.
        """
        logger.info("Scaling features...")
        
        # Fit on training data
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Scaled to mean≈0, std≈1")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def preprocess_pipeline(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        output_dir: str = None
    ) -> Dict:
        """
        Complete preprocessing pipeline with game-level splitting (NO DATA LEAKAGE).
        
        Steps:
        1. Load dataset
        2. Clean data
        3. Engineer features
        4. Select features
        5. Split data BY GAME (critical for no leakage)
        6. Scale features
        7. Save preprocessor
        
        Returns:
            Dictionary with all processed data
        """
        logger.info("=" * 80)
        logger.info("PREPROCESSING PIPELINE START (GAME-LEVEL SPLIT)")
        logger.info("=" * 80)
        
        # Load
        df = self.load_dataset(dataset_path)
        
        # Clean
        df = self.clean_data(df)
        
        # Engineer
        df = self.engineer_features(df)
        
        # Select (now returns game_ids too)
        X, y, game_ids = self.select_features(df)
        
        # Split BY GAME (not by move!)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, game_ids, test_size, val_size, random_state
        )
        
        # Scale
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        # Save preprocessor if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            preprocessor_path = output_path / 'preprocessor.joblib'
            joblib.dump(self, preprocessor_path)
            logger.info(f"Saved preprocessor to {preprocessor_path}")
        
        logger.info("=" * 80)
        logger.info("PREPROCESSING PIPELINE COMPLETE ✅")
        logger.info("=" * 80)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_val': y_val.values,
            'y_test': y_test.values,
            'feature_names': self.feature_columns,
            'preprocessor': self
        }
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Use for inference on new game states.
        """
        if self.feature_columns is None:
            raise ValueError("Preprocessor not fitted. Run preprocess_pipeline first.")
        
        # Clean and engineer
        df = self.clean_data(df)
        df = self.engineer_features(df)
        
        # Select features
        X = df[self.feature_columns]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled


if __name__ == "__main__":
    # Test preprocessing on actual dataset
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    preprocessor = Connect4Preprocessor()
    
    # Process dataset
    results = preprocessor.preprocess_pipeline(
        dataset_path='/mnt/user-data/uploads/dataset_v1.parquet',
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        output_dir='models/preprocessing'
    )
    
    print("\n" + "=" * 80)
    print("PREPROCESSING RESULTS")
    print("=" * 80)
    print(f"Training samples: {len(results['y_train'])}")
    print(f"Validation samples: {len(results['y_val'])}")
    print(f"Test samples: {len(results['y_test'])}")
    print(f"Features: {len(results['feature_names'])}")
    print(f"\nFeature names (first 20):")
    for i, name in enumerate(results['feature_names'][:20], 1):
        print(f"  {i:2d}. {name}")
    print(f"  ... and {len(results['feature_names']) - 20} more")
