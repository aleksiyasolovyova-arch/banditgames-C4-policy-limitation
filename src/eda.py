import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Connect4EDA:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_board_from_string(self, board_series):
        """Parses '.......X...O' string format"""
        char_map = {'.': 0, 'X': 1, 'O': -1, '1': 1, '2': -1}

        def parse(s):
            if not isinstance(s, str): return [0] * 42
            return [char_map.get(c, 0) for c in s.strip()[:42]]

        matrix = pd.DataFrame(board_series.apply(parse).tolist())
        return matrix.mean(axis=0).values.reshape(6, 7)

    def _parse_board_from_columns(self, df):
        """Parses 42 columns: board_before_r0c0 ... board_before_r5c6"""
        # Generate the list of expected column names
        cols = [f"board_before_r{r}c{c}" for r in range(6) for c in range(7)]

        # Check if all exist
        if not all(col in df.columns for col in cols):
            return None

        # Extract data
        board_data = df[cols].copy()

        # CONVERT: If data is 0, 1, 2... convert 2 to -1 so the heatmap shows contrast
        # (Assuming 1=Player, 2=Opponent)
        board_data = board_data.replace({2: -1})

        # Calculate mean
        heatmap_flat = board_data.mean(axis=0).values
        return heatmap_flat.reshape(6, 7)

    def generate_report(self, df: pd.DataFrame, version: str):
        logger.info(f" Generating EDA report for {version}...")

        # --- 1. Normalize Action Column ---
        if 'actionTaken' in df.columns:
            df['action_taken'] = df['actionTaken']

        # --- 2. Plot Action Distribution ---
        if 'action_taken' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=df['action_taken'])
            plt.title(f"Action Distribution (Version {version})")
            plt.xlabel("Column Chosen (0-6)")
            plt.ylabel("Count")
            output_path = self.output_dir / f"action_dist_{version}.png"
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved {output_path}")
        else:
            logger.warning("Skipping Action Plot: 'action_taken' not found.")

        # --- 3. Plot Board Heatmap ---
        heatmap_grid = None

        # Strategy A: Try String Column
        if 'boardBefore' in df.columns:  # legacy camelCase
            heatmap_grid = self._parse_board_from_string(df['boardBefore'])
        elif 'board_before' in df.columns:  # snake_case
            heatmap_grid = self._parse_board_from_string(df['board_before'])

        # Strategy B: Try Flattened Columns
        if heatmap_grid is None:
            heatmap_grid = self._parse_board_from_columns(df)

        if heatmap_grid is not None:
            try:
                plt.figure(figsize=(7, 6))
                sns.heatmap(heatmap_grid, cmap="coolwarm", center=0, annot=True, fmt=".2f")
                plt.title("Board Occupancy (Red=P1, Blue=P2)")
                output_path = self.output_dir / f"heatmap_{version}.png"
                plt.savefig(output_path)
                plt.close()
                logger.info(f"Saved {output_path}")
            except Exception as e:
                logger.error(f"Failed to plot heatmap: {e}")
        else:
            logger.warning("Skipping Heatmap: Could not find board data (neither string nor flattened columns).")