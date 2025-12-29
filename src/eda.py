import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Connect4EDA:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_board(self, board_series):
        """Parses '.......X...O' into a numeric series for heatmap analysis"""
        char_map = {'.': 0, 'X': 1, 'O': -1}  # X=1, O=-1 helps visualize balance

        def parse(s):
            if not isinstance(s, str): return [0] * 42
            return [char_map.get(c, 0) for c in s.strip()[:42]]

        # Convert to DataFrame
        matrix = pd.DataFrame(board_series.apply(parse).tolist())
        # Reshape for 6x7 board (mean across all games)
        heatmap_grid = matrix.mean(axis=0).values.reshape(6, 7)
        return heatmap_grid

    def generate_report(self, df: pd.DataFrame, version: str):
        logger.info(f" Generating EDA report for {version}...")

        # 1. Class Balance (Action Taken)
        plt.figure(figsize=(8, 5))
        if 'action_taken' in df.columns:
            sns.countplot(x=df['action_taken'])
            plt.title(f"Action Distribution (Version {version})")
            plt.xlabel("Column Chosen (0-6)")
            plt.ylabel("Count")
            plt.savefig(self.output_dir / f"action_dist_{version}.png")
            plt.close()

        # 2. Board Heatmap (Where are pieces usually played?)
        if 'board_before' in df.columns:
            try:
                heatmap_grid = self._parse_board(df['board_before'])
                plt.figure(figsize=(7, 6))
                sns.heatmap(heatmap_grid, cmap="coolwarm", center=0, annot=True)
                plt.title("Board Occupancy Heatmap (Red=P1, Blue=P2)")
                plt.savefig(self.output_dir / f"heatmap_{version}.png")
                plt.close()
            except Exception as e:
                logger.warning(f"Could not generate heatmap: {e}")

        logger.info(f" EDA charts saved to {self.output_dir}")