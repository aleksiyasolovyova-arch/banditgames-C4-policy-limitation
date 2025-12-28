"""
Player Performance Monitoring and Comparison

Tracks and compares performance of:
- MCTS AI (different difficulty levels)
- ML Model
- Human players

Provides visualizations and analytics.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerPerformanceMonitor:
    """
    Monitor and compare player performance across different player types.
    
    Player types:
    - MCTS (easy, medium, hard, expert)
    - ML Model
    - Human
    """
    
    def __init__(
        self,
        output_dir: str = "/workspace/monitoring",
        postgres_conn_str: Optional[str] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            output_dir: Directory for saving reports and visualizations
            postgres_conn_str: PostgreSQL connection string for game data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.postgres_conn_str = postgres_conn_str
    
    def collect_game_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Collect game data from database.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with game data
        """
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        if not self.postgres_conn_str:
            logger.error("No PostgreSQL connection string provided")
            return pd.DataFrame()
        
        # Default to last 30 days
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        query = """
        SELECT 
            g.id as game_id,
            g.created_at,
            g.finished_at,
            g.winner_id,
            p1.id as player1_id,
            p1.username as player1_name,
            p1.player_type as player1_type,
            p2.id as player2_id,
            p2.username as player2_name,
            p2.player_type as player2_type,
            m.move_number,
            m.player_id as move_player_id,
            m.column as move_column,
            m.inference_time_ms,
            m.mcts_visits,
            m.mcts_depth
        FROM games g
        JOIN players p1 ON g.player1_id = p1.id
        JOIN players p2 ON g.player2_id = p2.id
        LEFT JOIN moves m ON g.id = m.game_id
        WHERE g.created_at >= %s AND g.created_at <= %s
        ORDER BY g.id, m.move_number
        """
        
        try:
            conn = psycopg2.connect(self.postgres_conn_str)
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()
            
            logger.info(f"Collected {len(df)} moves from {df['game_id'].nunique()} games")
            return df
        
        except Exception as e:
            logger.error(f"Error collecting game data: {e}")
            return pd.DataFrame()
    
    def analyze_player_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance metrics for each player type.
        
        Returns:
            Dictionary with performance metrics per player type
        """
        results = {}
        
        # Get unique games
        games = df.drop_duplicates(subset=['game_id'])
        
        # Analyze each player type
        player_types = set(games['player1_type'].unique()) | set(games['player2_type'].unique())
        
        for player_type in player_types:
            # Games where this player type participated
            player_games = games[
                (games['player1_type'] == player_type) | 
                (games['player2_type'] == player_type)
            ]
            
            # Win rate
            wins_as_p1 = len(player_games[
                (player_games['player1_type'] == player_type) & 
                (player_games['winner_id'] == player_games['player1_id'])
            ])
            
            wins_as_p2 = len(player_games[
                (player_games['player2_type'] == player_type) & 
                (player_games['winner_id'] == player_games['player2_id'])
            ])
            
            total_games = len(player_games)
            win_rate = (wins_as_p1 + wins_as_p2) / total_games if total_games > 0 else 0
            
            # Average game duration
            player_games['duration'] = (
                pd.to_datetime(player_games['finished_at']) - 
                pd.to_datetime(player_games['created_at'])
            ).dt.total_seconds()
            avg_duration = player_games['duration'].mean()
            
            # Move statistics
            player_moves = df[df['move_player_id'].isin(
                player_games[player_games['player1_type'] == player_type]['player1_id'].tolist() +
                player_games[player_games['player2_type'] == player_type]['player2_id'].tolist()
            )]
            
            avg_inference_time = player_moves['inference_time_ms'].mean()
            avg_mcts_visits = player_moves['mcts_visits'].mean()
            
            results[player_type] = {
                'total_games': total_games,
                'wins': wins_as_p1 + wins_as_p2,
                'win_rate': win_rate,
                'avg_game_duration_sec': avg_duration,
                'avg_inference_time_ms': avg_inference_time,
                'avg_mcts_visits': avg_mcts_visits,
                'total_moves': len(player_moves)
            }
        
        return results
    
    def compare_move_quality(
        self,
        df: pd.DataFrame,
        ml_predictions: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Compare move quality across player types.
        
        Args:
            df: Game data
            ml_predictions: ML model predictions for comparison
            
        Returns:
            DataFrame with move quality comparison
        """
        # Calculate move quality metrics
        move_analysis = []
        
        for _, move in df.iterrows():
            analysis = {
                'game_id': move['game_id'],
                'move_number': move['move_number'],
                'player_type': move['player_type'],
                'column': move['move_column'],
                'inference_time_ms': move['inference_time_ms']
            }
            
            # If we have ML predictions, compare
            if ml_predictions and move['game_id'] in ml_predictions:
                ml_move = ml_predictions[move['game_id']].get(move['move_number'])
                analysis['ml_agreement'] = 1 if ml_move == move['move_column'] else 0
            
            move_analysis.append(analysis)
        
        return pd.DataFrame(move_analysis)
    
    def generate_speed_comparison(self, performance_data: Dict[str, Dict[str, float]]):
        """Generate speed comparison visualization"""
        import matplotlib.pyplot as plt
        
        player_types = list(performance_data.keys())
        inference_times = [performance_data[pt]['avg_inference_time_ms'] for pt in player_types]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {
            'ML': 'green',
            'MCTS_EASY': 'lightblue',
            'MCTS_MEDIUM': 'blue',
            'MCTS_HARD': 'darkblue',
            'MCTS_EXPERT': 'purple',
            'HUMAN': 'orange'
        }
        
        bars = ax.bar(
            range(len(player_types)),
            inference_times,
            color=[colors.get(pt, 'gray') for pt in player_types]
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1f}ms',
                ha='center',
                va='bottom'
            )
        
        ax.set_xlabel('Player Type')
        ax.set_ylabel('Average Inference Time (ms)')
        ax.set_title('Speed Comparison: Inference Time by Player Type')
        ax.set_xticks(range(len(player_types)))
        ax.set_xticklabels(player_types, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add speedup annotations
        if 'ML' in player_types and 'MCTS_EXPERT' in player_types:
            ml_time = performance_data['ML']['avg_inference_time_ms']
            expert_time = performance_data['MCTS_EXPERT']['avg_inference_time_ms']
            speedup = expert_time / ml_time
            
            ax.text(
                0.5, 0.95,
                f'ML is {speedup:.1f}x faster than MCTS Expert',
                transform=ax.transAxes,
                ha='center',
                va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=12,
                fontweight='bold'
            )
        
        plt.tight_layout()
        output_path = self.output_dir / 'speed_comparison.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Speed comparison saved to {output_path}")
        return output_path
    
    def generate_win_rate_comparison(self, performance_data: Dict[str, Dict[str, float]]):
        """Generate win rate comparison visualization"""
        import matplotlib.pyplot as plt
        
        player_types = list(performance_data.keys())
        win_rates = [performance_data[pt]['win_rate'] * 100 for pt in player_types]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(player_types)), win_rates, color='steelblue')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom'
            )
        
        ax.set_xlabel('Player Type')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate Comparison by Player Type')
        ax.set_xticks(range(len(player_types)))
        ax.set_xticklabels(player_types, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
        ax.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / 'win_rate_comparison.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Win rate comparison saved to {output_path}")
        return output_path
    
    def generate_move_distribution_heatmap(self, df: pd.DataFrame, player_type: str):
        """Generate heatmap of move distribution by column"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Filter by player type
        player_moves = df[df['player_type'] == player_type]
        
        # Create position heatmap (6 rows x 7 columns)
        heatmap_data = np.zeros((6, 7))
        
        for _, move in player_moves.iterrows():
            col = move['move_column']
            # Estimate row (simplified - in reality would need board state)
            heatmap_data[0, col] += 1  # Just count moves per column for now
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Move Count'},
            ax=ax
        )
        
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(f'Move Distribution Heatmap - {player_type}')
        
        plt.tight_layout()
        output_path = self.output_dir / f'move_distribution_{player_type}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Move distribution heatmap saved to {output_path}")
        return output_path
    
    def generate_skill_progression_chart(self, df: pd.DataFrame):
        """Show how AI skill varies by game state (early/mid/late game)"""
        import matplotlib.pyplot as plt
        
        # Divide games into phases
        df['game_phase'] = pd.cut(
            df['move_number'],
            bins=[0, 10, 25, 42],
            labels=['Early (1-10)', 'Mid (11-25)', 'Late (26+)']
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, phase in enumerate(['Early (1-10)', 'Mid (11-25)', 'Late (26+)']):
            phase_data = df[df['game_phase'] == phase]
            
            # Group by player type
            player_stats = phase_data.groupby('player_type')['inference_time_ms'].mean()
            
            axes[idx].bar(range(len(player_stats)), player_stats.values, color='coral')
            axes[idx].set_title(f'{phase} Game')
            axes[idx].set_xlabel('Player Type')
            axes[idx].set_ylabel('Avg Inference Time (ms)')
            axes[idx].set_xticks(range(len(player_stats)))
            axes[idx].set_xticklabels(player_stats.index, rotation=45, ha='right')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle('AI Performance by Game Phase', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'skill_by_game_phase.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Skill progression chart saved to {output_path}")
        return output_path
    
    def export_comparison_report(
        self,
        performance_data: Dict[str, Dict[str, float]],
        move_quality: pd.DataFrame
    ) -> Path:
        """Export comprehensive comparison report as JSON"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'performance_by_player_type': performance_data,
            'move_quality_summary': {
                'total_moves_analyzed': len(move_quality),
                'ml_agreement_rate': move_quality['ml_agreement'].mean() if 'ml_agreement' in move_quality.columns else None
            },
            'visualizations': {
                'speed_comparison': 'speed_comparison.png',
                'win_rate_comparison': 'win_rate_comparison.png',
                'skill_progression': 'skill_by_game_phase.png'
            }
        }
        
        output_path = self.output_dir / 'performance_comparison_report.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to {output_path}")
        return output_path
    
    def generate_full_report(self, start_date: Optional[datetime] = None):
        """Generate complete performance comparison report"""
        logger.info("Generating full performance comparison report...")
        
        # Collect data
        df = self.collect_game_data(start_date=start_date)
        
        if df.empty:
            logger.error("No data collected, cannot generate report")
            return
        
        # Analyze performance
        performance_data = self.analyze_player_performance(df)
        
        # Generate visualizations
        self.generate_speed_comparison(performance_data)
        self.generate_win_rate_comparison(performance_data)
        self.generate_skill_progression_chart(df)
        
        # Generate move distribution for each player type
        for player_type in performance_data.keys():
            self.generate_move_distribution_heatmap(df, player_type)
        
        # Compare move quality
        move_quality = self.compare_move_quality(df)
        
        # Export report
        report_path = self.export_comparison_report(performance_data, move_quality)
        
        logger.info(f"Full report generated at {self.output_dir}")
        logger.info(f"Report JSON: {report_path}")
        
        return self.output_dir
