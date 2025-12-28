"""
Exploratory Data Analysis (EDA) for Connect4 ML Dataset

Comprehensive analysis of training data including:
- Dataset statistics
- Feature distributions
- Target variable analysis
- MCTS behavior patterns
- Game phase analysis
- Correlation analysis
- Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Connect4EDA:
    """Exploratory Data Analysis for Connect4 dataset"""
    
    def __init__(self, output_dir: str = "./eda_reports"):
        """
        Initialize EDA.
        
        Args:
            output_dir: Directory to save plots and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"EDA output directory: {self.output_dir}")
    
    def generate_full_report(
        self,
        df: pd.DataFrame,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        save_plots: bool = True
    ) -> Dict:
        """
        Generate comprehensive EDA report.
        
        Args:
            df: Full dataset
            X_train: Training features (optional, for train-specific analysis)
            y_train: Training target (optional)
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("=" * 80)
        logger.info("GENERATING COMPREHENSIVE EDA REPORT")
        logger.info("=" * 80)
        
        report = {}
        
        # 1. Dataset overview
        logger.info("1. Dataset Overview")
        report['overview'] = self.dataset_overview(df)
        
        # 2. Target variable analysis
        logger.info("2. Target Variable Analysis")
        report['target_analysis'] = self.analyze_target(df, save_plots)
        
        # 3. Feature statistics
        logger.info("3. Feature Statistics")
        report['feature_stats'] = self.feature_statistics(df)
        
        # 4. MCTS behavior analysis
        logger.info("4. MCTS Behavior Analysis")
        report['mcts_analysis'] = self.analyze_mcts_behavior(df, save_plots)
        
        # 5. Game phase analysis
        logger.info("5. Game Phase Analysis")
        report['phase_analysis'] = self.analyze_game_phases(df, save_plots)
        
        # 6. Board state analysis
        logger.info("6. Board State Analysis")
        report['board_analysis'] = self.analyze_board_states(df, save_plots)
        
        # 7. Correlation analysis
        logger.info("7. Correlation Analysis")
        if X_train is not None and y_train is not None:
            report['correlation'] = self.correlation_analysis(X_train, y_train, save_plots)
        
        # 8. Data quality checks
        logger.info("8. Data Quality Checks")
        report['quality'] = self.data_quality_checks(df)
        
        # Save summary report
        self.save_text_report(report)
        
        logger.info("=" * 80)
        logger.info(f"EDA COMPLETE - Reports saved to: {self.output_dir}")
        logger.info("=" * 80)
        
        return report
    
    def dataset_overview(self, df: pd.DataFrame) -> Dict:
        """Generate dataset overview statistics"""
        overview = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'unique_games': df['gameId'].nunique() if 'gameId' in df.columns else None,
            'avg_moves_per_game': len(df) / df['gameId'].nunique() if 'gameId' in df.columns else None,
            'min_moves_per_game': df.groupby('gameId').size().min() if 'gameId' in df.columns else None,
            'max_moves_per_game': df.groupby('gameId').size().max() if 'gameId' in df.columns else None,
            'missing_values': df.isnull().sum().sum(),
            'duplicated_rows': df.duplicated().sum()
        }
        
        logger.info(f"Total rows: {overview['total_rows']:,}")
        logger.info(f"Total columns: {overview['total_columns']}")
        logger.info(f"Memory usage: {overview['memory_usage_mb']:.2f} MB")
        logger.info(f"Unique games: {overview['unique_games']:,}")
        logger.info(f"Avg moves/game: {overview['avg_moves_per_game']:.1f}")
        logger.info(f"Game length range: {overview['min_moves_per_game']}-{overview['max_moves_per_game']} moves")
        
        return overview
    
    def analyze_target(self, df: pd.DataFrame, save_plots: bool = True) -> Dict:
        """Analyze target variable distribution"""
        target_col = 'action_taken'
        
        # Value counts
        value_counts = df[target_col].value_counts().sort_index()
        percentages = df[target_col].value_counts(normalize=True).sort_index() * 100
        
        analysis = {
            'value_counts': value_counts.to_dict(),
            'percentages': percentages.to_dict(),
            'most_common_column': value_counts.idxmax(),
            'least_common_column': value_counts.idxmin(),
            'imbalance_ratio': value_counts.max() / value_counts.min()
        }
        
        logger.info(f"Target distribution (action_taken):")
        for col in range(7):
            count = value_counts.get(col, 0)
            pct = percentages.get(col, 0)
            logger.info(f"  Column {col}: {count:6,} ({pct:5.2f}%)")
        logger.info(f"Most popular: Column {analysis['most_common_column']}")
        logger.info(f"Imbalance ratio: {analysis['imbalance_ratio']:.2f}x")
        
        if save_plots:
            # Bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Count plot
            ax1.bar(value_counts.index, value_counts.values, color='steelblue', alpha=0.8)
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Count')
            ax1.set_title('Target Distribution - Move Counts per Column')
            ax1.set_xticks(range(7))
            ax1.grid(axis='y', alpha=0.3)
            
            # Percentage plot
            ax2.bar(percentages.index, percentages.values, color='coral', alpha=0.8)
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Target Distribution - Percentage of Moves per Column')
            ax2.set_xticks(range(7))
            ax2.axhline(y=100/7, color='red', linestyle='--', alpha=0.5, label='Uniform (14.29%)')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved: target_distribution.png")
        
        return analysis
    
    def feature_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate feature statistics"""
        stats = {
            'numeric_features': {},
            'categorical_features': {}
        }
        
        # Numeric features summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats['numeric_features'] = {
            'count': len(numeric_cols),
            'summary': df[numeric_cols].describe().to_dict()
        }
        
        logger.info(f"Numeric features: {len(numeric_cols)}")
        
        return stats
    
    def analyze_mcts_behavior(self, df: pd.DataFrame, save_plots: bool = True) -> Dict:
        """Analyze MCTS visit/Q-value/probability patterns"""
        analysis = {}
        
        # Get MCTS columns
        visit_cols = [col for col in df.columns if col.startswith('mcts_visits_')]
        qvalue_cols = [col for col in df.columns if col.startswith('mcts_qvalue_')]
        prob_cols = [col for col in df.columns if col.startswith('mcts_prob_')]
        
        if not visit_cols:
            logger.warning("No MCTS columns found")
            return analysis
        
        # Total visits per move
        df['total_visits'] = df[visit_cols].sum(axis=1)
        
        # Visit concentration (entropy)
        visit_probs = df[visit_cols].div(df['total_visits'], axis=0).fillna(0)
        df['visit_entropy'] = -(visit_probs * np.log(visit_probs + 1e-10)).sum(axis=1)
        
        analysis['avg_total_visits'] = df['total_visits'].mean()
        analysis['avg_visit_entropy'] = df['visit_entropy'].mean()
        analysis['max_visits_column'] = df[visit_cols].sum().idxmax()
        
        logger.info(f"Avg MCTS visits per move: {analysis['avg_total_visits']:.0f}")
        logger.info(f"Avg visit entropy: {analysis['avg_visit_entropy']:.3f}")
        
        if save_plots:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Total visits distribution
            axes[0, 0].hist(df['total_visits'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Total MCTS Visits')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Total MCTS Visits per Move')
            axes[0, 0].axvline(df['total_visits'].mean(), color='red', linestyle='--', label=f'Mean: {df["total_visits"].mean():.0f}')
            axes[0, 0].legend()
            
            # 2. Visit entropy distribution
            axes[0, 1].hist(df['visit_entropy'], bins=50, color='coral', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Visit Entropy')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Visit Entropy (Exploration vs Exploitation)')
            axes[0, 1].axvline(df['visit_entropy'].mean(), color='red', linestyle='--', label=f'Mean: {df["visit_entropy"].mean():.3f}')
            axes[0, 1].legend()
            
            # 3. Visits per column
            visit_sums = df[visit_cols].sum()
            visit_sums.index = [col.replace('mcts_visits_col', 'Col ') for col in visit_sums.index]
            axes[1, 0].bar(range(7), visit_sums.values, color='green', alpha=0.7)
            axes[1, 0].set_xlabel('Column')
            axes[1, 0].set_ylabel('Total Visits')
            axes[1, 0].set_title('MCTS Visits by Column')
            axes[1, 0].set_xticks(range(7))
            
            # 4. Average Q-values per column
            if qvalue_cols:
                qvalue_means = df[qvalue_cols].mean()
                qvalue_means.index = [col.replace('mcts_qvalue_col', 'Col ') for col in qvalue_means.index]
                axes[1, 1].bar(range(7), qvalue_means.values, color='purple', alpha=0.7)
                axes[1, 1].set_xlabel('Column')
                axes[1, 1].set_ylabel('Average Q-Value')
                axes[1, 1].set_title('Average MCTS Q-Values by Column')
                axes[1, 1].set_xticks(range(7))
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'mcts_behavior.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved: mcts_behavior.png")
        
        return analysis
    
    def analyze_game_phases(self, df: pd.DataFrame, save_plots: bool = True) -> Dict:
        """Analyze behavior across game phases"""
        analysis = {}
        
        if 'moveIndex' not in df.columns:
            return analysis
        
        # Define phases
        df['phase_category'] = pd.cut(
            df['moveIndex'],
            bins=[0, 10, 25, float('inf')],
            labels=['Early (1-10)', 'Mid (11-25)', 'Late (26+)'],
            include_lowest=True
        )
        
        # Moves per phase
        phase_counts = df['phase_category'].value_counts()
        analysis['moves_per_phase'] = phase_counts.to_dict()
        
        # Target distribution per phase
        phase_target = df.groupby(['phase_category', 'action_taken']).size().unstack(fill_value=0)
        
        logger.info("Moves by game phase:")
        for phase, count in phase_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  {phase}: {count:,} ({pct:.1f}%)")
        
        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Phase distribution
            axes[0].bar(range(len(phase_counts)), phase_counts.values, color='steelblue', alpha=0.8)
            axes[0].set_xlabel('Game Phase')
            axes[0].set_ylabel('Number of Moves')
            axes[0].set_title('Move Distribution by Game Phase')
            axes[0].set_xticks(range(len(phase_counts)))
            axes[0].set_xticklabels(phase_counts.index, rotation=0)
            
            # Target distribution by phase (stacked bar)
            phase_target_pct = phase_target.div(phase_target.sum(axis=1), axis=0) * 100
            phase_target_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10')
            axes[1].set_xlabel('Game Phase')
            axes[1].set_ylabel('Percentage (%)')
            axes[1].set_title('Column Preference by Game Phase')
            axes[1].legend(title='Column', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].set_xticklabels(phase_target_pct.index, rotation=0)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'game_phases.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved: game_phases.png")
        
        return analysis
    
    def analyze_board_states(self, df: pd.DataFrame, save_plots: bool = True) -> Dict:
        """Analyze board state patterns"""
        analysis = {}
        
        # Get board columns
        board_cols = [col for col in df.columns if col.startswith('board_before_')]
        
        if not board_cols:
            return analysis
        
        # Calculate board fullness
        df['board_fullness'] = df[board_cols].apply(lambda x: (x != 0).sum(), axis=1)
        df['board_fullness_pct'] = df['board_fullness'] / 42 * 100
        
        analysis['avg_board_fullness'] = df['board_fullness'].mean()
        analysis['avg_board_fullness_pct'] = df['board_fullness_pct'].mean()
        
        logger.info(f"Avg board fullness: {analysis['avg_board_fullness']:.1f} / 42 ({analysis['avg_board_fullness_pct']:.1f}%)")
        
        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Board fullness distribution
            axes[0].hist(df['board_fullness_pct'], bins=30, color='teal', alpha=0.7, edgecolor='black')
            axes[0].set_xlabel('Board Fullness (%)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of Board Fullness')
            axes[0].axvline(df['board_fullness_pct'].mean(), color='red', linestyle='--', label=f'Mean: {df["board_fullness_pct"].mean():.1f}%')
            axes[0].legend()
            
            # Fullness vs move index
            axes[1].scatter(df['moveIndex'], df['board_fullness_pct'], alpha=0.1, s=5)
            axes[1].set_xlabel('Move Index')
            axes[1].set_ylabel('Board Fullness (%)')
            axes[1].set_title('Board Fullness vs Move Index')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'board_states.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved: board_states.png")
        
        return analysis
    
    def correlation_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_plots: bool = True,
        top_n: int = 20
    ) -> Dict:
        """Analyze feature correlations with target"""
        analysis = {}
        
        # Calculate correlations with target
        # For categorical target, use point-biserial correlation for each class
        correlations = {}
        for col in X.columns:
            try:
                corr = X[col].corr(y)
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                pass
        
        # Sort by absolute correlation
        correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        analysis['top_correlations'] = dict(list(correlations.items())[:top_n])
        
        logger.info(f"Top {top_n} features by correlation with target:")
        for i, (feat, corr) in enumerate(list(correlations.items())[:top_n], 1):
            logger.info(f"  {i:2d}. {feat[:40]:40s} {corr:.4f}")
        
        if save_plots:
            # Top correlations bar chart
            top_feats = list(correlations.keys())[:top_n]
            top_corrs = [correlations[f] for f in top_feats]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(top_n), top_corrs, color='steelblue', alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([f[:40] for f in top_feats])
            ax.set_xlabel('Absolute Correlation')
            ax.set_title(f'Top {top_n} Features by Correlation with Target')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved: feature_correlations.png")
        
        return analysis
    
    def data_quality_checks(self, df: pd.DataFrame) -> Dict:
        """Perform data quality checks"""
        quality = {}
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        quality['missing_values'] = {
            col: {'count': int(count), 'percentage': float(pct)}
            for col, count, pct in zip(missing.index, missing.values, missing_pct.values)
            if count > 0
        }
        
        # Duplicates
        quality['duplicated_rows'] = int(df.duplicated().sum())
        
        # Value ranges
        if 'action_taken' in df.columns:
            invalid_actions = df[~df['action_taken'].between(0, 6)].shape[0]
            quality['invalid_actions'] = int(invalid_actions)
        
        logger.info("Data Quality Summary:")
        logger.info(f"  Missing values: {len(quality['missing_values'])} columns affected")
        logger.info(f"  Duplicated rows: {quality['duplicated_rows']}")
        logger.info(f"  Invalid actions: {quality.get('invalid_actions', 0)}")
        
        return quality
    
    def save_text_report(self, report: Dict):
        """Save text summary report"""
        report_path = self.output_dir / 'eda_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CONNECT4 ML DATASET - EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overview
            if 'overview' in report:
                f.write("DATASET OVERVIEW\n")
                f.write("-" * 80 + "\n")
                for key, value in report['overview'].items():
                    f.write(f"{key:30s}: {value}\n")
                f.write("\n")
            
            # Target analysis
            if 'target_analysis' in report:
                f.write("TARGET VARIABLE ANALYSIS\n")
                f.write("-" * 80 + "\n")
                for key, value in report['target_analysis'].items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Data quality
            if 'quality' in report:
                f.write("DATA QUALITY\n")
                f.write("-" * 80 + "\n")
                for key, value in report['quality'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        
        logger.info(f"Saved text report: {report_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eda.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    
    # Run EDA
    eda = Connect4EDA(output_dir="./eda_reports")
    report = eda.generate_full_report(df, save_plots=True)
    
    print("\nEDA Complete! Check ./eda_reports/ for visualizations and summary.")
