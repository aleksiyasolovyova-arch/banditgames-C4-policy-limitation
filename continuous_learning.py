"""
Continuous Learning Orchestrator for Connect4 ML System

This system creates a closed learning loop:
1. AI plays games → generates new data
2. Logger captures gameplay → exports dataset
3. ML model retrains on new data
4. Updated model improves AI player
5. Loop continues...

Usage:
    python continuous_learning.py --config config.yaml
"""

import asyncio
import logging
import time
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import yaml
import json
import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousLearningOrchestrator:
    """
    Orchestrates the continuous learning loop for Connect4 ML system.
    
    Workflow:
    1. Generate new self-play games
    2. Export dataset from logger
    3. Evaluate if retraining is needed
    4. Train new model version
    5. Compare with baseline
    6. Deploy if improved
    7. Repeat
    """
    
    def __init__(self, config_path: str = "learning_config.yaml"):
        """Initialize orchestrator with config"""
        self.config = self._load_config(config_path)
        self.iteration = 0
        self.best_accuracy = 0.0
        self.baseline_model_path = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file with environment variable overrides"""
        import os
        
        default_config = {
            'game_generation': {
                'games_per_iteration': int(os.getenv('GAMES_PER_ITERATION', 100)),
                'orchestration_dir': os.getenv('ORCHESTRATION_DIR', '~/KDGY3/Integration5/orchestration'),
                'generation_script': os.getenv('GENERATION_SCRIPT', 'scripts/generate_self_play_data.py')
            },
            'dataset_export': {
                'logger_api_url': os.getenv('LOGGER_API_URL', 'http://localhost:8010'),
                'export_endpoint': '/dataset/export',
                'dataset_dir': os.getenv('DATASET_DIR', '/workspace/datasets'),  # Container path
                'min_new_games': int(os.getenv('MIN_NEW_GAMES', 50))
            },
            'model_training': {
                'model_type': os.getenv('MODEL_TYPE', 'xgboost'),
                'test_size': 0.2,
                'val_size': 0.1,
                'models_dir': os.getenv('MODEL_DIR', '/workspace/models'),  # Container path
                'min_accuracy_improvement': float(os.getenv('MIN_ACCURACY_IMPROVEMENT', 0.01))
            },
            'deployment': {
                'ml_api_url': os.getenv('ML_API_URL', 'http://localhost:8000'),
                'auto_deploy': os.getenv('AUTO_DEPLOY', 'false').lower() == 'true',
                'backup_previous': True
            },
            'scheduling': {
                'mode': 'interval',  # 'interval' or 'cron'
                'interval_hours': int(os.getenv('INTERVAL_HOURS', 24)),
                'max_iterations': int(os.getenv('MAX_ITERATIONS')) if os.getenv('MAX_ITERATIONS') else None
            },
            'monitoring': {
                'log_dir': os.getenv('LOG_DIR', '/workspace/logs'),
                'metrics_file': 'learning_history.json'
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge configs
                for key in user_config:
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
        
        return default_config
    
    async def generate_games(self, num_games: int) -> bool:
        """
        Generate new self-play games.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating {num_games} new self-play games...")
        
        orchestration_dir = Path(self.config['game_generation']['orchestration_dir']).expanduser()
        script_path = orchestration_dir / self.config['game_generation']['generation_script']
        
        if not script_path.exists():
            logger.error(f"Generation script not found: {script_path}")
            return False
        
        try:
            # Run generation script
            cmd = ['python3', str(script_path), '--count', str(num_games)]
            result = subprocess.run(
                cmd,
                cwd=str(orchestration_dir),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully generated {num_games} games")
                return True
            else:
                logger.error(f"Game generation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Game generation timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error generating games: {e}")
            return False
    
    async def export_dataset(self) -> Optional[str]:
        """
        Export dataset from logger service.
        
        Returns:
            Dataset path if successful, None otherwise
        """
        logger.info("Exporting dataset from logger...")
        
        api_url = self.config['dataset_export']['logger_api_url']
        endpoint = self.config['dataset_export']['export_endpoint']
        dataset_dir = Path(self.config['dataset_export']['dataset_dir'])
        
        # Create version string
        version = f"v{self.iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            response = requests.post(
                f"{api_url}{endpoint}",
                json={"version": version},
                timeout=300  # 5 minutes
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Logger returns filename or relative path
                # We need to construct full container path
                dataset_filename = data.get('dataset_path', f'dataset_{version}.parquet')
                
                # If it's just a filename, prepend the dataset directory
                if '/' not in dataset_filename:
                    dataset_path = dataset_dir / dataset_filename
                else:
                    # If it's already a full path, use as-is
                    dataset_path = Path(dataset_filename)
                
                # Verify file exists (in container filesystem)
                if dataset_path.exists():
                    logger.info(f"Dataset exported: {dataset_path}")
                    return str(dataset_path)
                else:
                    # Try finding the latest dataset in the directory
                    logger.warning(f"Expected dataset not found at {dataset_path}, searching for latest...")
                    datasets = list(dataset_dir.glob('dataset_*.parquet'))
                    if datasets:
                        latest = max(datasets, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Using latest dataset: {latest}")
                        return str(latest)
                    else:
                        logger.error(f"No datasets found in {dataset_dir}")
                        return None
            else:
                logger.error(f"Dataset export failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            return None
    
    async def train_model(self, dataset_path: str) -> Optional[Dict]:
        """
        Train new model on updated dataset.
        
        Returns:
            Training results dict if successful, None otherwise
        """
        logger.info(f"Training model on {dataset_path}...")
        
        from src.train import train_connect4_model
        
        try:
            # Create version directory
            version = f"v{self.iteration}"
            output_dir = Path(self.config['model_training']['models_dir']) / version
            
            # Train model
            results = train_connect4_model(
                dataset_path=dataset_path,
                model_type=self.config['model_training']['model_type'],
                output_dir=str(output_dir),
                version=version,
                # Hyperparameters
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
            
            logger.info(f"Model trained - Accuracy: {results['metrics']['accuracy']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    async def evaluate_improvement(
        self, 
        new_results: Dict, 
        baseline_path: Optional[str] = None
    ) -> bool:
        """
        Evaluate if new model is better than baseline.
        
        Returns:
            True if new model should be deployed, False otherwise
        """
        new_accuracy = new_results['metrics']['accuracy']
        
        if baseline_path is None:
            # First model, always deploy
            logger.info(f"First model - deploying with {new_accuracy:.4f} accuracy")
            self.best_accuracy = new_accuracy
            return True
        
        # Compare with baseline
        min_improvement = self.config['model_training']['min_accuracy_improvement']
        improvement = new_accuracy - self.best_accuracy
        
        logger.info(f"New accuracy: {new_accuracy:.4f}")
        logger.info(f"Best accuracy: {self.best_accuracy:.4f}")
        logger.info(f"Improvement: {improvement:+.4f} (min required: {min_improvement:.4f})")
        
        if improvement >= min_improvement:
            logger.info("✓ Model improved - recommending deployment")
            self.best_accuracy = new_accuracy
            return True
        else:
            logger.info("✗ Model did not improve enough - keeping baseline")
            return False
    
    async def deploy_model(self, model_path: str, backup: bool = True) -> bool:
        """
        Deploy new model to production.
        
        Args:
            model_path: Path to new model
            backup: Whether to backup current model
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Deploying model from {model_path}...")
        
        if backup and self.baseline_model_path:
            # Backup current model
            backup_path = Path(self.baseline_model_path).parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            try:
                subprocess.run(['cp', self.baseline_model_path, str(backup_path)], check=True)
                logger.info(f"Backed up previous model to {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to backup model: {e}")
        
        # Deploy new model
        try:
            # In production, this might involve:
            # 1. Copying model to API server location
            # 2. Restarting API server
            # 3. Running health checks
            # 4. Gradual rollout
            
            production_path = Path("models/production/model.joblib")
            production_path.parent.mkdir(parents=True, exist_ok=True)
            
            subprocess.run(['cp', model_path, str(production_path)], check=True)
            logger.info(f"Deployed model to {production_path}")
            
            # Update baseline
            self.baseline_model_path = str(production_path)
            
            # Optionally restart API server
            if self.config['deployment'].get('restart_api', False):
                logger.info("Restarting ML API server...")
                # Add restart logic here
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def log_iteration_metrics(self, results: Dict, deployed: bool):
        """Log metrics for this iteration"""
        metrics_file = Path(self.config['monitoring']['log_dir']) / self.config['monitoring']['metrics_file']
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        iteration_data = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'accuracy': results['metrics']['accuracy'],
            'f1_macro': results['metrics']['f1_macro'],
            'top_3_accuracy': results['metrics']['top_3_accuracy'],
            'deployed': deployed,
            'best_accuracy': self.best_accuracy
        }
        
        # Load existing history
        history = []
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                history = json.load(f)
        
        # Append new iteration
        history.append(iteration_data)
        
        # Save updated history
        with open(metrics_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Logged metrics to {metrics_file}")
    
    async def run_iteration(self) -> bool:
        """
        Run one iteration of the continuous learning loop.
        
        Returns:
            True if iteration successful, False otherwise
        """
        self.iteration += 1
        logger.info("=" * 80)
        logger.info(f"CONTINUOUS LEARNING - ITERATION {self.iteration}")
        logger.info("=" * 80)
        
        # Step 1: Generate new games
        num_games = self.config['game_generation']['games_per_iteration']
        if not await self.generate_games(num_games):
            logger.error("Game generation failed - skipping iteration")
            return False
        
        # Step 2: Export dataset
        dataset_path = await self.export_dataset()
        if not dataset_path:
            logger.error("Dataset export failed - skipping iteration")
            return False
        
        # Step 3: Train new model
        results = await self.train_model(dataset_path)
        if not results:
            logger.error("Model training failed - skipping iteration")
            return False
        
        # Step 4: Evaluate improvement
        should_deploy = await self.evaluate_improvement(
            results, 
            self.baseline_model_path
        )
        
        # Step 5: Deploy if improved (or if auto-deploy disabled, prompt user)
        deployed = False
        if should_deploy:
            if self.config['deployment']['auto_deploy']:
                deployed = await self.deploy_model(
                    str(results['model_path']),
                    backup=self.config['deployment']['backup_previous']
                )
            else:
                logger.info("Auto-deploy disabled - manual approval required")
                logger.info(f"New model path: {results['model_path']}")
                logger.info(f"To deploy: cp {results['model_path']} models/production/")
        
        # Step 6: Log metrics
        self.log_iteration_metrics(results, deployed)
        
        logger.info("=" * 80)
        logger.info(f"ITERATION {self.iteration} COMPLETE")
        logger.info(f"Status: {'DEPLOYED' if deployed else 'NOT DEPLOYED'}")
        logger.info("=" * 80)
        
        return True
    
    async def run_continuous(self):
        """Run continuous learning loop indefinitely or for max_iterations"""
        logger.info("Starting continuous learning system...")
        
        max_iterations = self.config['scheduling'].get('max_iterations')
        interval_hours = self.config['scheduling']['interval_hours']
        
        iteration_count = 0
        
        while True:
            # Run iteration
            success = await self.run_iteration()
            
            if success:
                iteration_count += 1
            
            # Check if we've reached max iterations
            if max_iterations and iteration_count >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations}) - stopping")
                break
            
            # Wait for next iteration
            logger.info(f"Waiting {interval_hours} hours until next iteration...")
            await asyncio.sleep(interval_hours * 3600)


def create_default_config():
    """Create default configuration file"""
    config = {
        'game_generation': {
            'games_per_iteration': 100,
            'orchestration_dir': '~/KDGY3/Integration5/orchestration',
            'generation_script': 'scripts/generate_self_play_data.py'
        },
        'dataset_export': {
            'logger_api_url': 'http://localhost:8010',
            'export_endpoint': '/dataset/export',
            'min_new_games': 50
        },
        'model_training': {
            'model_type': 'xgboost',
            'test_size': 0.2,
            'val_size': 0.1,
            'models_dir': 'models',
            'min_accuracy_improvement': 0.01
        },
        'deployment': {
            'ml_api_url': 'http://localhost:8000',
            'auto_deploy': False,
            'backup_previous': True,
            'restart_api': False
        },
        'scheduling': {
            'mode': 'interval',
            'interval_hours': 24,
            'max_iterations': None
        },
        'monitoring': {
            'log_dir': 'logs',
            'metrics_file': 'learning_history.json'
        }
    }
    
    with open('learning_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Created default configuration: learning_config.yaml")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Learning Orchestrator')
    parser.add_argument('--config', default='learning_config.yaml', help='Config file path')
    parser.add_argument('--create-config', action='store_true', help='Create default config')
    parser.add_argument('--once', action='store_true', help='Run one iteration only')
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
        return
    
    orchestrator = ContinuousLearningOrchestrator(args.config)
    
    if args.once:
        # Run single iteration
        await orchestrator.run_iteration()
    else:
        # Run continuously
        await orchestrator.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())
