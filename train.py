"""
Training Script for Cryptocurrency Trading Bot
Orchestrates two-phase training: Imitation Learning followed by Reinforcement Learning
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import *
from data_manager import DataManager
from sentiment_analyzer import SentimentAnalyzer
from expert import ExpertTrader, validate_expert_strategy
from trading_env import CryptoTradingEnv
from agent import DuelingDDQNAgent, create_expert_dataset

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Manages the complete training pipeline for the cryptocurrency trading bot.
    """
    
    def __init__(self, config_override=None):
        """
        Initialize the training manager.
        
        Args:
            config_override: Dictionary to override default config values
        """
        self.config = config_override or {}
        
        # Initialize components
        self.data_manager = None
        self.sentiment_analyzer = None
        self.expert_trader = None
        self.env = None
        self.agent = None
        
        # Training data
        self.historical_data = None
        self.train_data = None
        self.test_data = None
        
        # Training metrics
        self.training_metrics = {
            'il_losses': [],
            'rl_rewards': [],
            'rl_episode_rewards': [],
            'rl_episode_lengths': [],
            'performance_metrics': [],
            'expert_performance': {}
        }
        
        logger.info("TrainingManager initialized")
    
    def setup_components(self):
        """Initialize all required components."""
        try:
            logger.info("Setting up components...")
            
            # Initialize data manager
            self.data_manager = DataManager(
                exchange_name='binance',
                use_testnet=True
            )
            
            # Initialize sentiment analyzer
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("Sentiment analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize sentiment analyzer: {e}")
                self.sentiment_analyzer = None
            
            # Initialize expert trader
            self.expert_trader = ExpertTrader()
            
            logger.info("All components setup successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            raise
    
    def prepare_data(self):
        """Prepare training and testing data."""
        try:
            logger.info("Preparing training data...")
            
            # Fetch historical data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=HISTORICAL_DATA_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
            
            self.historical_data = self.data_manager.fetch_historical_data(
                symbol=CRYPTO_PAIR,
                timeframe=TIMEFRAME,
                start_date=start_date,
                end_date=end_date,
                save_to_file=True
            )
            
            # Calculate technical indicators
            self.historical_data = self.data_manager.calculate_technical_indicators(self.historical_data)
            
            # Check data quality
            if len(self.historical_data) < MIN_DATA_POINTS:
                raise ValueError(f"Insufficient data: {len(self.historical_data)} < {MIN_DATA_POINTS}")
            
            # Split data into train/test sets (80/20 split)
            split_idx = int(len(self.historical_data) * 0.8)
            self.train_data = self.historical_data.iloc[:split_idx].copy()
            self.test_data = self.historical_data.iloc[split_idx:].copy()
            
            logger.info(f"Data prepared successfully:")
            logger.info(f"  Total samples: {len(self.historical_data)}")
            logger.info(f"  Training samples: {len(self.train_data)}")
            logger.info(f"  Testing samples: {len(self.test_data)}")
            logger.info(f"  Date range: {self.historical_data.index[0]} to {self.historical_data.index[-1]}")
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise
    
    def validate_expert_strategy(self):
        """Validate expert trading strategy on historical data."""
        try:
            logger.info("Validating expert trading strategy...")
            
            # Validate on training data
            validation_results = validate_expert_strategy(self.train_data, self.expert_trader)
            self.training_metrics['expert_performance'] = validation_results
            
            logger.info("Expert strategy validation results:")
            logger.info(f"  Total trades: {validation_results['strategy_performance']['total_trades']}")
            logger.info(f"  Total return: {validation_results['strategy_performance']['total_return']:.2%}")
            logger.info(f"  Win rate: {validation_results['strategy_performance']['win_rate']:.2%}")
            logger.info(f"  Sharpe ratio: {validation_results['strategy_performance']['sharpe_ratio']:.3f}")
            logger.info(f"  Max drawdown: {validation_results['strategy_performance']['max_drawdown']:.2%}")
            logger.info(f"  Buy & Hold return: {validation_results['buy_hold_return']:.2%}")
            
            # Check if expert strategy is profitable
            if validation_results['strategy_performance']['total_return'] <= 0:
                logger.warning("Expert strategy shows negative returns. Consider adjusting parameters.")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate expert strategy: {e}")
            raise
    
    def phase_1_imitation_learning(self):
        """Phase 1: Train agent using imitation learning on expert demonstrations."""
        try:
            logger.info("=" * 60)
            logger.info("PHASE 1: IMITATION LEARNING")
            logger.info("=" * 60)
            
            # Create training environment
            self.env = CryptoTradingEnv(
                df=self.train_data,
                initial_balance=INITIAL_BALANCE,
                window_size=STATE_WINDOW_SIZE,
                transaction_fee=TRANSACTION_FEE_PERCENT
            )
            
            # Initialize agent
            state_size = self.env.observation_space.shape[0]
            action_size = self.env.action_space.n
            
            self.agent = DuelingDDQNAgent(
                state_size=state_size,
                action_size=action_size,
                seed=SEED
            )
            
            logger.info(f"Environment created - State size: {state_size}, Action size: {action_size}")
            
            # Generate expert dataset
            logger.info("Generating expert demonstrations...")
            expert_states, expert_actions = create_expert_dataset(
                self.env, self.expert_trader, num_episodes=1
            )
            
            # Validate expert dataset
            action_counts = np.bincount(expert_actions, minlength=action_size)
            logger.info(f"Expert action distribution: {dict(zip(ACTIONS.values(), action_counts))}")
            
            if len(expert_states) == 0:
                raise ValueError("No expert demonstrations generated")
            
            # Train agent on expert data
            logger.info(f"Training agent on {len(expert_states)} expert demonstrations...")
            start_time = time.time()
            
            self.agent.train_on_expert_data(
                expert_states=expert_states,
                expert_actions=expert_actions,
                epochs=IL_EPOCHS
            )
            
            training_time = time.time() - start_time
            logger.info(f"Imitation learning completed in {training_time:.2f} seconds")
            
            # Save pre-trained model
            self.agent.save(EXPERT_MODEL_PATH)
            logger.info(f"Pre-trained model saved to {EXPERT_MODEL_PATH}")
            
            # Test agent performance after imitation learning
            self._evaluate_agent_performance("post_imitation_learning")
            
            return True
            
        except Exception as e:
            logger.error(f"Imitation learning failed: {e}")
            raise
    
    def phase_2_reinforcement_learning(self):
        """Phase 2: Fine-tune agent using reinforcement learning."""
        try:
            logger.info("=" * 60)
            logger.info("PHASE 2: REINFORCEMENT LEARNING")
            logger.info("=" * 60)
            
            # Reset environment for RL training
            self.env.reset()
            
            # Training loop
            total_timesteps = 0
            episode = 0
            start_time = time.time()
            
            logger.info(f"Starting RL training for {RL_TOTAL_TIMESTEPS} timesteps...")
            
            while total_timesteps < RL_TOTAL_TIMESTEPS:
                episode += 1
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_start_time = time.time()
                
                while True:
                    # Select action
                    action = self.agent.act(state)
                    
                    # Take action in environment
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store experience and learn
                    if total_timesteps >= RL_WARMUP_STEPS:
                        self.agent.step(state, action, reward, next_state, done)
                    
                    # Update state and counters
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    total_timesteps += 1
                    
                    # Check if episode is done or max timesteps reached
                    if done or total_timesteps >= RL_TOTAL_TIMESTEPS:
                        break
                
                # Record episode metrics
                episode_time = time.time() - episode_start_time
                self.training_metrics['rl_episode_rewards'].append(episode_reward)
                self.training_metrics['rl_episode_lengths'].append(episode_length)
                
                # Logging
                if episode % 100 == 0:
                    avg_reward = np.mean(self.training_metrics['rl_episode_rewards'][-100:])
                    avg_length = np.mean(self.training_metrics['rl_episode_lengths'][-100:])
                    training_stats = self.agent.get_training_stats()
                    
                    logger.info(f"Episode {episode:6d} | "
                              f"Timesteps: {total_timesteps:8d} | "
                              f"Avg Reward: {avg_reward:8.2f} | "
                              f"Avg Length: {avg_length:6.1f} | "
                              f"Epsilon: {training_stats['epsilon']:.3f} | "
                              f"Buffer: {training_stats['buffer_size']:6d}")
                
                # Periodic evaluation
                if episode % (RL_EVALUATION_FREQUENCY // 100) == 0:
                    self._evaluate_agent_performance(f"episode_{episode}")
                
                # Save model checkpoint
                if total_timesteps % SAVE_MODEL_FREQUENCY == 0:
                    checkpoint_path = MODEL_DIR / f"checkpoint_{total_timesteps}.pth"
                    self.agent.save(checkpoint_path)
                    logger.info(f"Model checkpoint saved: {checkpoint_path}")
            
            total_training_time = time.time() - start_time
            logger.info(f"RL training completed in {total_training_time:.2f} seconds")
            logger.info(f"Total episodes: {episode}")
            logger.info(f"Average episode reward: {np.mean(self.training_metrics['rl_episode_rewards']):.2f}")
            
            # Save final model
            self.agent.save(FINAL_MODEL_PATH)
            logger.info(f"Final model saved to {FINAL_MODEL_PATH}")
            
            return True
            
        except Exception as e:
            logger.error(f"Reinforcement learning failed: {e}")
            raise
    
    def _evaluate_agent_performance(self, phase_name):
        """Evaluate agent performance on test data."""
        try:
            # Create test environment
            test_env = CryptoTradingEnv(
                df=self.test_data,
                initial_balance=INITIAL_BALANCE,
                window_size=STATE_WINDOW_SIZE,
                transaction_fee=TRANSACTION_FEE_PERCENT
            )
            
            # Run evaluation episode
            state = test_env.reset()
            total_reward = 0
            episode_length = 0
            
            while True:
                # Use greedy policy (no exploration)
                action = self.agent.act(state, epsilon=0.0)
                next_state, reward, done, info = test_env.step(action)
                
                state = next_state
                total_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Get performance metrics
            performance = test_env.get_performance_metrics()
            performance['phase'] = phase_name
            performance['total_reward'] = total_reward
            performance['episode_length'] = episode_length
            performance['timestamp'] = datetime.now()
            
            self.training_metrics['performance_metrics'].append(performance)
            
            logger.info(f"Evaluation ({phase_name}):")
            logger.info(f"  Total reward: {total_reward:.2f}")
            logger.info(f"  Episode length: {episode_length}")
            logger.info(f"  Net worth: ${performance.get('net_worth', 0):.2f}")
            logger.info(f"  Total return: {performance.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe ratio: {performance.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max drawdown: {performance.get('max_drawdown', 0):.2%}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    
    def generate_training_report(self):
        """Generate comprehensive training report with visualizations."""
        try:
            logger.info("Generating training report...")
            
            # Create results directory
            report_dir = RESULTS_DIR / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report_dir.mkdir(exist_ok=True)
            
            # Save training metrics
            metrics_file = report_dir / "training_metrics.pkl"
            pd.to_pickle(self.training_metrics, metrics_file)
            
            # Generate performance plots
            self._create_performance_plots(report_dir)
            
            # Generate summary report
            self._create_summary_report(report_dir)
            
            logger.info(f"Training report generated in {report_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate training report: {e}")
    
    def _create_performance_plots(self, report_dir):
        """Create performance visualization plots."""
        try:
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
            
            # Plot 1: RL Episode Rewards
            if self.training_metrics['rl_episode_rewards']:
                plt.figure(figsize=(12, 6))
                rewards = self.training_metrics['rl_episode_rewards']
                episodes = range(1, len(rewards) + 1)
                
                plt.subplot(1, 2, 1)
                plt.plot(episodes, rewards, alpha=0.6, color='blue')
                
                # Add moving average
                if len(rewards) > 100:
                    moving_avg = pd.Series(rewards).rolling(window=100).mean()
                    plt.plot(episodes, moving_avg, color='red', linewidth=2, label='100-episode MA')
                    plt.legend()
                
                plt.title('RL Training: Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.grid(True, alpha=0.3)
                
                # Plot 2: Episode Lengths
                plt.subplot(1, 2, 2)
                lengths = self.training_metrics['rl_episode_lengths']
                plt.plot(episodes, lengths, alpha=0.6, color='green')
                
                if len(lengths) > 100:
                    moving_avg = pd.Series(lengths).rolling(window=100).mean()
                    plt.plot(episodes, moving_avg, color='red', linewidth=2, label='100-episode MA')
                    plt.legend()
                
                plt.title('RL Training: Episode Lengths')
                plt.xlabel('Episode')
                plt.ylabel('Episode Length')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(report_dir / 'rl_training_progress.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 2: Performance Comparison
            if self.training_metrics['performance_metrics']:
                perf_df = pd.DataFrame(self.training_metrics['performance_metrics'])
                
                plt.figure(figsize=(15, 10))
                
                # Net worth progression
                plt.subplot(2, 3, 1)
                for phase in perf_df['phase'].unique():
                    phase_data = perf_df[perf_df['phase'] == phase]
                    plt.scatter(range(len(phase_data)), phase_data['net_worth'], 
                              label=phase, alpha=0.7, s=50)
                plt.title('Net Worth Progression')
                plt.xlabel('Evaluation Point')
                plt.ylabel('Net Worth ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Total return
                plt.subplot(2, 3, 2)
                plt.bar(range(len(perf_df)), perf_df['total_return'], 
                       color='skyblue', alpha=0.7)
                plt.title('Total Return by Evaluation')
                plt.xlabel('Evaluation Point')
                plt.ylabel('Total Return')
                plt.xticks(range(len(perf_df)), perf_df['phase'], rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Sharpe ratio
                plt.subplot(2, 3, 3)
                plt.plot(range(len(perf_df)), perf_df['sharpe_ratio'], 
                        marker='o', linewidth=2, markersize=6)
                plt.title('Sharpe Ratio Progression')
                plt.xlabel('Evaluation Point')
                plt.ylabel('Sharpe Ratio')
                plt.grid(True, alpha=0.3)
                
                # Max drawdown
                plt.subplot(2, 3, 4)
                plt.bar(range(len(perf_df)), perf_df['max_drawdown'], 
                       color='salmon', alpha=0.7)
                plt.title('Maximum Drawdown')
                plt.xlabel('Evaluation Point')
                plt.ylabel('Max Drawdown')
                plt.xticks(range(len(perf_df)), perf_df['phase'], rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Win rate
                plt.subplot(2, 3, 5)
                plt.plot(range(len(perf_df)), perf_df['win_rate'], 
                        marker='s', linewidth=2, markersize=6, color='green')
                plt.title('Win Rate Progression')
                plt.xlabel('Evaluation Point')
                plt.ylabel('Win Rate')
                plt.grid(True, alpha=0.3)
                
                # Total trades
                plt.subplot(2, 3, 6)
                plt.bar(range(len(perf_df)), perf_df['total_trades'], 
                       color='orange', alpha=0.7)
                plt.title('Total Trades')
                plt.xlabel('Evaluation Point')
                plt.ylabel('Number of Trades')
                plt.xticks(range(len(perf_df)), perf_df['phase'], rotation=45)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(report_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create performance plots: {e}")
    
    def _create_summary_report(self, report_dir):
        """Create summary report text file."""
        try:
            report_file = report_dir / "training_summary.txt"
            
            with open(report_file, 'w') as f:
                f.write("CRYPTOCURRENCY TRADING BOT - TRAINING SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Cryptocurrency Pair: {CRYPTO_PAIR}\n")
                f.write(f"Timeframe: {TIMEFRAME}\n")
                f.write(f"Initial Balance: ${INITIAL_BALANCE:,.2f}\n\n")
                
                # Data summary
                f.write("DATA SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Data Points: {len(self.historical_data):,}\n")
                f.write(f"Training Data Points: {len(self.train_data):,}\n")
                f.write(f"Testing Data Points: {len(self.test_data):,}\n")
                f.write(f"Data Range: {self.historical_data.index[0]} to {self.historical_data.index[-1]}\n\n")
                
                # Expert strategy performance
                if self.training_metrics['expert_performance']:
                    expert_perf = self.training_metrics['expert_performance']['strategy_performance']
                    f.write("EXPERT STRATEGY PERFORMANCE\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Total Trades: {expert_perf['total_trades']}\n")
                    f.write(f"Total Return: {expert_perf['total_return']:.2%}\n")
                    f.write(f"Win Rate: {expert_perf['win_rate']:.2%}\n")
                    f.write(f"Sharpe Ratio: {expert_perf['sharpe_ratio']:.3f}\n")
                    f.write(f"Max Drawdown: {expert_perf['max_drawdown']:.2%}\n\n")
                
                # RL training summary
                if self.training_metrics['rl_episode_rewards']:
                    f.write("REINFORCEMENT LEARNING SUMMARY\n")
                    f.write("-" * 35 + "\n")
                    f.write(f"Total Episodes: {len(self.training_metrics['rl_episode_rewards'])}\n")
                    f.write(f"Average Episode Reward: {np.mean(self.training_metrics['rl_episode_rewards']):.2f}\n")
                    f.write(f"Best Episode Reward: {np.max(self.training_metrics['rl_episode_rewards']):.2f}\n")
                    f.write(f"Final Episode Reward: {self.training_metrics['rl_episode_rewards'][-1]:.2f}\n\n")
                
                # Final performance
                if self.training_metrics['performance_metrics']:
                    final_perf = self.training_metrics['performance_metrics'][-1]
                    f.write("FINAL AGENT PERFORMANCE\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Final Net Worth: ${final_perf.get('net_worth', 0):,.2f}\n")
                    f.write(f"Total Return: {final_perf.get('total_return', 0):.2%}\n")
                    f.write(f"Sharpe Ratio: {final_perf.get('sharpe_ratio', 0):.3f}\n")
                    f.write(f"Max Drawdown: {final_perf.get('max_drawdown', 0):.2%}\n")
                    f.write(f"Win Rate: {final_perf.get('win_rate', 0):.2%}\n")
                    f.write(f"Total Trades: {final_perf.get('total_trades', 0)}\n\n")
                
                # Configuration
                f.write("CONFIGURATION PARAMETERS\n")
                f.write("-" * 25 + "\n")
                f.write(f"State Window Size: {STATE_WINDOW_SIZE}\n")
                f.write(f"IL Epochs: {IL_EPOCHS}\n")
                f.write(f"RL Total Timesteps: {RL_TOTAL_TIMESTEPS:,}\n")
                f.write(f"Learning Rate: {LEARNING_RATE}\n")
                f.write(f"Gamma: {GAMMA}\n")
                f.write(f"Replay Buffer Size: {REPLAY_BUFFER_SIZE:,}\n")
                f.write(f"Batch Size: {BATCH_SIZE}\n")
                
        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
    
    def run_complete_training(self):
        """Run the complete two-phase training pipeline."""
        try:
            logger.info("Starting complete training pipeline...")
            start_time = time.time()
            
            # Setup
            self.setup_components()
            self.prepare_data()
            
            # Validate expert strategy
            self.validate_expert_strategy()
            
            # Phase 1: Imitation Learning
            self.phase_1_imitation_learning()
            
            # Phase 2: Reinforcement Learning
            self.phase_2_reinforcement_learning()
            
            # Generate report
            self.generate_training_report()
            
            total_time = time.time() - start_time
            logger.info(f"Complete training pipeline finished in {total_time:.2f} seconds")
            logger.info("Training completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Cryptocurrency Trading Bot")
    parser.add_argument('--skip-il', action='store_true', 
                       help='Skip imitation learning phase')
    parser.add_argument('--skip-rl', action='store_true', 
                       help='Skip reinforcement learning phase')
    parser.add_argument('--load-model', type=str, 
                       help='Path to pre-trained model to load')
    parser.add_argument('--config', type=str, 
                       help='Path to custom config file')
    
    args = parser.parse_args()
    
    try:
        # Initialize training manager
        trainer = TrainingManager()
        
        if args.skip_il and args.skip_rl:
            logger.error("Cannot skip both training phases")
            return
        
        if args.load_model and Path(args.load_model).exists():
            logger.info(f"Loading pre-trained model from {args.load_model}")
            # Load model logic would go here
        
        # Run training based on arguments
        if not args.skip_il and not args.skip_rl:
            # Run complete training
            trainer.run_complete_training()
        else:
            # Run partial training
            trainer.setup_components()
            trainer.prepare_data()
            
            if not args.skip_il:
                trainer.validate_expert_strategy()
                trainer.phase_1_imitation_learning()
            
            if not args.skip_rl:
                trainer.phase_2_reinforcement_learning()
            
            trainer.generate_training_report()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()