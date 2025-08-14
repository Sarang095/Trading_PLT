"""
Cryptocurrency Trading Environment for Reinforcement Learning
Custom OpenAI Gym environment for training the RL trading agent.
"""

import gym
from gym import spaces
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import *

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class CryptoTradingEnv(gym.Env):
    """
    Custom cryptocurrency trading environment for reinforcement learning.
    
    State Space: Normalized OHLCV data + technical indicators + position info
    Action Space: Discrete(3) - {0: Hold, 1: Buy, 2: Sell}
    Reward: Multi-objective function based on profit, Sharpe ratio, and drawdown
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 initial_balance: float = INITIAL_BALANCE,
                 window_size: int = STATE_WINDOW_SIZE,
                 transaction_fee: float = TRANSACTION_FEE_PERCENT):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            initial_balance: Starting balance in quote currency
            window_size: Number of historical time steps for state
            transaction_fee: Transaction fee percentage
        """
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.transaction_fee = transaction_fee
        
        # Validate and prepare data
        self._validate_data()
        self._prepare_data()
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(self.df) - self.window_size - 1
        
        # Portfolio state
        self.balance = initial_balance  # Quote currency (USDT)
        self.crypto_held = 0.0  # Base currency (BTC)
        self.net_worth = initial_balance
        self.position = 0  # 0: No position, 1: Long position
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        
        # Trading history
        self.trade_history = []
        self.portfolio_history = []
        self.action_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.peak_net_worth = initial_balance
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._define_observation_space()
        
        logger.info(f"CryptoTradingEnv initialized with {len(self.df)} data points")
        logger.info(f"State window size: {self.window_size}, Max steps: {self.max_steps}")
    
    def _validate_data(self):
        """Validate input data format and completeness."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        if len(self.df) < self.window_size + 1:
            raise ValueError(f"DataFrame must have at least {self.window_size + 1} rows")
        
        # Check for missing values
        if self.df[required_columns].isnull().any().any():
            logger.warning("Found missing values in data, forward filling...")
            self.df[required_columns] = self.df[required_columns].fillna(method='ffill')
        
        logger.info("Data validation completed successfully")
    
    def _prepare_data(self):
        """Prepare data by calculating technical indicators and normalizing."""
        try:
            # Calculate technical indicators
            self.df = self._calculate_technical_indicators(self.df)
            
            # Store raw price data for trading calculations
            self.prices = self.df['close'].values
            
            # Define feature columns for state representation
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'ema_9', 'ema_21', 'ema_200', 'rsi_14', 
                'bb_percent_b_20_2', 'macd_hist', 'volume_sma_20'
            ]
            
            # Normalize features for neural network input
            self.normalized_data = self._normalize_features()
            
            logger.info("Data preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset."""
        # Exponential Moving Averages
        df['ema_9'] = df['close'].ewm(span=EMA_SHORT_PERIOD).mean()
        df['ema_21'] = df['close'].ewm(span=EMA_LONG_PERIOD).mean()
        df['ema_200'] = df['close'].ewm(span=EMA_TREND_PERIOD).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands %B
        bb_middle = df['close'].rolling(window=BB_PERIOD).mean()
        bb_std = df['close'].rolling(window=BB_PERIOD).std()
        bb_upper = bb_middle + (bb_std * BB_STD)
        bb_lower = bb_middle - (bb_std * BB_STD)
        df['bb_percent_b_20_2'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD Histogram
        ema_fast = df['close'].ewm(span=MACD_FAST).mean()
        ema_slow = df['close'].ewm(span=MACD_SLOW).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=MACD_SIGNAL).mean()
        df['macd_hist'] = macd_line - macd_signal
        
        # Volume SMA (normalized)
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        volume_norm = df['volume_sma_20'].rolling(window=50).mean()
        df['volume_sma_20'] = df['volume_sma_20'] / volume_norm
        
        # Fill NaN values with forward fill
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _normalize_features(self) -> np.ndarray:
        """Normalize features using rolling statistics for stationarity."""
        features = self.df[self.feature_columns].values
        normalized_features = np.zeros_like(features)
        
        # Use rolling normalization to maintain stationarity
        for i in range(len(features)):
            if i < self.window_size:
                # Use available data for early periods
                start_idx = 0
                end_idx = i + 1
            else:
                # Use rolling window
                start_idx = i - self.window_size + 1
                end_idx = i + 1
            
            window_data = features[start_idx:end_idx]
            
            # Calculate rolling statistics
            window_mean = np.mean(window_data, axis=0)
            window_std = np.std(window_data, axis=0)
            
            # Avoid division by zero
            window_std = np.where(window_std == 0, 1, window_std)
            
            # Normalize current observation
            normalized_features[i] = (features[i] - window_mean) / window_std
        
        return normalized_features
    
    def _define_observation_space(self):
        """Define the observation space for the environment."""
        # Features: OHLCV + technical indicators
        n_features = len(self.feature_columns)
        
        # Additional state information
        n_position_features = 2  # position type, unrealized P&L
        
        # Total observation size
        total_features = (n_features * self.window_size) + n_position_features
        
        # Define bounds (normalized features typically in [-3, 3] range)
        low = np.full(total_features, -10.0, dtype=np.float32)
        high = np.full(total_features, 10.0, dtype=np.float32)
        
        # Position features have specific bounds
        low[-2] = 0  # position: 0 or 1
        high[-2] = 1
        low[-1] = -1  # unrealized P&L: normalized
        high[-1] = 1
        
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        
        logger.info(f"Observation space defined: {self.observation_space.shape}")
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        try:
            # Get normalized features for the current window
            start_idx = self.current_step
            end_idx = self.current_step + self.window_size
            
            if end_idx > len(self.normalized_data):
                # Pad with last available data if needed
                window_data = self.normalized_data[start_idx:]
                padding = np.tile(self.normalized_data[-1], 
                                (end_idx - len(self.normalized_data), 1))
                window_data = np.vstack([window_data, padding])
            else:
                window_data = self.normalized_data[start_idx:end_idx]
            
            # Flatten the window data
            flattened_features = window_data.flatten()
            
            # Add position information
            position_features = np.array([
                float(self.position),  # Current position (0 or 1)
                self._normalize_pnl(self.unrealized_pnl)  # Normalized unrealized P&L
            ], dtype=np.float32)
            
            # Combine all features
            state = np.concatenate([flattened_features, position_features])
            
            return state.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            # Return zero state as fallback
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _normalize_pnl(self, pnl: float) -> float:
        """Normalize P&L for state representation."""
        # Clamp P&L to reasonable range and normalize
        clamped_pnl = np.clip(pnl, -0.5, 0.5)  # -50% to +50%
        return clamped_pnl * 2  # Scale to [-1, 1]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        try:
            # Validate action
            if action not in [0, 1, 2]:
                raise ValueError(f"Invalid action: {action}")
            
            # Store previous state for reward calculation
            prev_net_worth = self.net_worth
            prev_position = self.position
            
            # Execute action
            executed = self._execute_action(action)
            
            # Update portfolio state
            self._update_portfolio()
            
            # Calculate reward
            reward = self._calculate_reward(action, executed, prev_net_worth, prev_position)
            
            # Update step
            self.current_step += 1
            
            # Check if episode is done
            done = self._is_done()
            
            # Get next state
            next_state = self._get_state()
            
            # Store action history
            self.action_history.append(action)
            
            # Create info dictionary
            info = {
                'balance': self.balance,
                'crypto_held': self.crypto_held,
                'net_worth': self.net_worth,
                'position': self.position,
                'unrealized_pnl': self.unrealized_pnl,
                'total_trades': self.total_trades,
                'action_executed': executed,
                'current_price': self.prices[self.current_step + self.window_size - 1]
            }
            
            return next_state, reward, done, info
            
        except Exception as e:
            logger.error(f"Error in step function: {e}")
            # Return safe defaults
            return self._get_state(), 0.0, True, {'error': str(e)}
    
    def _execute_action(self, action: int) -> bool:
        """
        Execute trading action.
        
        Args:
            action: Action to execute
            
        Returns:
            True if action was executed, False otherwise
        """
        current_price = self.prices[self.current_step + self.window_size - 1]
        executed = False
        
        if action == 1:  # Buy
            if self.position == 0 and self.balance > MIN_TRADE_AMOUNT:
                # Calculate how much crypto we can buy
                effective_balance = self.balance * (1 - self.transaction_fee)
                max_crypto = effective_balance / current_price
                
                # Limit position size
                max_position_value = self.net_worth * MAX_POSITION_SIZE
                max_crypto_by_position = max_position_value / current_price
                
                crypto_to_buy = min(max_crypto, max_crypto_by_position)
                
                if crypto_to_buy > 0:
                    cost = crypto_to_buy * current_price
                    total_cost = cost * (1 + self.transaction_fee)
                    
                    if self.balance >= total_cost:
                        self.balance -= total_cost
                        self.crypto_held += crypto_to_buy
                        self.position = 1
                        self.entry_price = current_price
                        
                        # Record trade
                        trade_record = {
                            'timestamp': self.current_step,
                            'action': 'buy',
                            'price': current_price,
                            'quantity': crypto_to_buy,
                            'cost': total_cost,
                            'balance': self.balance,
                            'crypto_held': self.crypto_held
                        }
                        self.trade_history.append(trade_record)
                        self.total_trades += 1
                        executed = True
        
        elif action == 2:  # Sell
            if self.position == 1 and self.crypto_held > 0:
                # Sell all crypto
                proceeds = self.crypto_held * current_price
                total_proceeds = proceeds * (1 - self.transaction_fee)
                
                # Calculate realized P&L
                cost_basis = self.crypto_held * self.entry_price
                realized_pnl = total_proceeds - cost_basis
                
                if realized_pnl > 0:
                    self.winning_trades += 1
                
                self.total_profit += realized_pnl
                self.balance += total_proceeds
                
                # Record trade
                trade_record = {
                    'timestamp': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'quantity': self.crypto_held,
                    'proceeds': total_proceeds,
                    'realized_pnl': realized_pnl,
                    'balance': self.balance,
                    'crypto_held': 0
                }
                self.trade_history.append(trade_record)
                
                # Reset position
                self.crypto_held = 0
                self.position = 0
                self.entry_price = 0
                self.unrealized_pnl = 0
                executed = True
        
        # action == 0 (Hold) requires no execution
        
        return executed
    
    def _update_portfolio(self):
        """Update portfolio valuation and metrics."""
        current_price = self.prices[self.current_step + self.window_size - 1]
        
        # Calculate current net worth
        crypto_value = self.crypto_held * current_price
        self.net_worth = self.balance + crypto_value
        
        # Update unrealized P&L if in position
        if self.position == 1 and self.crypto_held > 0:
            cost_basis = self.crypto_held * self.entry_price
            self.unrealized_pnl = (crypto_value - cost_basis) / cost_basis
        else:
            self.unrealized_pnl = 0
        
        # Update maximum drawdown
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)
        current_drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Store portfolio history
        portfolio_record = {
            'timestamp': self.current_step,
            'price': current_price,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'net_worth': self.net_worth,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'drawdown': current_drawdown
        }
        self.portfolio_history.append(portfolio_record)
    
    def _calculate_reward(self, action: int, executed: bool, 
                         prev_net_worth: float, prev_position: int) -> float:
        """
        Calculate multi-objective reward function.
        
        Args:
            action: Action taken
            executed: Whether action was executed
            prev_net_worth: Previous net worth
            prev_position: Previous position
            
        Returns:
            Calculated reward
        """
        try:
            reward = 0.0
            
            # 1. Profit/Loss Component
            portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth
            profit_reward = portfolio_return * REWARD_WEIGHTS['profit']
            
            # 2. Sharpe Ratio Component (if enough history)
            sharpe_reward = 0.0
            if len(self.portfolio_history) >= SHARPE_LOOKBACK_WINDOW:
                returns = []
                for i in range(-SHARPE_LOOKBACK_WINDOW, 0):
                    if i == -SHARPE_LOOKBACK_WINDOW:
                        prev_worth = self.initial_balance
                    else:
                        prev_worth = self.portfolio_history[i-1]['net_worth']
                    
                    current_worth = self.portfolio_history[i]['net_worth']
                    period_return = (current_worth - prev_worth) / prev_worth
                    returns.append(period_return)
                
                if len(returns) > 1:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        sharpe_ratio = mean_return / std_return
                        sharpe_reward = sharpe_ratio * REWARD_WEIGHTS['sharpe']
            
            # 3. Drawdown Penalty
            drawdown_penalty = self.max_drawdown * REWARD_WEIGHTS['drawdown']
            
            # 4. Transaction Cost Penalty
            transaction_penalty = 0.0
            if executed and action in [1, 2]:
                transaction_penalty = self.transaction_fee * REWARD_WEIGHTS['transaction_cost']
            
            # 5. Position Holding Bonus (encourage holding profitable positions)
            hold_bonus = 0.0
            if self.position == 1 and self.unrealized_pnl > 0:
                hold_bonus = min(self.unrealized_pnl, 0.05) * REWARD_WEIGHTS['hold_time']
            
            # Combine all reward components
            reward = (profit_reward + sharpe_reward + drawdown_penalty + 
                     transaction_penalty + hold_bonus)
            
            # Scale reward
            reward *= REWARD_SCALING_FACTOR
            
            # Clip reward to prevent extreme values
            reward = np.clip(reward, -10.0, 10.0)
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is finished."""
        # Episode ends if we've reached the end of data
        if self.current_step >= self.max_steps:
            return True
        
        # Episode ends if we've lost too much money
        if self.net_worth <= self.initial_balance * (1 - MAX_DRAWDOWN_PERCENT):
            logger.warning("Episode ended due to excessive drawdown")
            return True
        
        return False
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Reset step counter
        self.current_step = 0
        
        # Reset portfolio
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.net_worth = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        
        # Reset metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.peak_net_worth = self.initial_balance
        
        # Clear history
        self.trade_history = []
        self.portfolio_history = []
        self.action_history = []
        
        logger.info("Environment reset successfully")
        
        return self._get_state()
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            current_price = self.prices[self.current_step + self.window_size - 1]
            print(f"\nStep: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Crypto Held: {self.crypto_held:.6f}")
            print(f"Net Worth: ${self.net_worth:.2f}")
            print(f"Position: {ACTIONS[self.position]}")
            print(f"Unrealized P&L: {self.unrealized_pnl:.2%}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Max Drawdown: {self.max_drawdown:.2%}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            if not self.portfolio_history:
                return {}
            
            # Calculate returns
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_worth = self.portfolio_history[i-1]['net_worth']
                current_worth = self.portfolio_history[i]['net_worth']
                period_return = (current_worth - prev_worth) / prev_worth
                returns.append(period_return)
            
            if not returns:
                return {}
            
            # Performance metrics
            total_return = (self.net_worth - self.initial_balance) / self.initial_balance
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            # Sharpe ratio
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Additional metrics
            max_net_worth = max([p['net_worth'] for p in self.portfolio_history])
            min_net_worth = min([p['net_worth'] for p in self.portfolio_history])
            
            return {
                'total_return': total_return,
                'total_profit': self.total_profit,
                'net_worth': self.net_worth,
                'max_drawdown': self.max_drawdown,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_net_worth': max_net_worth,
                'min_net_worth': min_net_worth,
                'volatility': std_return,
                'mean_return': mean_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def save_history(self, filename_prefix: str):
        """Save trading and portfolio history to files."""
        try:
            # Save trade history
            if self.trade_history:
                trade_df = pd.DataFrame(self.trade_history)
                trade_file = DATA_DIR / f"{filename_prefix}_trades.csv"
                trade_df.to_csv(trade_file, index=False)
                logger.info(f"Trade history saved to {trade_file}")
            
            # Save portfolio history
            if self.portfolio_history:
                portfolio_df = pd.DataFrame(self.portfolio_history)
                portfolio_file = DATA_DIR / f"{filename_prefix}_portfolio.csv"
                portfolio_df.to_csv(portfolio_file, index=False)
                logger.info(f"Portfolio history saved to {portfolio_file}")
                
        except Exception as e:
            logger.error(f"Error saving history: {e}")


if __name__ == "__main__":
    # Example usage and testing
    try:
        # Generate sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='h')
        np.random.seed(42)
        
        # Create realistic price data
        n_periods = len(dates)
        returns = np.random.normal(0.0001, 0.02, n_periods)
        prices = 30000 * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        test_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(100, 1000)
            
            test_data.append({
                'timestamp': date,
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(test_data)
        df.set_index('timestamp', inplace=True)
        
        print(f"Generated test data: {len(df)} periods")
        
        # Initialize environment
        env = CryptoTradingEnv(df)
        
        print(f"Environment initialized successfully")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test environment
        state = env.reset()
        print(f"Initial state shape: {state.shape}")
        
        # Run a few random steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step}: Action={ACTIONS[action]}, Reward={reward:.4f}, "
                  f"Net Worth=${info['net_worth']:.2f}")
            
            if done:
                break
        
        print(f"\nTotal reward: {total_reward:.4f}")
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise