"""
Live Trading Script for Cryptocurrency Trading Bot
Executes trades in real-time using the trained agent and live market data.
"""

import os
import sys
import time
import signal
import argparse
import logging
import threading
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import json
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import *
from data_manager import DataManager
from sentiment_analyzer import SentimentAnalyzer
from agent import DuelingDDQNAgent
import ccxt

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(TRADING_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live cryptocurrency trading bot using trained RL agent.
    """
    
    def __init__(self, model_path: str, use_testnet: bool = True):
        """
        Initialize the live trader.
        
        Args:
            model_path: Path to trained model
            use_testnet: Whether to use testnet/paper trading
        """
        self.model_path = model_path
        self.use_testnet = use_testnet
        
        # Components
        self.data_manager = None
        self.sentiment_analyzer = None
        self.agent = None
        self.exchange = None
        
        # Market data
        self.market_data_buffer = deque(maxlen=STATE_WINDOW_SIZE + 100)
        self.last_kline_data = None
        self.current_state = None
        
        # Trading state
        self.is_trading = False
        self.current_position = 0  # 0: No position, 1: Long
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        
        # Safety and monitoring
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.session_start_time = datetime.now()
        self.performance_metrics = []
        
        # Event handling
        self.shutdown_event = threading.Event()
        
        logger.info(f"LiveTrader initialized - Model: {model_path}, Testnet: {use_testnet}")
    
    def setup_components(self):
        """Initialize all trading components."""
        try:
            logger.info("Setting up live trading components...")
            
            # Initialize data manager
            self.data_manager = DataManager(
                exchange_name='binance',
                use_testnet=self.use_testnet
            )
            
            # Initialize sentiment analyzer (optional)
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("Sentiment analyzer initialized")
            except Exception as e:
                logger.warning(f"Sentiment analyzer failed to initialize: {e}")
                self.sentiment_analyzer = None
            
            # Setup exchange for trading
            self.exchange = self.data_manager.exchange
            
            # Load trained agent
            self._load_trained_agent()
            
            # Fetch initial market data
            self._initialize_market_data()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            raise
    
    def _load_trained_agent(self):
        """Load the trained RL agent."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Calculate state size (must match training configuration)
            n_features = len(TECHNICAL_INDICATORS) + 5  # OHLCV + indicators
            state_size = (STATE_WINDOW_SIZE * n_features) + 2  # + position features
            action_size = N_ACTIONS
            
            # Initialize agent
            self.agent = DuelingDDQNAgent(
                state_size=state_size,
                action_size=action_size,
                seed=SEED
            )
            
            # Load trained weights
            self.agent.load(self.model_path)
            
            # Set to evaluation mode (no exploration)
            self.agent.epsilon = 0.0
            
            logger.info(f"Trained agent loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load trained agent: {e}")
            raise
    
    def _initialize_market_data(self):
        """Fetch initial historical data to populate buffer."""
        try:
            logger.info("Initializing market data buffer...")
            
            # Fetch recent historical data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            historical_data = self.data_manager.fetch_historical_data(
                symbol=CRYPTO_PAIR,
                timeframe=TIMEFRAME,
                start_date=start_date,
                end_date=end_date,
                save_to_file=False
            )
            
            # Calculate technical indicators
            historical_data = self.data_manager.calculate_technical_indicators(historical_data)
            
            # Populate buffer with recent data
            for _, row in historical_data.tail(STATE_WINDOW_SIZE + 50).iterrows():
                kline_data = {
                    'timestamp': row.name,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'is_closed': True
                }
                self.market_data_buffer.append(kline_data)
            
            logger.info(f"Market data buffer initialized with {len(self.market_data_buffer)} data points")
            
        except Exception as e:
            logger.error(f"Failed to initialize market data: {e}")
            raise
    
    def start_live_trading(self):
        """Start live trading with WebSocket data feed."""
        try:
            logger.info("Starting live trading...")
            self.is_trading = True
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start WebSocket data feed
            self.data_manager.start_websocket_feed(
                symbol=CRYPTO_PAIR,
                timeframe=TIMEFRAME,
                callback=self._on_market_data
            )
            
            # Start monitoring and decision making thread
            monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitoring_thread.start()
            
            # Main trading loop
            logger.info("Live trading started. Press Ctrl+C to stop.")
            
            while self.is_trading and not self.shutdown_event.is_set():
                try:
                    # Check safety conditions
                    if not self._safety_checks():
                        logger.warning("Safety checks failed, pausing trading")
                        time.sleep(60)
                        continue
                    
                    # Generate current state
                    if self._update_current_state():
                        # Get action from agent
                        action = self.agent.act(self.current_state, epsilon=0.0)
                        
                        # Execute action if valid
                        self._execute_trading_action(action)
                    
                    # Wait before next decision
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(30)  # Wait before retrying
            
            logger.info("Live trading stopped")
            
        except Exception as e:
            logger.error(f"Live trading failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _on_market_data(self, data):
        """Handle incoming market data from WebSocket."""
        try:
            # Only process closed candles for decision making
            if not data.get('is_closed', False):
                return
            
            # Add to buffer
            self.market_data_buffer.append(data)
            self.last_kline_data = data
            
            # Log market data periodically
            if len(self.market_data_buffer) % 10 == 0:
                logger.debug(f"Market data: {data['symbol']} - "
                           f"Price: ${data['close']:.2f}, "
                           f"Volume: {data['volume']:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _update_current_state(self):
        """Update current state from market data buffer."""
        try:
            if len(self.market_data_buffer) < STATE_WINDOW_SIZE:
                return False
            
            # Convert buffer to DataFrame
            recent_data = list(self.market_data_buffer)[-STATE_WINDOW_SIZE:]
            df = pd.DataFrame(recent_data)
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self.data_manager.calculate_technical_indicators(df)
            
            # Extract features
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'ema_9', 'ema_21', 'ema_200', 'rsi_14',
                'bb_percent_b_20_2', 'macd_hist', 'volume_sma_20'
            ]
            
            # Normalize features (simple z-score normalization)
            features = df[feature_columns].values
            normalized_features = []
            
            for i in range(len(features)):
                if i < 10:  # Use available data for early periods
                    window_data = features[:i+1]
                else:
                    window_data = features[i-9:i+1]
                
                mean = np.mean(window_data, axis=0)
                std = np.std(window_data, axis=0)
                std = np.where(std == 0, 1, std)  # Avoid division by zero
                
                normalized = (features[i] - mean) / std
                normalized_features.append(normalized)
            
            # Flatten normalized features
            flattened_features = np.array(normalized_features).flatten()
            
            # Add position information
            position_features = np.array([
                float(self.current_position),
                self._get_unrealized_pnl()
            ])
            
            # Combine all features
            self.current_state = np.concatenate([flattened_features, position_features])
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating current state: {e}")
            return False
    
    def _execute_trading_action(self, action):
        """Execute trading action based on agent decision."""
        try:
            action_name = ACTIONS[action]
            current_price = self.last_kline_data['close']
            
            # Implement action logic
            if action == 1 and self.current_position == 0:  # Buy
                self._execute_buy_order(current_price)
            elif action == 2 and self.current_position == 1:  # Sell
                self._execute_sell_order(current_price)
            # action == 0 (Hold) requires no execution
            
        except Exception as e:
            logger.error(f"Error executing trading action: {e}")
    
    def _execute_buy_order(self, price):
        """Execute buy order."""
        try:
            # Check if we can place a buy order
            if self.current_position != 0:
                return
            
            # Calculate position size
            account_balance = self.data_manager.get_account_balance()
            available_balance = account_balance.get(QUOTE_CURRENCY, 0)
            
            if available_balance < MIN_TRADE_AMOUNT:
                logger.warning(f"Insufficient balance for trade: ${available_balance:.2f}")
                return
            
            # Calculate order size (use percentage of available balance)
            order_value = min(available_balance * MAX_POSITION_SIZE, available_balance * 0.95)
            order_quantity = order_value / price
            
            if self.use_testnet:
                # Simulate order execution
                logger.info(f"SIMULATED BUY: {order_quantity:.6f} {BASE_CURRENCY} at ${price:.2f}")
                execution_successful = True
            else:
                # Execute real order
                try:
                    order = self.exchange.create_market_buy_order(
                        symbol=CRYPTO_PAIR,
                        amount=order_quantity
                    )
                    logger.info(f"BUY ORDER EXECUTED: {order}")
                    execution_successful = True
                except Exception as e:
                    logger.error(f"Failed to execute buy order: {e}")
                    execution_successful = False
            
            if execution_successful:
                # Update position
                self.current_position = 1
                self.position_size = order_quantity
                self.entry_price = price
                self.entry_time = datetime.now()
                self.total_trades += 1
                self.daily_trades += 1
                self.last_trade_time = datetime.now()
                
                # Log trade
                trade_record = {
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'price': price,
                    'quantity': order_quantity,
                    'value': order_value,
                    'position': self.current_position
                }
                self._log_trade(trade_record)
                
                logger.info(f"Buy order executed: {order_quantity:.6f} {BASE_CURRENCY} at ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
    
    def _execute_sell_order(self, price):
        """Execute sell order."""
        try:
            # Check if we have position to sell
            if self.current_position != 1 or self.position_size <= 0:
                return
            
            if self.use_testnet:
                # Simulate order execution
                logger.info(f"SIMULATED SELL: {self.position_size:.6f} {BASE_CURRENCY} at ${price:.2f}")
                execution_successful = True
                proceeds = self.position_size * price
            else:
                # Execute real order
                try:
                    order = self.exchange.create_market_sell_order(
                        symbol=CRYPTO_PAIR,
                        amount=self.position_size
                    )
                    logger.info(f"SELL ORDER EXECUTED: {order}")
                    proceeds = order.get('cost', self.position_size * price)
                    execution_successful = True
                except Exception as e:
                    logger.error(f"Failed to execute sell order: {e}")
                    execution_successful = False
                    proceeds = 0
            
            if execution_successful:
                # Calculate P&L
                cost_basis = self.position_size * self.entry_price
                realized_pnl = proceeds - cost_basis
                realized_pnl_pct = (realized_pnl / cost_basis) * 100
                
                # Update tracking
                if realized_pnl > 0:
                    self.successful_trades += 1
                
                self.total_pnl += realized_pnl
                self.daily_pnl += realized_pnl
                
                # Log trade
                trade_record = {
                    'timestamp': datetime.now(),
                    'action': 'SELL',
                    'price': price,
                    'quantity': self.position_size,
                    'proceeds': proceeds,
                    'entry_price': self.entry_price,
                    'realized_pnl': realized_pnl,
                    'realized_pnl_pct': realized_pnl_pct,
                    'hold_duration': datetime.now() - self.entry_time,
                    'position': 0
                }
                self._log_trade(trade_record)
                
                logger.info(f"Sell order executed: {self.position_size:.6f} {BASE_CURRENCY} at ${price:.2f}")
                logger.info(f"Realized P&L: ${realized_pnl:.2f} ({realized_pnl_pct:.2f}%)")
                
                # Reset position
                self.current_position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.entry_time = None
                self.total_trades += 1
                self.daily_trades += 1
                self.last_trade_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
    
    def _get_unrealized_pnl(self):
        """Calculate current unrealized P&L."""
        if self.current_position != 1 or self.position_size <= 0:
            return 0.0
        
        if self.last_kline_data:
            current_price = self.last_kline_data['close']
            cost_basis = self.position_size * self.entry_price
            current_value = self.position_size * current_price
            unrealized_pnl = (current_value - cost_basis) / cost_basis
            return np.clip(unrealized_pnl, -0.5, 0.5)  # Clamp to reasonable range
        
        return 0.0
    
    def _safety_checks(self):
        """Perform safety checks before trading."""
        try:
            # Check daily trade limits
            if self.daily_trades >= MAX_DAILY_TRADES:
                logger.warning(f"Daily trade limit reached: {self.daily_trades}")
                return False
            
            # Check daily loss limits
            if self.daily_pnl <= -INITIAL_BALANCE * MAX_DAILY_LOSS_PERCENT:
                logger.warning(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
                return False
            
            # Check market conditions
            if not self.data_manager.is_market_open():
                return False
            
            # Check WebSocket connection
            if not self.data_manager.is_websocket_running:
                logger.warning("WebSocket connection lost")
                return False
            
            # Check if we have recent data
            if self.last_kline_data:
                last_update = self.last_kline_data['timestamp']
                if isinstance(last_update, str):
                    last_update = pd.to_datetime(last_update)
                
                time_since_update = datetime.now() - last_update.to_pydatetime()
                if time_since_update.total_seconds() > 300:  # 5 minutes
                    logger.warning("Market data is stale")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return False
    
    def _log_trade(self, trade_record):
        """Log trade to file and update performance metrics."""
        try:
            # Save to CSV
            trade_df = pd.DataFrame([trade_record])
            trade_file = DATA_DIR / 'live_trades.csv'
            
            if trade_file.exists():
                trade_df.to_csv(trade_file, mode='a', header=False, index=False)
            else:
                trade_df.to_csv(trade_file, index=False)
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def _update_performance_metrics(self):
        """Update and log performance metrics."""
        try:
            current_time = datetime.now()
            session_duration = current_time - self.session_start_time
            
            # Calculate metrics
            win_rate = (self.successful_trades / max(self.total_trades, 1)) * 100
            avg_pnl = self.total_pnl / max(self.total_trades, 1)
            
            metrics = {
                'timestamp': current_time,
                'session_duration_hours': session_duration.total_seconds() / 3600,
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'win_rate': win_rate,
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'avg_pnl_per_trade': avg_pnl,
                'current_position': self.current_position,
                'unrealized_pnl': self._get_unrealized_pnl()
            }
            
            self.performance_metrics.append(metrics)
            
            # Log performance periodically
            if len(self.performance_metrics) % 10 == 0:
                logger.info(f"Performance Update - "
                          f"Trades: {self.total_trades}, "
                          f"Win Rate: {win_rate:.1f}%, "
                          f"Total P&L: ${self.total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring and reporting loop."""
        while self.is_trading and not self.shutdown_event.is_set():
            try:
                # Log status every 5 minutes
                self._log_status()
                
                # Save performance metrics
                if self.performance_metrics:
                    metrics_df = pd.DataFrame(self.performance_metrics)
                    metrics_df.to_csv(PERFORMANCE_LOG_FILE, index=False)
                
                # Sleep for 5 minutes
                for _ in range(300):  # 5 minutes = 300 seconds
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _log_status(self):
        """Log current trading status."""
        try:
            if self.last_kline_data:
                current_price = self.last_kline_data['close']
                
                status_msg = (f"Status: Price=${current_price:.2f}, "
                            f"Position={ACTIONS[self.current_position]}, "
                            f"Daily P&L=${self.daily_pnl:.2f}, "
                            f"Total Trades={self.total_trades}")
                
                if self.current_position == 1:
                    unrealized_pnl = self._get_unrealized_pnl() * 100
                    status_msg += f", Unrealized P&L={unrealized_pnl:.2f}%"
                
                logger.info(status_msg)
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_trading = False
        self.shutdown_event.set()
    
    def _cleanup(self):
        """Cleanup resources and save final state."""
        try:
            logger.info("Cleaning up...")
            
            # Stop WebSocket feed
            if self.data_manager:
                self.data_manager.stop_websocket_feed()
            
            # Save final performance metrics
            if self.performance_metrics:
                final_metrics_file = LOG_DIR / f"final_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                metrics_df = pd.DataFrame(self.performance_metrics)
                metrics_df.to_csv(final_metrics_file, index=False)
                logger.info(f"Final performance metrics saved to {final_metrics_file}")
            
            # Log final statistics
            self._log_final_statistics()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _log_final_statistics(self):
        """Log final trading session statistics."""
        try:
            session_duration = datetime.now() - self.session_start_time
            win_rate = (self.successful_trades / max(self.total_trades, 1)) * 100
            
            logger.info("=" * 60)
            logger.info("TRADING SESSION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Session Duration: {session_duration}")
            logger.info(f"Total Trades: {self.total_trades}")
            logger.info(f"Successful Trades: {self.successful_trades}")
            logger.info(f"Win Rate: {win_rate:.2f}%")
            logger.info(f"Total P&L: ${self.total_pnl:.2f}")
            logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
            logger.info(f"Final Position: {ACTIONS[self.current_position]}")
            
            if self.current_position == 1:
                unrealized_pnl = self._get_unrealized_pnl() * 100
                logger.info(f"Unrealized P&L: {unrealized_pnl:.2f}%")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error logging final statistics: {e}")


def main():
    """Main trading function."""
    parser = argparse.ArgumentParser(description="Live Cryptocurrency Trading Bot")
    parser.add_argument('--model', type=str, default=str(FINAL_MODEL_PATH),
                       help='Path to trained model file')
    parser.add_argument('--testnet', action='store_true', default=True,
                       help='Use testnet/paper trading (default: True)')
    parser.add_argument('--live', action='store_true',
                       help='Use live trading (overrides testnet)')
    
    args = parser.parse_args()
    
    # Determine trading mode
    use_testnet = not args.live  # Live mode overrides testnet
    
    if not use_testnet:
        confirmation = input("WARNING: Live trading mode selected. "
                           "This will use real money. Continue? (yes/no): ")
        if confirmation.lower() != 'yes':
            logger.info("Live trading cancelled by user")
            return
    
    try:
        # Initialize live trader
        trader = LiveTrader(
            model_path=args.model,
            use_testnet=use_testnet
        )
        
        # Setup components
        trader.setup_components()
        
        # Start live trading
        trader.start_live_trading()
        
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    except Exception as e:
        logger.error(f"Trading failed: {e}")
        raise


if __name__ == "__main__":
    main()