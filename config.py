"""
Configuration settings for the Cryptocurrency Trading Bot
All hyperparameters and system settings are defined here.
"""

import os
from pathlib import Path

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Exchange API Keys (Replace with your actual keys)
EXCHANGE_API_KEYS = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', 'your_binance_api_key_here'),
        'secret': os.getenv('BINANCE_SECRET_KEY', 'your_binance_secret_key_here'),
        'sandbox': True,  # Set to False for live trading
    },
    'coinbase': {
        'api_key': os.getenv('COINBASE_API_KEY', 'your_coinbase_api_key_here'),
        'secret': os.getenv('COINBASE_SECRET_KEY', 'your_coinbase_secret_key_here'),
        'passphrase': os.getenv('COINBASE_PASSPHRASE', 'your_coinbase_passphrase_here'),
        'sandbox': True,
    }
}

# Data Provider API Keys
DATA_API_KEYS = {
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', 'your_alpha_vantage_key_here'),
    'news_api': os.getenv('NEWS_API_KEY', 'your_news_api_key_here'),
}

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Asset Configuration
CRYPTO_PAIR = 'BTC/USDT'
BASE_CURRENCY = 'BTC'
QUOTE_CURRENCY = 'USDT'
TIMEFRAME = '1h'

# Trading Parameters
INITIAL_BALANCE = 10000  # Starting balance in USDT
TRANSACTION_FEE_PERCENT = 0.001  # 0.1% transaction fee
MAX_POSITION_SIZE = 0.95  # Maximum 95% of portfolio in one position
MIN_TRADE_AMOUNT = 10  # Minimum trade amount in USDT

# Risk Management
STOP_LOSS_PERCENT = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENT = 0.15  # 15% take profit
MAX_DRAWDOWN_PERCENT = 0.20  # 20% maximum drawdown before stopping

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# State Space Configuration
STATE_WINDOW_SIZE = 60  # Number of historical time steps to include in state
TECHNICAL_INDICATORS = [
    'ema_9', 'ema_21', 'ema_200',
    'rsi_14', 'bb_percent_b_20_2',
    'macd_hist', 'volume_sma_20'
]

# Action Space
ACTIONS = {
    0: 'HOLD',
    1: 'BUY', 
    2: 'SELL'
}
N_ACTIONS = len(ACTIONS)

# =============================================================================
# REINFORCEMENT LEARNING HYPERPARAMETERS
# =============================================================================

# Q-Learning Parameters
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.0001
TAU = 0.005  # Soft update parameter for target network

# Experience Replay
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
PRIORITY_ALPHA = 0.6  # Prioritized experience replay alpha
PRIORITY_BETA_START = 0.4  # Prioritized experience replay beta start
PRIORITY_BETA_FRAMES = 100000  # Frames to anneal beta to 1.0

# Training Schedule
TARGET_UPDATE_FREQUENCY = 1000  # Update target network every N steps
SAVE_MODEL_FREQUENCY = 10000  # Save model every N steps

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================

# Network Architecture
CONV_LAYERS = [
    {'out_channels': 32, 'kernel_size': 8, 'stride': 4},
    {'out_channels': 64, 'kernel_size': 4, 'stride': 2},
    {'out_channels': 64, 'kernel_size': 3, 'stride': 1}
]

DENSE_LAYERS = [512, 256]
DUELING_LAYERS = [128]  # Layers for both value and advantage streams

DROPOUT_RATE = 0.3
ACTIVATION_FUNCTION = 'relu'

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Imitation Learning Phase
IL_EPOCHS = 10
IL_LEARNING_RATE = 0.001
IL_BATCH_SIZE = 128

# Reinforcement Learning Phase
RL_TOTAL_TIMESTEPS = 1000000
RL_WARMUP_STEPS = 10000  # Steps before training starts
RL_TRAIN_FREQUENCY = 4  # Train every N steps
RL_EVALUATION_FREQUENCY = 50000  # Evaluate every N steps

# Expert Trader Strategy Parameters
EMA_SHORT_PERIOD = 9
EMA_LONG_PERIOD = 21
EMA_TREND_PERIOD = 200
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
PARABOLIC_SAR_AF = 0.02
PARABOLIC_SAR_MAX_AF = 0.2

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Historical Data
HISTORICAL_DATA_START = '2020-01-01'  # Start date for historical data
HISTORICAL_DATA_LOOKBACK_DAYS = 1095  # 3 years of data
MIN_DATA_POINTS = 1000  # Minimum data points required for training

# WebSocket Configuration
WEBSOCKET_ENDPOINTS = {
    'binance': 'wss://stream.binance.com:9443/ws/btcusdt@kline_1h',
    'binance_testnet': 'wss://testnet.binance.vision/ws/btcusdt@kline_1h'
}

WEBSOCKET_RECONNECT_DELAY = 5  # Seconds to wait before reconnecting
WEBSOCKET_MAX_RETRIES = 10

# =============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# =============================================================================

# Sentiment Model
SENTIMENT_MODEL_NAME = 'ProsusAI/finbert'
SENTIMENT_MAX_LENGTH = 512
SENTIMENT_BATCH_SIZE = 16

# News Sources Configuration
NEWS_SOURCES = [
    'coindesk', 'cointelegraph', 'cryptonews',
    'reuters', 'bloomberg', 'marketwatch'
]

SENTIMENT_UPDATE_FREQUENCY = 3600  # Update sentiment every hour (seconds)
SENTIMENT_WEIGHT = 0.1  # Weight of sentiment in decision making

# =============================================================================
# REWARD FUNCTION CONFIGURATION
# =============================================================================

# Multi-Objective Reward Components
REWARD_WEIGHTS = {
    'profit': 1.0,           # Weight for profit/loss
    'sharpe': 0.3,           # Weight for Sharpe ratio improvement
    'drawdown': -0.5,        # Weight for drawdown penalty
    'transaction_cost': -1.0, # Weight for transaction cost penalty
    'hold_time': 0.1         # Weight for position holding time bonus
}

# Reward Scaling
REWARD_SCALING_FACTOR = 100
SHARPE_LOOKBACK_WINDOW = 252  # Trading days for Sharpe calculation

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'logs/trading_bot.log'

# Performance Metrics
METRICS_UPDATE_FREQUENCY = 100  # Update metrics every N steps
PERFORMANCE_METRICS = [
    'total_return', 'sharpe_ratio', 'max_drawdown',
    'win_rate', 'profit_factor', 'avg_trade_duration'
]

# =============================================================================
# FILE PATHS
# =============================================================================

# Directory Structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
LOG_DIR = PROJECT_ROOT / 'logs'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# File Paths
HISTORICAL_DATA_FILE = DATA_DIR / f'{BASE_CURRENCY}_{QUOTE_CURRENCY}_historical.csv'
MODEL_CHECKPOINT_PATH = MODEL_DIR / 'ddqn_checkpoint.pth'
EXPERT_MODEL_PATH = MODEL_DIR / 'expert_pretrained.pth'
FINAL_MODEL_PATH = MODEL_DIR / 'final_model.pth'
TRADING_LOG_FILE = LOG_DIR / 'trading_execution.log'
PERFORMANCE_LOG_FILE = LOG_DIR / 'performance_metrics.csv'

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Hardware Configuration
DEVICE = 'cuda'  # 'cuda' if GPU available, 'cpu' otherwise
NUM_WORKERS = 4  # Number of workers for data loading
SEED = 42  # Random seed for reproducibility

# Memory Management
MAX_MEMORY_USAGE = 0.8  # Maximum memory usage (80%)
GARBAGE_COLLECTION_FREQUENCY = 1000  # Run GC every N steps

# Safety Checks
ENABLE_SAFETY_CHECKS = True
MAX_DAILY_TRADES = 50
MAX_DAILY_LOSS_PERCENT = 0.05  # Stop trading if daily loss exceeds 5%

# =============================================================================
# ENVIRONMENT VARIABLES VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration settings and environment variables."""
    required_vars = []
    missing_vars = []
    
    # Check for required environment variables in production
    if not EXCHANGE_API_KEYS['binance']['sandbox']:
        required_vars.extend(['BINANCE_API_KEY', 'BINANCE_SECRET_KEY'])
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Validate numeric parameters
    assert 0 < GAMMA <= 1, "GAMMA must be between 0 and 1"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
    assert REPLAY_BUFFER_SIZE > BATCH_SIZE, "REPLAY_BUFFER_SIZE must be larger than BATCH_SIZE"
    assert 0 < TRANSACTION_FEE_PERCENT < 1, "TRANSACTION_FEE_PERCENT must be between 0 and 1"
    assert STATE_WINDOW_SIZE > 0, "STATE_WINDOW_SIZE must be positive"
    
    print("Configuration validation passed!")

if __name__ == "__main__":
    validate_config()