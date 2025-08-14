"""
Expert Trader Module for Cryptocurrency Trading Bot
Implements the EMA + Heikin Ashi + Parabolic SAR strategy for generating expert actions.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import *

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ExpertTrader:
    """
    Expert trading strategy implementation using EMA + Heikin Ashi + Parabolic SAR.
    
    Trading Rules:
    - Buy Signal: Price above EMA(200), Heikin Ashi candle is green, and Parabolic SAR dot below candles
    - Sell Signal: Price below EMA(200), Heikin Ashi candle is red, and Parabolic SAR dot above candles
                  OR long position open and SAR dot flips above red Heikin Ashi candle
    """
    
    def __init__(self):
        """Initialize the ExpertTrader with strategy parameters."""
        self.position = 0  # 0: No position, 1: Long position
        self.entry_price = 0.0
        self.entry_timestamp = None
        self.trade_history = []
        
        # Strategy parameters from config
        self.ema_trend_period = EMA_TREND_PERIOD  # 200
        self.sar_af = PARABOLIC_SAR_AF  # 0.02
        self.sar_max_af = PARABOLIC_SAR_MAX_AF  # 0.2
        
        logger.info("ExpertTrader initialized with EMA + Heikin Ashi + Parabolic SAR strategy")
    
    def generate_expert_actions(self, df: pd.DataFrame) -> List[int]:
        """
        Generate expert trading actions based on the strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of actions (0: Hold, 1: Buy, 2: Sell)
        """
        try:
            logger.info(f"Generating expert actions for {len(df)} data points")
            
            # Make a copy to avoid modifying original data
            data = df.copy()
            
            # Calculate all required indicators
            data = self._calculate_indicators(data)
            
            # Generate trading signals
            actions = self._generate_signals(data)
            
            logger.info(f"Generated {len(actions)} expert actions")
            return actions
            
        except Exception as e:
            logger.error(f"Error generating expert actions: {e}")
            raise
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for the strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        try:
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in DataFrame")
            
            # Calculate EMA(200) for trend determination
            df['ema_200'] = df['close'].ewm(span=self.ema_trend_period).mean()
            
            # Calculate Heikin Ashi candles
            df = self._calculate_heikin_ashi(df)
            
            # Calculate Parabolic SAR
            df = self._calculate_parabolic_sar(df)
            
            # Generate trading signals
            df = self._generate_signal_flags(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    def _calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin Ashi candles.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Heikin Ashi values added
        """
        # Initialize Heikin Ashi columns
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['ha_open'] = np.nan
        df['ha_high'] = np.nan
        df['ha_low'] = np.nan
        
        # Calculate first Heikin Ashi open
        df.iloc[0, df.columns.get_loc('ha_open')] = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2
        
        # Calculate Heikin Ashi values
        for i in range(1, len(df)):
            # HA Open = (previous HA Open + previous HA Close) / 2
            df.iloc[i, df.columns.get_loc('ha_open')] = (
                df.iloc[i-1]['ha_open'] + df.iloc[i-1]['ha_close']
            ) / 2
        
        # Calculate HA High and Low
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        # Determine Heikin Ashi candle color
        df['ha_color'] = np.where(df['ha_close'] > df['ha_open'], 'green', 'red')
        
        return df
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Parabolic SAR indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Parabolic SAR values added
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Initialize arrays
        psar = np.zeros(len(df))
        psaraf = np.zeros(len(df))
        psardir = np.zeros(len(df))
        
        # Initialize first values
        psar[0] = low[0]
        psaraf[0] = self.sar_af
        psardir[0] = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(df)):
            if psardir[i-1] == 1:  # Uptrend
                psar[i] = psar[i-1] + psaraf[i-1] * (high[i-1] - psar[i-1])
                
                if low[i] <= psar[i]:  # Trend reversal
                    psardir[i] = -1
                    psar[i] = max(high[i-1], high[i-2] if i > 1 else high[i-1])
                    psaraf[i] = self.sar_af
                else:
                    psardir[i] = 1
                    if high[i] > high[i-1]:
                        psaraf[i] = min(psaraf[i-1] + self.sar_af, self.sar_max_af)
                    else:
                        psaraf[i] = psaraf[i-1]
                    
                    # Ensure SAR doesn't go above low of previous two periods
                    psar[i] = min(psar[i], min(low[i-1], low[i-2] if i > 1 else low[i-1]))
            
            else:  # Downtrend
                psar[i] = psar[i-1] + psaraf[i-1] * (low[i-1] - psar[i-1])
                
                if high[i] >= psar[i]:  # Trend reversal
                    psardir[i] = 1
                    psar[i] = min(low[i-1], low[i-2] if i > 1 else low[i-1])
                    psaraf[i] = self.sar_af
                else:
                    psardir[i] = -1
                    if low[i] < low[i-1]:
                        psaraf[i] = min(psaraf[i-1] + self.sar_af, self.sar_max_af)
                    else:
                        psaraf[i] = psaraf[i-1]
                    
                    # Ensure SAR doesn't go below high of previous two periods
                    psar[i] = max(psar[i], max(high[i-1], high[i-2] if i > 1 else high[i-1]))
        
        df['psar'] = psar
        df['psar_direction'] = psardir
        
        return df
    
    def _generate_signal_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signal flags based on the strategy rules.
        
        Args:
            df: DataFrame with all indicators calculated
            
        Returns:
            DataFrame with signal flags added
        """
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['trend_bullish'] = df['close'] > df['ema_200']
        df['trend_bearish'] = df['close'] < df['ema_200']
        df['sar_bullish'] = df['psar'] < df['close']
        df['sar_bearish'] = df['psar'] > df['close']
        
        # Buy signals: Price above EMA(200), green Heikin Ashi, SAR below price
        df['buy_signal'] = (
            df['trend_bullish'] & 
            (df['ha_color'] == 'green') & 
            df['sar_bullish']
        )
        
        # Sell signals: Price below EMA(200), red Heikin Ashi, SAR above price
        # OR SAR flips above during red candle (stop loss)
        df['sell_signal'] = (
            (df['trend_bearish'] & (df['ha_color'] == 'red') & df['sar_bearish']) |
            ((df['ha_color'] == 'red') & df['sar_bearish'] & (df['psar_direction'].shift(1) != df['psar_direction']))
        )
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> List[int]:
        """
        Generate actual trading actions based on signals and position management.
        
        Args:
            df: DataFrame with signal flags
            
        Returns:
            List of actions (0: Hold, 1: Buy, 2: Sell)
        """
        actions = []
        self.position = 0  # Reset position for strategy evaluation
        self.trade_history = []
        
        for i, row in df.iterrows():
            action = 0  # Default to HOLD
            
            # Skip if not enough data for indicators
            if pd.isna(row['ema_200']) or pd.isna(row['psar']) or pd.isna(row['ha_close']):
                actions.append(action)
                continue
            
            if self.position == 0:  # No position
                if row['buy_signal']:
                    action = 1  # BUY
                    self.position = 1
                    self.entry_price = row['close']
                    self.entry_timestamp = i
                    
                    # Record trade
                    trade_record = {
                        'type': 'buy',
                        'timestamp': i,
                        'price': row['close'],
                        'ema_200': row['ema_200'],
                        'ha_color': row['ha_color'],
                        'psar': row['psar'],
                        'psar_direction': row['psar_direction']
                    }
                    self.trade_history.append(trade_record)
                    
            elif self.position == 1:  # Long position
                if row['sell_signal']:
                    action = 2  # SELL
                    self.position = 0
                    
                    # Calculate trade performance
                    pnl = (row['close'] - self.entry_price) / self.entry_price
                    
                    # Record trade
                    trade_record = {
                        'type': 'sell',
                        'timestamp': i,
                        'price': row['close'],
                        'entry_price': self.entry_price,
                        'pnl': pnl,
                        'hold_duration': i - self.entry_timestamp if self.entry_timestamp else 0,
                        'ema_200': row['ema_200'],
                        'ha_color': row['ha_color'],
                        'psar': row['psar'],
                        'psar_direction': row['psar_direction']
                    }
                    self.trade_history.append(trade_record)
                    
                    # Reset position variables
                    self.entry_price = 0.0
                    self.entry_timestamp = None
            
            actions.append(action)
        
        return actions
    
    def get_strategy_performance(self, df: pd.DataFrame, actions: List[int]) -> Dict[str, Any]:
        """
        Calculate strategy performance metrics.
        
        Args:
            df: DataFrame with price data
            actions: List of trading actions
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if len(actions) != len(df):
                raise ValueError("Actions list length must match DataFrame length")
            
            # Calculate returns
            returns = []
            position = 0
            entry_price = 0
            
            for i, action in enumerate(actions):
                if action == 1 and position == 0:  # Buy
                    position = 1
                    entry_price = df.iloc[i]['close']
                elif action == 2 and position == 1:  # Sell
                    exit_price = df.iloc[i]['close']
                    trade_return = (exit_price - entry_price) / entry_price
                    returns.append(trade_return)
                    position = 0
            
            if not returns:
                return {
                    'total_trades': 0,
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Calculate metrics
            total_trades = len(returns)
            total_return = np.prod([1 + r for r in returns]) - 1
            win_rate = len([r for r in returns if r > 0]) / total_trades
            avg_return = np.mean(returns)
            
            # Sharpe ratio (assuming daily returns and risk-free rate of 0)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod([1 + r for r in returns])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            return {
                'total_trades': total_trades,
                'total_return': total_return,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'individual_returns': returns
            }
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {e}")
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
    
    def analyze_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the quality and distribution of trading signals.
        
        Args:
            df: DataFrame with calculated indicators and signals
            
        Returns:
            Dictionary with signal analysis
        """
        try:
            # Calculate signals if not already done
            df_analyzed = self._calculate_indicators(df)
            
            total_periods = len(df_analyzed)
            buy_signals = df_analyzed['buy_signal'].sum()
            sell_signals = df_analyzed['sell_signal'].sum()
            
            # Signal frequency analysis
            signal_frequency = (buy_signals + sell_signals) / total_periods if total_periods > 0 else 0
            
            # Trend analysis
            bullish_periods = df_analyzed['trend_bullish'].sum()
            bearish_periods = df_analyzed['trend_bearish'].sum()
            
            # Heikin Ashi analysis
            green_candles = (df_analyzed['ha_color'] == 'green').sum()
            red_candles = (df_analyzed['ha_color'] == 'red').sum()
            
            # SAR analysis
            sar_bullish_periods = df_analyzed['sar_bullish'].sum()
            sar_bearish_periods = df_analyzed['sar_bearish'].sum()
            
            return {
                'total_periods': total_periods,
                'buy_signals': int(buy_signals),
                'sell_signals': int(sell_signals),
                'signal_frequency': signal_frequency,
                'bullish_trend_ratio': bullish_periods / total_periods if total_periods > 0 else 0,
                'bearish_trend_ratio': bearish_periods / total_periods if total_periods > 0 else 0,
                'green_candle_ratio': green_candles / total_periods if total_periods > 0 else 0,
                'red_candle_ratio': red_candles / total_periods if total_periods > 0 else 0,
                'sar_bullish_ratio': sar_bullish_periods / total_periods if total_periods > 0 else 0,
                'sar_bearish_ratio': sar_bearish_periods / total_periods if total_periods > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing signals: {e}")
            return {}
    
    def save_trade_history(self, filename: str):
        """
        Save trade history to CSV file.
        
        Args:
            filename: Output filename
        """
        try:
            if self.trade_history:
                df = pd.DataFrame(self.trade_history)
                filepath = DATA_DIR / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Trade history saved to {filepath}")
            else:
                logger.warning("No trade history to save")
                
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def load_trade_history(self, filename: str) -> pd.DataFrame:
        """
        Load trade history from CSV file.
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame with trade history
        """
        try:
            filepath = DATA_DIR / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                logger.info(f"Trade history loaded from {filepath}")
                return df
            else:
                logger.warning(f"Trade history file not found: {filepath}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            return pd.DataFrame()
    
    def get_current_position(self) -> Dict[str, Any]:
        """
        Get current position information.
        
        Returns:
            Dictionary with current position data
        """
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_timestamp': self.entry_timestamp,
            'trade_count': len(self.trade_history)
        }
    
    def reset_position(self):
        """Reset position and trade history."""
        self.position = 0
        self.entry_price = 0.0
        self.entry_timestamp = None
        self.trade_history = []
        logger.info("ExpertTrader position reset")


def validate_expert_strategy(df: pd.DataFrame, expert_trader: ExpertTrader) -> Dict[str, Any]:
    """
    Validate the expert trading strategy on historical data.
    
    Args:
        df: DataFrame with historical OHLCV data
        expert_trader: ExpertTrader instance
        
    Returns:
        Dictionary with validation results
    """
    try:
        logger.info("Validating expert trading strategy")
        
        # Generate expert actions
        actions = expert_trader.generate_expert_actions(df)
        
        # Calculate performance
        performance = expert_trader.get_strategy_performance(df, actions)
        
        # Analyze signals
        df_with_indicators = expert_trader._calculate_indicators(df.copy())
        signal_analysis = expert_trader.analyze_signals(df)
        
        # Buy and hold benchmark
        buy_hold_return = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']
        
        validation_results = {
            'strategy_performance': performance,
            'signal_analysis': signal_analysis,
            'buy_hold_return': buy_hold_return,
            'outperformance': performance.get('total_return', 0) - buy_hold_return,
            'actions_distribution': {
                'hold': actions.count(0),
                'buy': actions.count(1),
                'sell': actions.count(2)
            },
            'data_period': {
                'start': df.index[0] if hasattr(df.index, '__getitem__') else 'N/A',
                'end': df.index[-1] if hasattr(df.index, '__getitem__') else 'N/A',
                'total_periods': len(df)
            }
        }
        
        logger.info(f"Strategy validation completed. Total return: {performance.get('total_return', 0):.2%}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating expert strategy: {e}")
        return {}


if __name__ == "__main__":
    # Example usage and testing
    try:
        from data_manager import DataManager
        
        # Initialize components
        data_manager = DataManager(exchange_name='binance', use_testnet=True)
        expert_trader = ExpertTrader()
        
        # Generate sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h')
        np.random.seed(42)
        
        # Create realistic price data with trend
        n_periods = len(dates)
        returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift with volatility
        prices = 30000 * np.exp(np.cumsum(returns))  # Starting at $30,000
        
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
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Test expert strategy
        validation_results = validate_expert_strategy(df, expert_trader)
        
        print("\nExpert Strategy Validation Results:")
        print(f"Total trades: {validation_results['strategy_performance']['total_trades']}")
        print(f"Total return: {validation_results['strategy_performance']['total_return']:.2%}")
        print(f"Win rate: {validation_results['strategy_performance']['win_rate']:.2%}")
        print(f"Sharpe ratio: {validation_results['strategy_performance']['sharpe_ratio']:.3f}")
        print(f"Max drawdown: {validation_results['strategy_performance']['max_drawdown']:.2%}")
        print(f"Buy & Hold return: {validation_results['buy_hold_return']:.2%}")
        print(f"Outperformance: {validation_results['outperformance']:.2%}")
        
        print(f"\nActions distribution:")
        print(f"Hold: {validation_results['actions_distribution']['hold']}")
        print(f"Buy: {validation_results['actions_distribution']['buy']}")
        print(f"Sell: {validation_results['actions_distribution']['sell']}")
        
        print(f"\nSignal analysis:")
        print(f"Signal frequency: {validation_results['signal_analysis']['signal_frequency']:.3f}")
        print(f"Bullish trend ratio: {validation_results['signal_analysis']['bullish_trend_ratio']:.3f}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")