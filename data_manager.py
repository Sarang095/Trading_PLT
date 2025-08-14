"""
Data Manager Module for Cryptocurrency Trading Bot
Handles historical data fetching and live WebSocket data streaming.
"""

import ccxt
import pandas as pd
import numpy as np
import json
import logging
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any, List
import websocket
from websocket import WebSocketApp
import requests
from pathlib import Path

from config import *

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages cryptocurrency data fetching and streaming.
    Supports both historical data retrieval and live WebSocket feeds.
    """
    
    def __init__(self, exchange_name: str = 'binance', use_testnet: bool = True):
        """
        Initialize the DataManager.
        
        Args:
            exchange_name: Name of the exchange ('binance', 'coinbase', etc.)
            use_testnet: Whether to use testnet/sandbox environment
        """
        self.exchange_name = exchange_name
        self.use_testnet = use_testnet
        self.exchange = None
        self.websocket_app = None
        self.is_websocket_running = False
        self.websocket_thread = None
        self.reconnect_count = 0
        self.last_heartbeat = None
        self.data_callback = None
        
        self._setup_exchange()
        
    def _setup_exchange(self):
        """Setup the exchange connection using ccxt."""
        try:
            if self.exchange_name.lower() == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': EXCHANGE_API_KEYS['binance']['api_key'],
                    'secret': EXCHANGE_API_KEYS['binance']['secret'],
                    'sandbox': self.use_testnet,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
            elif self.exchange_name.lower() == 'coinbase':
                self.exchange = ccxt.coinbasepro({
                    'apiKey': EXCHANGE_API_KEYS['coinbase']['api_key'],
                    'secret': EXCHANGE_API_KEYS['coinbase']['secret'],
                    'password': EXCHANGE_API_KEYS['coinbase']['passphrase'],
                    'sandbox': self.use_testnet,
                    'enableRateLimit': True
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange_name}")
                
            # Test connection
            self.exchange.load_markets()
            logger.info(f"Successfully connected to {self.exchange_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup exchange {self.exchange_name}: {e}")
            raise
    
    def fetch_historical_data(self, 
                            symbol: str, 
                            timeframe: str, 
                            start_date: Optional[str] = None, 
                            end_date: Optional[str] = None,
                            limit: Optional[int] = None,
                            save_to_file: bool = True) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from the exchange.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for the data (e.g., '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of candles to fetch
            save_to_file: Whether to save data to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching historical data for {symbol} ({timeframe})")
            
            # Convert dates to timestamps if provided
            since = None
            if start_date:
                since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
            
            until = None
            if end_date:
                until = self.exchange.parse8601(f"{end_date}T23:59:59Z")
            
            # Fetch data in chunks if needed
            all_data = []
            current_since = since
            
            while True:
                try:
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=min(1000, limit) if limit else 1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    
                    # Check if we've reached the end date or limit
                    if until and ohlcv[-1][0] >= until:
                        break
                    if limit and len(all_data) >= limit:
                        all_data = all_data[:limit]
                        break
                    
                    # Update since for next iteration
                    current_since = ohlcv[-1][0] + 1
                    
                    # Rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                except ccxt.BaseError as e:
                    logger.warning(f"Error fetching batch: {e}")
                    time.sleep(5)
                    continue
            
            if not all_data:
                raise ValueError("No data retrieved")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any duplicate timestamps
            df = df[~df.index.duplicated(keep='last')]
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            logger.info(f"Retrieved {len(df)} data points from {df.index[0]} to {df.index[-1]}")
            
            # Save to file if requested
            if save_to_file:
                filename = DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}_historical.csv"
                df.to_csv(filename)
                logger.info(f"Historical data saved to {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise
    
    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load historical data from saved CSV file.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for the data
            
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            filename = DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}_historical.csv"
            
            if not filename.exists():
                logger.warning(f"Historical data file not found: {filename}")
                return self.fetch_historical_data(symbol, timeframe)
            
            df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
            logger.info(f"Loaded {len(df)} historical data points from {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise
    
    def start_websocket_feed(self, 
                           symbol: str, 
                           timeframe: str, 
                           callback: Callable[[Dict[str, Any]], None]):
        """
        Start WebSocket feed for live market data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for the data (e.g., '1h')
            callback: Function to call with new data
        """
        try:
            self.data_callback = callback
            
            # Construct WebSocket URL based on exchange
            if self.exchange_name.lower() == 'binance':
                # Convert symbol format for Binance WebSocket
                ws_symbol = symbol.replace('/', '').lower()
                
                # Convert timeframe to Binance format
                timeframe_map = {
                    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                    '1h': '1h', '4h': '4h', '1d': '1d'
                }
                ws_timeframe = timeframe_map.get(timeframe, '1h')
                
                if self.use_testnet:
                    ws_url = f"wss://testnet.binance.vision/ws/{ws_symbol}@kline_{ws_timeframe}"
                else:
                    ws_url = f"wss://stream.binance.com:9443/ws/{ws_symbol}@kline_{ws_timeframe}"
            else:
                raise ValueError(f"WebSocket not implemented for {self.exchange_name}")
            
            logger.info(f"Starting WebSocket connection to {ws_url}")
            
            # Create WebSocket app
            self.websocket_app = WebSocketApp(
                ws_url,
                on_open=self._on_websocket_open,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close
            )
            
            # Start WebSocket in separate thread
            self.is_websocket_running = True
            self.websocket_thread = threading.Thread(
                target=self._run_websocket,
                daemon=True
            )
            self.websocket_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket feed: {e}")
            raise
    
    def _run_websocket(self):
        """Run the WebSocket connection with automatic reconnection."""
        while self.is_websocket_running:
            try:
                self.websocket_app.run_forever(
                    ping_interval=30,
                    ping_timeout=10
                )
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
            
            if self.is_websocket_running:
                self.reconnect_count += 1
                
                if self.reconnect_count > WEBSOCKET_MAX_RETRIES:
                    logger.error("Maximum WebSocket reconnection attempts exceeded")
                    break
                
                logger.info(f"Attempting to reconnect WebSocket (attempt {self.reconnect_count})")
                time.sleep(WEBSOCKET_RECONNECT_DELAY)
    
    def _on_websocket_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info("WebSocket connection opened")
        self.reconnect_count = 0
        self.last_heartbeat = time.time()
    
    def _on_websocket_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            self.last_heartbeat = time.time()
            
            # Parse Binance kline data
            if 'k' in data:
                kline = data['k']
                
                # Extract OHLCV data
                parsed_data = {
                    'symbol': kline['s'],
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_closed': kline['x']  # Whether this kline is closed
                }
                
                # Call the callback function
                if self.data_callback:
                    self.data_callback(parsed_data)
            
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_websocket_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.warning(f"WebSocket connection closed: {close_status_code} - {close_msg}")
    
    def stop_websocket_feed(self):
        """Stop the WebSocket feed."""
        logger.info("Stopping WebSocket feed")
        self.is_websocket_running = False
        
        if self.websocket_app:
            self.websocket_app.close()
        
        if self.websocket_thread and self.websocket_thread.is_alive():
            self.websocket_thread.join(timeout=5)
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current price
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get the current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of orders to retrieve
            
        Returns:
            Order book data
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            raise
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance for all currencies.
        
        Returns:
            Dictionary of currency balances
        """
        try:
            balance = self.exchange.fetch_balance()
            return {currency: info['free'] for currency, info in balance.items() 
                   if info['free'] > 0}
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            df = df.copy()
            
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
            
            # Bollinger Bands
            bb_middle = df['close'].rolling(window=BB_PERIOD).mean()
            bb_std = df['close'].rolling(window=BB_PERIOD).std()
            bb_upper = bb_middle + (bb_std * BB_STD)
            bb_lower = bb_middle - (bb_std * BB_STD)
            df['bb_percent_b_20_2'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # MACD
            ema_fast = df['close'].ewm(span=MACD_FAST).mean()
            ema_slow = df['close'].ewm(span=MACD_SLOW).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=MACD_SIGNAL).mean()
            df['macd_hist'] = macd_line - macd_signal
            
            # Volume SMA
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            
            # Normalize volume
            df['volume_sma_20'] = df['volume_sma_20'] / df['volume_sma_20'].rolling(window=50).mean()
            
            logger.debug("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            raise
    
    def is_market_open(self) -> bool:
        """
        Check if the market is open (cryptocurrency markets are always open).
        
        Returns:
            True (crypto markets are 24/7)
        """
        return True
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status and trading statistics.
        
        Returns:
            Market status information
        """
        try:
            status = {
                'timestamp': datetime.now(),
                'is_open': self.is_market_open(),
                'websocket_connected': self.is_websocket_running,
                'last_heartbeat': self.last_heartbeat,
                'reconnect_count': self.reconnect_count
            }
            
            # Add current prices for major pairs
            major_pairs = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
            for pair in major_pairs:
                try:
                    ticker = self.exchange.fetch_ticker(pair)
                    status[f"{pair}_price"] = ticker['last']
                    status[f"{pair}_change_24h"] = ticker['percentage']
                except:
                    continue
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {'error': str(e)}
    
    def __del__(self):
        """Cleanup when DataManager is destroyed."""
        self.stop_websocket_feed()


# Utility functions for data preprocessing
def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax', 'zscore')
        
    Returns:
        Normalized data array
    """
    if method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sequences for time series data.
    
    Args:
        data: Input data array
        sequence_length: Length of each sequence
        
    Returns:
        Array of sequences
    """
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize DataManager
        dm = DataManager(exchange_name='binance', use_testnet=True)
        
        # Fetch historical data
        df = dm.fetch_historical_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date='2024-01-01',
            limit=1000
        )
        
        # Calculate technical indicators
        df_with_indicators = dm.calculate_technical_indicators(df)
        print(f"Historical data shape: {df_with_indicators.shape}")
        print(f"Columns: {df_with_indicators.columns.tolist()}")
        
        # Test WebSocket (comment out for automated testing)
        # def on_data(data):
        #     print(f"Received: {data}")
        # 
        # dm.start_websocket_feed('BTC/USDT', '1h', on_data)
        # time.sleep(10)
        # dm.stop_websocket_feed()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")