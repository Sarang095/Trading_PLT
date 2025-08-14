"""
Sentiment Analyzer Module for Cryptocurrency Trading Bot
Uses Hugging Face's FinBERT model for financial sentiment analysis.
"""

import torch
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import requests
import re
import time
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from transformers.pipelines import TextClassificationPipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from config import *

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Financial sentiment analysis using pre-trained FinBERT model.
    Processes news articles, social media posts, and other text data
    to generate sentiment scores for trading decisions.
    """
    
    def __init__(self, model_name: str = SENTIMENT_MODEL_NAME):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
        
    def _load_model(self):
        """Load the pre-trained sentiment analysis model."""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1,
                return_all_scores=True
            )
            
            logger.info(f"Successfully loaded model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get sentiment score for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment score from -1 (negative) to 1 (positive)
        """
        try:
            if not text or not isinstance(text, str):
                return 0.0
            
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            if not cleaned_text:
                return 0.0
            
            # Get sentiment prediction
            results = self.pipeline(cleaned_text)
            
            # Convert to sentiment score
            sentiment_score = self._convert_to_score(results[0])
            
            return sentiment_score
            
        except Exception as e:
            logger.warning(f"Error analyzing sentiment for text: {e}")
            return 0.0
    
    def get_batch_sentiment_scores(self, texts: List[str]) -> List[float]:
        """
        Get sentiment scores for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment scores
        """
        try:
            if not texts:
                return []
            
            # Preprocess texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Filter out empty texts
            valid_texts = [(i, text) for i, text in enumerate(cleaned_texts) if text]
            
            if not valid_texts:
                return [0.0] * len(texts)
            
            # Get predictions for valid texts
            indices, valid_text_list = zip(*valid_texts)
            
            # Process in batches to avoid memory issues
            batch_size = SENTIMENT_BATCH_SIZE
            all_scores = []
            
            for i in range(0, len(valid_text_list), batch_size):
                batch = list(valid_text_list[i:i + batch_size])
                
                try:
                    batch_results = self.pipeline(batch)
                    batch_scores = [self._convert_to_score(result) for result in batch_results]
                    all_scores.extend(batch_scores)
                except Exception as e:
                    logger.warning(f"Error processing batch {i//batch_size}: {e}")
                    all_scores.extend([0.0] * len(batch))
            
            # Map scores back to original positions
            final_scores = [0.0] * len(texts)
            for idx, score in zip(indices, all_scores):
                final_scores[idx] = score
            
            return final_scores
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {e}")
            return [0.0] * len(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text:
            return ""
        
        # Convert to string if necessary
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?;:\-]', '', text)
        
        # Trim and convert to lowercase
        text = text.strip().lower()
        
        # Truncate if too long
        if len(text) > SENTIMENT_MAX_LENGTH:
            text = text[:SENTIMENT_MAX_LENGTH]
        
        return text
    
    def _convert_to_score(self, prediction_result: List[Dict[str, Any]]) -> float:
        """
        Convert model prediction to sentiment score.
        
        Args:
            prediction_result: Raw prediction from the model
            
        Returns:
            Sentiment score from -1 to 1
        """
        try:
            # FinBERT typically outputs: negative, neutral, positive
            score_map = {}
            
            for item in prediction_result:
                label = item['label'].lower()
                score = item['score']
                score_map[label] = score
            
            # Calculate weighted sentiment score
            positive_score = score_map.get('positive', 0.0)
            negative_score = score_map.get('negative', 0.0)
            neutral_score = score_map.get('neutral', 0.0)
            
            # Convert to -1 to 1 scale
            if positive_score > negative_score:
                sentiment_score = positive_score - neutral_score * 0.5
            else:
                sentiment_score = -(negative_score - neutral_score * 0.5)
            
            # Ensure score is within bounds
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            return sentiment_score
            
        except Exception as e:
            logger.warning(f"Error converting prediction to score: {e}")
            return 0.0
    
    def analyze_news_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles.
        
        Args:
            news_articles: List of news article dictionaries with 'title' and 'content'
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if not news_articles:
                return {
                    'overall_sentiment': 0.0,
                    'article_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'confidence': 0.0
                }
            
            # Extract text from articles
            texts = []
            for article in news_articles:
                # Combine title and content for analysis
                title = article.get('title', '')
                content = article.get('content', '')
                combined_text = f"{title}. {content}".strip()
                texts.append(combined_text)
            
            # Get sentiment scores
            scores = self.get_batch_sentiment_scores(texts)
            
            # Calculate statistics
            positive_count = sum(1 for score in scores if score > 0.1)
            negative_count = sum(1 for score in scores if score < -0.1)
            neutral_count = len(scores) - positive_count - negative_count
            
            overall_sentiment = np.mean(scores) if scores else 0.0
            confidence = np.std(scores) if len(scores) > 1 else 0.0
            
            return {
                'overall_sentiment': overall_sentiment,
                'article_count': len(news_articles),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'confidence': confidence,
                'individual_scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'confidence': 0.0
            }
    
    def get_crypto_news_sentiment(self, symbol: str = 'BTC') -> Dict[str, Any]:
        """
        Fetch and analyze sentiment from cryptocurrency news.
        
        Args:
            symbol: Cryptocurrency symbol to search for
            
        Returns:
            Sentiment analysis results
        """
        try:
            news_articles = self._fetch_crypto_news(symbol)
            return self.analyze_news_sentiment(news_articles)
            
        except Exception as e:
            logger.error(f"Error getting crypto news sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'confidence': 0.0
            }
    
    def _fetch_crypto_news(self, symbol: str, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Fetch recent cryptocurrency news (placeholder implementation).
        
        Args:
            symbol: Cryptocurrency symbol
            hours_back: How many hours back to fetch news
            
        Returns:
            List of news articles
        """
        try:
            # This is a placeholder implementation
            # In a real implementation, you would integrate with news APIs like:
            # - NewsAPI
            # - CoinGecko News
            # - CryptoPanic
            # - Reddit API
            # - Twitter API
            
            # Example structure for news articles
            sample_articles = [
                {
                    'title': f'{symbol} shows strong performance in recent trading',
                    'content': f'Cryptocurrency {symbol} has demonstrated resilience in the current market conditions with increased trading volume.',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'source': 'crypto_news'
                },
                {
                    'title': f'Market analysis: {symbol} technical indicators suggest bullish trend',
                    'content': f'Technical analysis of {symbol} reveals positive momentum with key resistance levels being tested.',
                    'timestamp': datetime.now() - timedelta(hours=5),
                    'source': 'analysis_site'
                }
            ]
            
            logger.info(f"Fetched {len(sample_articles)} news articles for {symbol}")
            return sample_articles
            
        except Exception as e:
            logger.error(f"Error fetching crypto news: {e}")
            return []
    
    def create_sentiment_features(self, 
                                sentiment_scores: List[float], 
                                window_size: int = 24) -> Dict[str, float]:
        """
        Create sentiment-based features for the trading model.
        
        Args:
            sentiment_scores: List of historical sentiment scores
            window_size: Window size for feature calculation
            
        Returns:
            Dictionary of sentiment features
        """
        try:
            if not sentiment_scores or len(sentiment_scores) < window_size:
                return {
                    'sentiment_mean': 0.0,
                    'sentiment_std': 0.0,
                    'sentiment_trend': 0.0,
                    'sentiment_momentum': 0.0,
                    'positive_ratio': 0.0,
                    'negative_ratio': 0.0
                }
            
            recent_scores = sentiment_scores[-window_size:]
            
            # Basic statistics
            sentiment_mean = np.mean(recent_scores)
            sentiment_std = np.std(recent_scores)
            
            # Trend calculation (linear regression slope)
            x = np.arange(len(recent_scores))
            sentiment_trend = np.polyfit(x, recent_scores, 1)[0]
            
            # Momentum (recent vs earlier period)
            mid_point = len(recent_scores) // 2
            recent_half = recent_scores[mid_point:]
            earlier_half = recent_scores[:mid_point]
            sentiment_momentum = np.mean(recent_half) - np.mean(earlier_half)
            
            # Positive/negative ratios
            positive_count = sum(1 for score in recent_scores if score > 0.1)
            negative_count = sum(1 for score in recent_scores if score < -0.1)
            total_count = len(recent_scores)
            
            positive_ratio = positive_count / total_count if total_count > 0 else 0.0
            negative_ratio = negative_count / total_count if total_count > 0 else 0.0
            
            return {
                'sentiment_mean': sentiment_mean,
                'sentiment_std': sentiment_std,
                'sentiment_trend': sentiment_trend,
                'sentiment_momentum': sentiment_momentum,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio
            }
            
        except Exception as e:
            logger.error(f"Error creating sentiment features: {e}")
            return {
                'sentiment_mean': 0.0,
                'sentiment_std': 0.0,
                'sentiment_trend': 0.0,
                'sentiment_momentum': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0
            }
    
    def get_sentiment_signal(self, current_sentiment: float) -> Dict[str, Any]:
        """
        Generate trading signal based on sentiment analysis.
        
        Args:
            current_sentiment: Current sentiment score
            
        Returns:
            Dictionary with sentiment signal information
        """
        try:
            # Define sentiment thresholds
            strong_positive_threshold = 0.6
            positive_threshold = 0.2
            negative_threshold = -0.2
            strong_negative_threshold = -0.6
            
            # Generate signal
            if current_sentiment >= strong_positive_threshold:
                signal = 'strong_bullish'
                confidence = min(1.0, abs(current_sentiment))
                weight = 1.0
            elif current_sentiment >= positive_threshold:
                signal = 'bullish'
                confidence = min(1.0, abs(current_sentiment))
                weight = 0.7
            elif current_sentiment <= strong_negative_threshold:
                signal = 'strong_bearish'
                confidence = min(1.0, abs(current_sentiment))
                weight = -1.0
            elif current_sentiment <= negative_threshold:
                signal = 'bearish'
                confidence = min(1.0, abs(current_sentiment))
                weight = -0.7
            else:
                signal = 'neutral'
                confidence = 0.5
                weight = 0.0
            
            return {
                'signal': signal,
                'sentiment_score': current_sentiment,
                'confidence': confidence,
                'weight': weight,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating sentiment signal: {e}")
            return {
                'signal': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'weight': 0.0,
                'timestamp': datetime.now()
            }
    
    def save_sentiment_data(self, sentiment_data: List[Dict[str, Any]], filename: str):
        """
        Save sentiment analysis data to file.
        
        Args:
            sentiment_data: List of sentiment analysis results
            filename: Output filename
        """
        try:
            df = pd.DataFrame(sentiment_data)
            filepath = DATA_DIR / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Sentiment data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
    
    def load_sentiment_data(self, filename: str) -> pd.DataFrame:
        """
        Load sentiment analysis data from file.
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame with sentiment data
        """
        try:
            filepath = DATA_DIR / filename
            if filepath.exists():
                df = pd.read_csv(filepath, parse_dates=['timestamp'])
                logger.info(f"Loaded sentiment data from {filepath}")
                return df
            else:
                logger.warning(f"Sentiment data file not found: {filepath}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return pd.DataFrame()


# Utility functions for sentiment analysis
def aggregate_sentiment_scores(scores: List[float], method: str = 'mean') -> float:
    """
    Aggregate multiple sentiment scores using specified method.
    
    Args:
        scores: List of sentiment scores
        method: Aggregation method ('mean', 'median', 'weighted_mean')
        
    Returns:
        Aggregated sentiment score
    """
    if not scores:
        return 0.0
    
    if method == 'mean':
        return np.mean(scores)
    elif method == 'median':
        return np.median(scores)
    elif method == 'weighted_mean':
        # Give more weight to recent scores
        weights = np.exp(np.linspace(-1, 0, len(scores)))
        return np.average(scores, weights=weights)
    else:
        return np.mean(scores)


def sentiment_momentum(scores: List[float], period: int = 5) -> float:
    """
    Calculate sentiment momentum over a period.
    
    Args:
        scores: List of sentiment scores
        period: Period for momentum calculation
        
    Returns:
        Sentiment momentum score
    """
    if len(scores) < period * 2:
        return 0.0
    
    recent = np.mean(scores[-period:])
    previous = np.mean(scores[-period*2:-period])
    
    return recent - previous


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        # Test single text analysis
        test_text = "Bitcoin shows strong bullish momentum with institutional adoption increasing"
        score = analyzer.get_sentiment_score(test_text)
        print(f"Sentiment score for test text: {score}")
        
        # Test batch analysis
        test_texts = [
            "Bitcoin price surges to new all-time high",
            "Cryptocurrency market faces regulatory uncertainty",
            "Institutional investors continue to buy digital assets"
        ]
        
        batch_scores = analyzer.get_batch_sentiment_scores(test_texts)
        print(f"Batch sentiment scores: {batch_scores}")
        
        # Test news sentiment analysis
        crypto_sentiment = analyzer.get_crypto_news_sentiment('BTC')
        print(f"Crypto news sentiment: {crypto_sentiment}")
        
        # Test sentiment features
        historical_scores = [0.1, 0.2, -0.1, 0.3, 0.0, 0.4, -0.2, 0.1]
        features = analyzer.create_sentiment_features(historical_scores)
        print(f"Sentiment features: {features}")
        
        # Test sentiment signal
        signal = analyzer.get_sentiment_signal(0.7)
        print(f"Sentiment signal: {signal}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")