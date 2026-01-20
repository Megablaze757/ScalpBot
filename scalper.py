# =============================================
# AI ENHANCED SCALPING BOT - MEDIUM RANGE
# =============================================

import asyncio
import json
import random
import time
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sqlite3
import yfinance as yf
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass
from enum import Enum
import aiohttp
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pickle

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION - TELEGRAM BOT TOKEN
# =============================================
TELEGRAM_BOT_TOKEN = "8285366409:AAH9kdy1D-xULBmGakAPFYUME19fmVCDJ9E"
TELEGRAM_CHAT_ID = "-1003525746518"  # Your channel chat ID

# Trading Parameters
SCAN_INTERVAL = 10  # 10 seconds for better API rate limiting
MAX_CONCURRENT_TRADES = 2
MAX_TRADE_DURATION = 900  # 15 minutes
MIN_CONFIDENCE = 70  # ML model confidence threshold
RISK_PER_TRADE = 0.02  # 2% risk per trade

# Analysis Mode: "ML" or "TECHNICAL"
ANALYSIS_MODE = "TECHNICAL"  # Start with technical for immediate signals
USE_LIVE_DATA = True  # Use real Yahoo Finance data
MIN_SAMPLES_FOR_ML = 50  # Minimum samples before using ML

# ML Training Parameters
TRAIN_WITH_HISTORICAL = True  # Train ML with historical data
HISTORICAL_DAYS = 365  # 1 year of historical data
TRAIN_INTERVAL = 3600  # Retrain every hour (3600 seconds)
BACKTEST_PERIOD = 30  # Backtest last 30 days

# Only BTC and ETH
TRADING_PAIRS = [
    "BTC-USD",
    "ETH-USD"
]

# Yahoo Finance symbols
YF_SYMBOLS = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD"
}

# Pip configurations
PIP_CONFIG = {
    "BTC-USD": 0.01,   # 1 pip = 0.01
    "ETH-USD": 0.01,   # 1 pip = 0.01
}

# ML Model Configuration
MODEL_SAVE_PATH = "ml_models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# =============================================
# TELEGRAM MANAGER WITH RETRY LOGIC
# =============================================

class TelegramManager:
    """Manages Telegram notifications with robust error handling."""
    
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.session = None
        self.test_connection()
    
    def test_connection(self):
        """Test Telegram connection on startup."""
        try:
            print("🔍 Testing Telegram connection...")
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    bot_name = data['result']['username']
                    print(f"✅ Telegram bot connected: @{bot_name}")
                    
                    # Send startup message
                    self.send_message_sync("🤖 AI Scalping Bot Started\n"
                                          f"Mode: {ANALYSIS_MODE}\n"
                                          f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    return True
                else:
                    print("❌ Telegram bot response not OK")
            else:
                print(f"❌ Telegram connection failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Telegram test error: {e}")
        
        return False
    
    def send_message_sync(self, message: str) -> bool:
        """Send Telegram message synchronously with retry logic."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                payload = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                }
                
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    return True
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff
                    print(f"⚠️ Telegram rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ Telegram error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Telegram network error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                print(f"⚠️ Telegram error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return False
    
    async def send_message_async(self, message: str) -> bool:
        """Send Telegram message asynchronously."""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    return response.status == 200
                    
        except Exception as e:
            print(f"⚠️ Telegram async error: {e}")
            # Fallback to sync method
            return self.send_message_sync(message)

# =============================================
# HISTORICAL DATA COLLECTOR FOR ML TRAINING
# =============================================

class HistoricalDataCollector:
    """Collects and prepares historical data for ML training."""
    
    def __init__(self):
        self.historical_data = {}
        self.training_features = {}
        self.training_labels = {}
        self.metrics = {}
        
    def fetch_historical_data(self, symbol: str, days: int = 365):
        """Fetch historical data for training."""
        try:
            print(f"📊 Fetching {days} days of historical data for {symbol}...")
            yf_symbol = YF_SYMBOLS.get(symbol, symbol)
            
            # Download historical data
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=f"{days}d", interval="1h")
            
            if len(data) > 100:  # Need enough data
                self.historical_data[symbol] = data
                print(f"✅ Loaded {len(data)} hours of historical data for {symbol}")
                return True
            else:
                print(f"⚠️ Insufficient historical data for {symbol}: {len(data)} records")
                
        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}")
            
        return False
    
    def prepare_training_data(self, symbol: str):
        """Prepare features and labels from historical data."""
        if symbol not in self.historical_data:
            return False
        
        try:
            data = self.historical_data[symbol]
            
            # Calculate technical indicators
            prices = data['Close'].values
            volumes = data['Volume'].values if 'Volume' in data.columns else np.ones_like(prices) * 1000
            highs = data['High'].values
            lows = data['Low'].values
            
            features = []
            labels = []
            
            # We need at least 50 periods to calculate indicators
            for i in range(50, len(prices) - 1):
                # Get window of prices
                window_prices = prices[i-50:i]
                window_volumes = volumes[i-50:i]
                window_highs = highs[i-50:i]
                window_lows = lows[i-50:i]
                
                # Calculate indicators for this window
                indicators = self.calculate_indicators(
                    window_prices, window_volumes, window_highs, window_lows
                )
                
                if indicators is None:
                    continue
                
                # Create feature vector
                feature = np.array([
                    indicators['rsi'] / 100,
                    indicators['macd'],
                    indicators['macd_histogram'],
                    indicators['bb_width'],
                    indicators['atr'] / prices[i] if prices[i] > 0 else 0,
                    indicators['volume_ratio'],
                    indicators['momentum'] / 100,
                    indicators['volatility'] / 100,
                    (prices[i] - indicators['bb_middle']) / indicators['bb_width'] if indicators['bb_width'] > 0 else 0,
                    indicators['obv'] / 1000000 if abs(indicators['obv']) > 0 else 0
                ])
                
                # Determine label (1 if price increased next period, 0 otherwise)
                future_price = prices[i + 1] if i + 1 < len(prices) else prices[i]
                label = 1 if future_price > prices[i] else 0
                
                features.append(feature)
                labels.append(label)
            
            if len(features) > 100:  # Need minimum samples
                self.training_features[symbol] = np.array(features)
                self.training_labels[symbol] = np.array(labels)
                print(f"✅ Prepared {len(features)} training samples for {symbol}")
                return True
                
        except Exception as e:
            print(f"❌ Error preparing training data for {symbol}: {e}")
            
        return False
    
    def calculate_indicators(self, prices, volumes, highs, lows):
        """Calculate technical indicators for training data."""
        try:
            if len(prices) < 20:
                return None
            
            current_price = prices[-1]
            
            # Calculate RSI
            rsi = self.calculate_rsi(prices, period=14)
            
            # Calculate MACD
            macd, signal, histogram = self.calculate_macd(prices)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self.calculate_bollinger_bands(prices, period=20)
            
            # Calculate ATR
            atr = self.calculate_atr(prices, highs, lows, period=14)
            
            # Calculate Volume Ratio
            volume_ratio = self.calculate_volume_ratio(volumes)
            
            # Calculate Momentum
            momentum = self.calculate_momentum(prices, period=10)
            
            # Calculate Volatility
            volatility = self.calculate_volatility(prices, period=20)
            
            # Calculate OBV
            obv = self.calculate_obv(prices, volumes)
            
            return {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal,
                'macd_histogram': histogram,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_width': bb_width,
                'atr': atr,
                'obv': obv,
                'momentum': momentum,
                'volatility': volatility,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return None
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return float(macd.iloc[-1]), float(signal.iloc[-1]), float(histogram.iloc[-1])
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            middle = np.mean(prices)
            std = np.std(prices) if len(prices) > 1 else 0
            return middle + 2*std, middle, middle - 2*std, 4*std
        
        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = middle + 2 * std
        lower = middle - 2 * std
        width = upper - lower
        
        return float(upper), float(middle), float(lower), float(width)
    
    def calculate_atr(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(prices) < period or len(highs) < period or len(lows) < period:
            return 0.0
        
        try:
            # Calculate true ranges
            tr = np.zeros(len(prices) - 1)
            for i in range(1, len(prices)):
                hl = highs[i] - lows[i]
                hc = abs(highs[i] - prices[i-1])
                lc = abs(lows[i] - prices[i-1])
                tr[i-1] = max(hl, hc, lc)
            
            # Calculate ATR
            atr = np.mean(tr[-period:])
            return float(atr)
            
        except:
            # Simplified ATR calculation
            high_low = np.max(prices[-period:]) - np.min(prices[-period:])
            return float(high_low / period)
    
    def calculate_volume_ratio(self, volumes: np.ndarray) -> float:
        """Calculate volume ratio (current vs average)."""
        if len(volumes) < 10:
            return 1.0
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:])
        
        return float(current_volume / avg_volume) if avg_volume > 0 else 1.0
    
    def calculate_momentum(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate momentum."""
        if len(prices) < period:
            return 0.0
        
        return float((prices[-1] / prices[-period] - 1) * 100)
    
    def calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate volatility."""
        if len(prices) < period:
            return 0.0
        
        returns = np.diff(prices[-period:]) / prices[-period:-1]
        return float(np.std(returns) * 100)
    
    def calculate_obv(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate On-Balance Volume."""
        if len(prices) < 2:
            return 0.0
        
        obv = 0.0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
        
        return obv
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate ML performance metrics."""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate confidence scores
            confidence_scores = np.max(y_proba, axis=1)
            avg_confidence = np.mean(confidence_scores) * 100
            
            return {
                'accuracy': accuracy * 100,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1 * 100,
                'avg_confidence': avg_confidence,
                'samples': len(y_true)
            }
        except:
            return None

# =============================================
# LIVE DATA FETCHER
# =============================================

class LiveDataFetcher:
    """Fetches live market data from Yahoo Finance."""
    
    def __init__(self):
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 30  # Cache for 30 seconds
        
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price from Yahoo Finance."""
        try:
            # Check cache
            current_time = time.time()
            if symbol in self.cache and symbol in self.cache_time:
                if current_time - self.cache_time[symbol] < self.cache_duration:
                    return self.cache[symbol]
            
            # Fetch from Yahoo Finance
            yf_symbol = YF_SYMBOLS.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            
            # Get current price
            data = ticker.history(period='1d', interval='1m')
            
            if len(data) > 0:
                current_price = data['Close'].iloc[-1]
                current_volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 1000
                
                # Update cache
                self.cache[symbol] = current_price
                self.cache_time[symbol] = current_time
                
                return current_price
                
        except Exception as e:
            print(f"⚠️ Error fetching data for {symbol}: {e}")
            
        return None
    
    def get_historical_data(self, symbol: str, period: str = '1d', interval: str = '1m') -> Optional[pd.DataFrame]:
        """Get historical data for technical analysis."""
        try:
            yf_symbol = YF_SYMBOLS.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if len(data) > 0:
                return data
                
        except Exception as e:
            print(f"⚠️ Error fetching historical data for {symbol}: {e}")
            
        return None

# =============================================
# ENHANCED DATA STRUCTURES
# =============================================

@dataclass
class MarketState:
    """Current market state with all indicators."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    spread: float
    
    # Technical indicators
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    atr: float
    vwap: float
    obv: float
    momentum: float
    volatility: float
    volume_ratio: float
    
    # ML features
    features: np.ndarray
    
    def to_feature_array(self) -> np.ndarray:
        """Convert to feature array for ML model."""
        return np.array([
            self.rsi / 100,
            self.macd,
            self.macd_histogram,
            self.bb_width,
            self.atr / self.price if self.price > 0 else 0,
            self.volume_ratio,
            self.momentum,
            self.volatility,
            (self.price - self.bb_middle) / self.bb_width if self.bb_width > 0 else 0,
            self.obv / 1000000 if abs(self.obv) > 0 else 0
        ])

@dataclass
class MLPrediction:
    """ML model prediction."""
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float
    probability_long: float
    probability_short: float
    features_used: np.ndarray
    model_name: str

@dataclass
class TechnicalPrediction:
    """Pure technical analysis prediction."""
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float
    rsi_signal: str
    macd_signal: str
    bb_signal: str
    volume_signal: str
    reason: str

@dataclass
class ScalpingSignal:
    """Scalping signal with ML enhancement."""
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    confidence: float
    risk_reward: float
    position_size: float
    ml_prediction: Optional[MLPrediction] = None
    technical_prediction: Optional[TechnicalPrediction] = None
    market_state: Optional[MarketState] = None
    reason: str = ""
    created_at: datetime = None
    expiry: datetime = None
    status: str = "PENDING"
    
    def calculate_pips(self) -> Tuple[float, float]:
        """Calculate risk and target in pips."""
        pip_size = PIP_CONFIG.get(self.symbol, 0.01)
        
        if self.direction == "LONG":
            risk_pips = (self.entry_price - self.stop_loss) / pip_size
            target_pips = (self.take_profit_3 - self.entry_price) / pip_size
        else:  # SHORT
            risk_pips = (self.stop_loss - self.entry_price) / pip_size
            target_pips = (self.entry_price - self.take_profit_3) / pip_size
        
        return abs(risk_pips), abs(target_pips)

# =============================================
# ENHANCED ML MODEL MANAGER WITH HISTORICAL TRAINING
# =============================================

class EnhancedMLModelManager:
    """Manages ML models with historical data training and testing."""
    
    def __init__(self, historical_collector: HistoricalDataCollector):
        self.models = {}
        self.scalers = {}
        self.feature_history = deque(maxlen=10000)
        self.labels_history = deque(maxlen=10000)
        self.historical_collector = historical_collector
        self.metrics = {}
        self.last_training_time = {}
        self.init_models()
        
        # Start background training if enabled
        if TRAIN_WITH_HISTORICAL:
            self.start_background_training()
    
    def init_models(self):
        """Initialize ML models."""
        # Try to load existing models
        for symbol in TRADING_PAIRS:
            model_path = f"{MODEL_SAVE_PATH}{symbol}_model.pkl"
            scaler_path = f"{MODEL_SAVE_PATH}{symbol}_scaler.pkl"
            metrics_path = f"{MODEL_SAVE_PATH}{symbol}_metrics.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[symbol] = joblib.load(model_path)
                    self.scalers[symbol] = joblib.load(scaler_path)
                    
                    # Load metrics if available
                    if os.path.exists(metrics_path):
                        self.metrics[symbol] = joblib.load(metrics_path)
                    
                    print(f"✅ Loaded ML model for {symbol}")
                    
                    # Check if metrics exist
                    if symbol not in self.metrics or not self.metrics[symbol]:
                        print(f"⚠️ No metrics found for {symbol}, will train with historical data")
                        if TRAIN_WITH_HISTORICAL:
                            self.train_with_historical_data(symbol)
                    
                except Exception as e:
                    print(f"⚠️ Could not load ML model for {symbol}: {e}, creating new")
                    self.create_new_model(symbol)
                    if TRAIN_WITH_HISTORICAL:
                        self.train_with_historical_data(symbol)
            else:
                self.create_new_model(symbol)
                if TRAIN_WITH_HISTORICAL:
                    self.train_with_historical_data(symbol)
    
    def create_new_model(self, symbol: str):
        """Create new ML model for symbol."""
        # Random Forest for robustness
        model = RandomForestClassifier(
            n_estimators=200,  # Increased for better accuracy
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Neural Network for non-linear patterns
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42,
            early_stopping=True
        )
        
        # Ensemble of both models
        self.models[symbol] = {
            'rf': model,
            'nn': nn_model,
            'ensemble_weights': [0.6, 0.4]  # Weighted ensemble
        }
        
        # Create scaler
        self.scalers[symbol] = StandardScaler()
        
        print(f"✅ Created new ML model for {symbol}")
    
    def start_background_training(self):
        """Start background training thread."""
        print("🚀 Starting background ML training...")
        
        def background_trainer():
            while True:
                try:
                    # Train models for all symbols
                    for symbol in TRADING_PAIRS:
                        # Check if it's time to retrain
                        current_time = time.time()
                        last_train = self.last_training_time.get(symbol, 0)
                        
                        if current_time - last_train > TRAIN_INTERVAL:
                            print(f"🔄 Retraining ML model for {symbol}...")
                            self.train_with_historical_data(symbol)
                            self.last_training_time[symbol] = current_time
                    
                    # Sleep for 1 minute before checking again
                    time.sleep(60)
                    
                except Exception as e:
                    print(f"⚠️ Background training error: {e}")
                    time.sleep(60)
        
        # Start background thread
        bg_thread = threading.Thread(target=background_trainer, daemon=True)
        bg_thread.start()
    
    def train_with_historical_data(self, symbol: str):
        """Train ML model with historical data."""
        try:
            print(f"📚 Training {symbol} ML model with historical data...")
            
            # Fetch historical data if not already loaded
            if symbol not in self.historical_collector.historical_data:
                if not self.historical_collector.fetch_historical_data(symbol, HISTORICAL_DAYS):
                    print(f"❌ Failed to fetch historical data for {symbol}")
                    return False
            
            # Prepare training data
            if not self.historical_collector.prepare_training_data(symbol):
                print(f"❌ Failed to prepare training data for {symbol}")
                return False
            
            if symbol not in self.historical_collector.training_features:
                print(f"❌ No training features available for {symbol}")
                return False
            
            # Get training data
            X = self.historical_collector.training_features[symbol]
            y = self.historical_collector.training_labels[symbol]
            
            if len(X) < 100:
                print(f"⚠️ Insufficient training data for {symbol}: {len(X)} samples")
                return False
            
            # Split data for training and testing (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scalers[symbol].fit_transform(X_train)
            X_test_scaled = self.scalers[symbol].transform(X_test)
            
            # Train Random Forest
            self.models[symbol]['rf'].fit(X_train_scaled, y_train)
            
            # Train Neural Network
            self.models[symbol]['nn'].fit(X_train_scaled, y_train)
            
            # Make predictions for evaluation
            rf_pred = self.models[symbol]['rf'].predict(X_test_scaled)
            nn_pred = self.models[symbol]['nn'].predict(X_test_scaled)
            
            # Ensemble predictions
            rf_proba = self.models[symbol]['rf'].predict_proba(X_test_scaled)
            nn_proba = self.models[symbol]['nn'].predict_proba(X_test_scaled)
            
            # Weighted ensemble probabilities
            weights = self.models[symbol]['ensemble_weights']
            ensemble_proba = (rf_proba * weights[0]) + (nn_proba * weights[1])
            ensemble_pred = np.argmax(ensemble_proba, axis=1)
            
            # Calculate metrics for each model
            rf_metrics = self.historical_collector.calculate_metrics(y_test, rf_pred, rf_proba)
            nn_metrics = self.historical_collector.calculate_metrics(y_test, nn_pred, nn_proba)
            ensemble_metrics = self.historical_collector.calculate_metrics(y_test, ensemble_pred, ensemble_proba)
            
            if rf_metrics and nn_metrics and ensemble_metrics:
                self.metrics[symbol] = {
                    'random_forest': rf_metrics,
                    'neural_network': nn_metrics,
                    'ensemble': ensemble_metrics,
                    'training_samples': len(X_train),
                    'testing_samples': len(X_test),
                    'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_samples': len(X)
                }
                
                # Save model and metrics
                self.save_models(symbol)
                
                print(f"✅ Successfully trained {symbol} ML model")
                print(f"   Training samples: {len(X_train)}")
                print(f"   Testing samples: {len(X_test)}")
                print(f"   Ensemble Accuracy: {ensemble_metrics['accuracy']:.1f}%")
                print(f"   Ensemble F1-Score: {ensemble_metrics['f1_score']:.1f}%")
                
                return True
                
        except Exception as e:
            print(f"❌ Error training ML model for {symbol}: {e}")
            traceback.print_exc()
        
        return False
    
    def save_models(self, symbol: str):
        """Save ML models and metrics to disk."""
        try:
            joblib.dump(self.models[symbol], f"{MODEL_SAVE_PATH}{symbol}_model.pkl")
            joblib.dump(self.scalers[symbol], f"{MODEL_SAVE_PATH}{symbol}_scaler.pkl")
            
            if symbol in self.metrics:
                joblib.dump(self.metrics[symbol], f"{MODEL_SAVE_PATH}{symbol}_metrics.pkl")
            
        except Exception as e:
            print(f"❌ Error saving model for {symbol}: {e}")
    
    def predict(self, symbol: str, features: np.ndarray) -> MLPrediction:
        """Make prediction using ensemble model."""
        if symbol not in self.models:
            return MLPrediction(
                direction="NEUTRAL",
                confidence=0,
                probability_long=0.5,
                probability_short=0.5,
                features_used=features,
                model_name="NO_MODEL"
            )
        
        try:
            # Scale features
            scaled_features = self.scalers[symbol].transform(features.reshape(1, -1))
            
            # Get predictions from both models
            rf_pred = self.models[symbol]['rf'].predict_proba(scaled_features)[0]
            nn_pred = self.models[symbol]['nn'].predict_proba(scaled_features)[0]
            
            # Ensemble prediction (weighted average)
            weights = self.models[symbol]['ensemble_weights']
            ensemble_proba = (rf_pred * weights[0]) + (nn_pred * weights[1])
            
            # Determine direction and confidence
            probability_long = ensemble_proba[1] if len(ensemble_proba) > 1 else 0.5
            probability_short = ensemble_proba[0] if len(ensemble_proba) > 0 else 0.5
            
            # Get model metrics for confidence adjustment
            model_confidence = 1.0
            if symbol in self.metrics and 'ensemble' in self.metrics[symbol]:
                model_confidence = self.metrics[symbol]['ensemble']['accuracy'] / 100
            
            if probability_long > 0.6:  # Long signal
                direction = "LONG"
                confidence = probability_long * model_confidence * 100
            elif probability_short > 0.6:  # Short signal
                direction = "SHORT"
                confidence = probability_short * model_confidence * 100
            else:  # Neutral
                direction = "NEUTRAL"
                confidence = max(probability_long, probability_short) * 100
            
            return MLPrediction(
                direction=direction,
                confidence=confidence,
                probability_long=probability_long,
                probability_short=probability_short,
                features_used=features,
                model_name="ENSEMBLE_RF_NN"
            )
            
        except Exception as e:
            print(f"❌ ML prediction error for {symbol}: {e}")
            return MLPrediction(
                direction="NEUTRAL",
                confidence=0,
                probability_long=0.5,
                probability_short=0.5,
                features_used=features,
                model_name="ERROR"
            )
    
    def add_training_data(self, symbol: str, features: np.ndarray, label: int):
        """Add live trading data for incremental learning."""
        self.feature_history.append(features)
        self.labels_history.append(label)
        
        # Train periodically with live data
        if len(self.feature_history) % 50 == 0:
            try:
                feature_array = np.array(self.feature_history)
                label_array = np.array(self.labels_history)
                
                if symbol in self.models:
                    # Scale features
                    scaled_features = self.scalers[symbol].transform(feature_array)
                    
                    # Incremental training
                    self.models[symbol]['rf'].fit(scaled_features, label_array)
                    self.models[symbol]['nn'].partial_fit(scaled_features, label_array, classes=[0, 1])
                    
                    print(f"✅ Incremental training with {len(feature_array)} live samples")
                    
            except Exception as e:
                print(f"❌ Incremental training error: {e}")
    
    def get_model_metrics(self, symbol: str) -> Dict:
        """Get metrics for a specific model."""
        if symbol in self.metrics:
            return self.metrics[symbol]
        return {}
    
    def get_all_metrics(self) -> Dict:
        """Get metrics for all models."""
        return self.metrics

# =============================================
# ENHANCED TECHNICAL ANALYSIS WITH LIVE DATA
# =============================================

class EnhancedTechnicalAnalyzer:
    """Advanced technical analysis with live data."""
    
    def __init__(self, data_fetcher: LiveDataFetcher):
        self.data_fetcher = data_fetcher
        self.history = {}
        self.initialize_history()
    
    def initialize_history(self):
        """Initialize history for all symbols."""
        for symbol in TRADING_PAIRS:
            self.history[symbol] = {
                'prices': deque(maxlen=200),
                'volumes': deque(maxlen=200),
                'highs': deque(maxlen=200),
                'lows': deque(maxlen=200),
                'timestamps': deque(maxlen=200)
            }
    
    async def update_history(self, symbol: str):
        """Update price history with live data."""
        try:
            # Get historical data
            data = self.data_fetcher.get_historical_data(symbol, period='1d', interval='1m')
            
            if data is not None and len(data) > 0:
                # Clear old history
                self.history[symbol]['prices'].clear()
                self.history[symbol]['volumes'].clear()
                self.history[symbol]['highs'].clear()
                self.history[symbol]['lows'].clear()
                self.history[symbol]['timestamps'].clear()
                
                # Fill with live data
                for idx in range(len(data)):
                    self.history[symbol]['prices'].append(float(data['Close'].iloc[idx]))
                    self.history[symbol]['volumes'].append(float(data['Volume'].iloc[idx]) if 'Volume' in data.columns else 1000)
                    self.history[symbol]['highs'].append(float(data['High'].iloc[idx]))
                    self.history[symbol]['lows'].append(float(data['Low'].iloc[idx]))
                    self.history[symbol]['timestamps'].append(datetime.now())
                
                return True
                
        except Exception as e:
            print(f"⚠️ Error updating history for {symbol}: {e}")
            
        return False
    
    def calculate_indicators(self, symbol: str) -> Optional[Dict]:
        """Calculate all technical indicators from live data."""
        try:
            if symbol not in self.history or len(self.history[symbol]['prices']) < 20:
                # Try to get fresh data
                data = self.data_fetcher.get_historical_data(symbol, period='1d', interval='1m')
                if data is None or len(data) < 20:
                    return None
                
                # Create history from data
                prices = data['Close'].values[-100:]  # Last 100 periods
                volumes = data['Volume'].values[-100:] if 'Volume' in data.columns else np.ones(100) * 1000
                highs = data['High'].values[-100:]
                lows = data['Low'].values[-100:]
            else:
                prices = np.array(self.history[symbol]['prices'])
                volumes = np.array(self.history[symbol]['volumes'])
                highs = np.array(self.history[symbol]['highs'])
                lows = np.array(self.history[symbol]['lows'])
            
            if len(prices) < 20:
                return None
            
            current_price = prices[-1]
            
            # Calculate RSI
            rsi = self.calculate_rsi(prices, period=14)
            
            # Calculate MACD
            macd, signal, histogram = self.calculate_macd(prices)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self.calculate_bollinger_bands(prices, period=20)
            
            # Calculate ATR
            atr = self.calculate_atr(prices, highs, lows, period=14)
            
            # Calculate VWAP
            vwap = self.calculate_vwap(prices, volumes)
            
            # Calculate OBV
            obv = self.calculate_obv(prices, volumes)
            
            # Calculate Momentum
            momentum = self.calculate_momentum(prices, period=10)
            
            # Calculate Volatility
            volatility = self.calculate_volatility(prices, period=20)
            
            # Calculate Volume Ratio
            volume_ratio = self.calculate_volume_ratio(volumes)
            
            return {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal,
                'macd_histogram': histogram,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'bb_width': bb_width,
                'atr': atr,
                'vwap': vwap,
                'obv': obv,
                'momentum': momentum,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating indicators for {symbol}: {e}")
            return None
    
    # ... (All the technical indicator calculation methods remain the same)
    # [Previous calculate_rsi, calculate_macd, etc. methods remain unchanged]
    
    def analyze_technical_signals(self, indicators: Dict) -> TechnicalPrediction:
        """Analyze technical indicators to generate prediction."""
        current_price = indicators['current_price']
        
        # Analyze RSI
        rsi = indicators['rsi']
        if rsi < 30:
            rsi_signal = "OVERSOLD"
            rsi_direction = "LONG"
            rsi_strength = (30 - rsi) / 30 * 100
        elif rsi > 70:
            rsi_signal = "OVERBOUGHT"
            rsi_direction = "SHORT"
            rsi_strength = (rsi - 70) / 30 * 100
        elif 30 <= rsi <= 50:
            rsi_signal = "NEUTRAL_TO_BULLISH"
            rsi_direction = "LONG"
            rsi_strength = (50 - rsi) / 20 * 50
        elif 50 <= rsi <= 70:
            rsi_signal = "NEUTRAL_TO_BEARISH"
            rsi_direction = "SHORT"
            rsi_strength = (rsi - 50) / 20 * 50
        else:
            rsi_signal = "NEUTRAL"
            rsi_direction = "NEUTRAL"
            rsi_strength = 0
        
        # Analyze MACD
        macd_hist = indicators['macd_histogram']
        macd = indicators['macd']
        signal = indicators['macd_signal']
        
        if macd_hist > 0 and macd > signal:
            macd_signal = "BULLISH"
            macd_direction = "LONG"
            macd_strength = abs(macd_hist) * 100
        elif macd_hist < 0 and macd < signal:
            macd_signal = "BEARISH"
            macd_direction = "SHORT"
            macd_strength = abs(macd_hist) * 100
        else:
            macd_signal = "NEUTRAL"
            macd_direction = "NEUTRAL"
            macd_strength = 0
        
        # Analyze Bollinger Bands
        bb_upper = indicators['bb_upper']
        bb_middle = indicators['bb_middle']
        bb_lower = indicators['bb_lower']
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        if current_price < bb_lower * 1.01:
            bb_signal = "OVERSOLD"
            bb_direction = "LONG"
            bb_strength = ((bb_lower * 1.01 - current_price) / bb_lower) * 200
        elif current_price > bb_upper * 0.99:
            bb_signal = "OVERBOUGHT"
            bb_direction = "SHORT"
            bb_strength = ((current_price - bb_upper * 0.99) / bb_upper) * 200
        elif bb_position < 0.3:
            bb_signal = "LOWER_BAND"
            bb_direction = "LONG"
            bb_strength = (0.3 - bb_position) * 100
        elif bb_position > 0.7:
            bb_signal = "UPPER_BAND"
            bb_direction = "SHORT"
            bb_strength = (bb_position - 0.7) * 100
        else:
            bb_signal = "MIDDLE_BAND"
            bb_direction = "NEUTRAL"
            bb_strength = 0
        
        # Analyze Volume
        volume_ratio = indicators['volume_ratio']
        if volume_ratio > 1.5:
            volume_signal = "HIGH_VOLUME"
            volume_strength = min(100, (volume_ratio - 1) * 50)
        elif volume_ratio < 0.5:
            volume_signal = "LOW_VOLUME"
            volume_strength = min(100, (1 - volume_ratio) * 50)
        else:
            volume_signal = "NORMAL_VOLUME"
            volume_strength = 0
        
        # Combine signals
        signals = {
            "LONG": 0,
            "SHORT": 0,
            "NEUTRAL": 0
        }
        
        # Weight the signals
        signals[rsi_direction] += rsi_strength * 0.3
        signals[macd_direction] += macd_strength * 0.4
        signals[bb_direction] += bb_strength * 0.3
        
        # Determine final direction
        if signals["LONG"] > 60 and signals["LONG"] > signals["SHORT"] * 1.2:
            direction = "LONG"
            confidence = min(100, signals["LONG"])
        elif signals["SHORT"] > 60 and signals["SHORT"] > signals["LONG"] * 1.2:
            direction = "SHORT"
            confidence = min(100, signals["SHORT"])
        else:
            direction = "NEUTRAL"
            confidence = 0
        
        # Create technical prediction
        return TechnicalPrediction(
            direction=direction,
            confidence=confidence,
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            bb_signal=bb_signal,
            volume_signal=volume_signal,
            reason=f"RSI: {rsi_signal} ({rsi:.1f}), MACD: {macd_signal}, BB: {bb_signal}, Volume: {volume_signal}"
        )

# =============================================
# SIGNAL GENERATOR (SAME AS BEFORE)
# =============================================

class MLEnhancedSignalGenerator:
    """Generates signals using ML and technical analysis with live data."""
    
    def __init__(self, ml_manager: EnhancedMLModelManager, ta_analyzer: EnhancedTechnicalAnalyzer, telegram: TelegramManager):
        self.ml_manager = ml_manager
        self.ta_analyzer = ta_analyzer
        self.telegram = telegram
        self.signal_history = []
        self.data_fetcher = ta_analyzer.data_fetcher
    
    async def generate_signal(self, symbol: str, log_callback) -> Optional[ScalpingSignal]:
        """Generate enhanced scalping signal with live data."""
        try:
            # Update history with live data
            await self.ta_analyzer.update_history(symbol)
            
            # Get current market state
            indicators = self.ta_analyzer.calculate_indicators(symbol)
            if indicators is None:
                log_callback(f"⚠️ {symbol}: No indicators available")
                return None
            
            current_price = indicators['current_price']
            
            # Prepare features for ML
            features = np.array([
                indicators['rsi'] / 100,
                indicators['macd'],
                indicators['macd_histogram'],
                indicators['bb_width'],
                indicators['atr'] / current_price if current_price > 0 else 0,
                indicators['volume_ratio'],
                indicators['momentum'] / 100,
                indicators['volatility'] / 100,
                (current_price - indicators['bb_middle']) / indicators['bb_width'] if indicators['bb_width'] > 0 else 0,
                indicators['obv'] / 1000000 if abs(indicators['obv']) > 0 else 0
            ])
            
            # Create market state
            market_state = MarketState(
                symbol=symbol,
                timestamp=datetime.now(),
                price=current_price,
                volume=indicators.get('volume_ratio', 1.0) * 1000,
                spread=0.001,
                
                # Technical indicators
                rsi=indicators['rsi'],
                macd=indicators['macd'],
                macd_signal=indicators['macd_signal'],
                macd_histogram=indicators['macd_histogram'],
                bb_upper=indicators['bb_upper'],
                bb_middle=indicators['bb_middle'],
                bb_lower=indicators['bb_lower'],
                bb_width=indicators['bb_width'],
                atr=indicators['atr'],
                vwap=indicators['vwap'],
                obv=indicators['obv'],
                momentum=indicators['momentum'],
                volatility=indicators['volatility'],
                volume_ratio=indicators['volume_ratio'],
                
                # ML features
                features=features
            )
            
            # Use ML or Technical analysis based on mode
            if ANALYSIS_MODE == "ML":
                # Check if ML model is trained enough
                metrics = self.ml_manager.get_model_metrics(symbol)
                if not metrics or metrics.get('testing_samples', 0) < 100:
                    log_callback(f"⚠️ {symbol}: ML model needs more training ({metrics.get('testing_samples', 0)} samples)")
                    # Fall back to technical analysis
                    analysis_mode = "TECHNICAL"
                else:
                    analysis_mode = "ML"
            else:
                analysis_mode = "TECHNICAL"
            
            if analysis_mode == "ML":
                # Get ML prediction
                ml_prediction = self.ml_manager.predict(symbol, features)
                
                # Adjust confidence based on model performance
                if symbol in self.ml_manager.metrics:
                    model_accuracy = self.ml_manager.metrics[symbol]['ensemble']['accuracy']
                    adjusted_confidence = ml_prediction.confidence * (model_accuracy / 100)
                else:
                    adjusted_confidence = ml_prediction.confidence
                
                if adjusted_confidence < MIN_CONFIDENCE:
                    log_callback(f"⏸️ {symbol}: ML confidence {adjusted_confidence:.1f}% < {MIN_CONFIDENCE}%")
                    return None
                
                direction = ml_prediction.direction
                confidence = adjusted_confidence
                prediction_source = "ML"
                reason = f"ML Signal | Confidence: {confidence:.1f}% | Model Accuracy: {model_accuracy:.1f}%" if symbol in self.ml_manager.metrics else f"ML Signal | Confidence: {confidence:.1f}%"
                ml_pred = ml_prediction
                tech_pred = None
                
            else:
                # Use pure technical analysis
                technical_prediction = self.ta_analyzer.analyze_technical_signals(indicators)
                
                if technical_prediction.confidence < MIN_CONFIDENCE:
                    log_callback(f"⏸️ {symbol}: Technical confidence {technical_prediction.confidence:.1f}% < {MIN_CONFIDENCE}%")
                    return None
                
                direction = technical_prediction.direction
                confidence = technical_prediction.confidence
                prediction_source = "TECHNICAL"
                reason = technical_prediction.reason
                ml_pred = None
                tech_pred = technical_prediction
            
            if direction == "NEUTRAL":
                log_callback(f"⏸️ {symbol}: {prediction_source} neutral (Confidence: {confidence:.1f}%)")
                return None
            
            # Calculate targets based on ATR for 30-100 pip range
            atr_pips = indicators['atr'] / PIP_CONFIG.get(symbol, 0.01)
            
            # Base target: 2-3x ATR (medium-range scalping)
            if atr_pips < 10:
                target_multiplier = 3.0
            elif atr_pips < 20:
                target_multiplier = 2.5
            else:
                target_multiplier = 2.0
            
            # Calculate risk: 1x ATR
            risk_pips = atr_pips
            target_pips = atr_pips * target_multiplier
            
            # Ensure within 30-100 pip range
            target_pips = max(30, min(100, target_pips))
            risk_pips = min(risk_pips, target_pips / 2)  # Max risk is half target
            
            # Calculate price levels
            pip_size = PIP_CONFIG.get(symbol, 0.01)
            
            if direction == "LONG":
                entry = current_price
                stop_loss = entry - (risk_pips * pip_size)
                take_profit_1 = entry + (target_pips * 0.5 * pip_size)
                take_profit_2 = entry + (target_pips * 0.75 * pip_size)
                take_profit_3 = entry + (target_pips * pip_size)
            else:  # SHORT
                entry = current_price
                stop_loss = entry + (risk_pips * pip_size)
                take_profit_1 = entry - (target_pips * 0.5 * pip_size)
                take_profit_2 = entry - (target_pips * 0.75 * pip_size)
                take_profit_3 = entry - (target_pips * pip_size)
            
            # Calculate risk/reward
            risk_amount = abs(entry - stop_loss)
            reward_amount = abs(take_profit_3 - entry)
            risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Require minimum 1:2 RRR
            if risk_reward < 2.0:
                log_callback(f"⏸️ {symbol}: RRR 1:{risk_reward:.1f} < 1:2")
                return None
            
            # Calculate position size based on 2% risk
            position_size = (RISK_PER_TRADE * 10000) / risk_pips  # Simplified calculation
            
            # Create signal
            signal = ScalpingSignal(
                signal_id=f"SIG-{int(time.time())}-{random.randint(1000, 9999)}",
                symbol=symbol,
                direction=direction,
                entry_price=round(float(entry), 4),
                stop_loss=round(float(stop_loss), 4),
                take_profit_1=round(float(take_profit_1), 4),
                take_profit_2=round(float(take_profit_2), 4),
                take_profit_3=round(float(take_profit_3), 4),
                confidence=confidence,
                risk_reward=float(risk_reward),
                position_size=round(float(position_size), 4),
                ml_prediction=ml_pred,
                technical_prediction=tech_pred,
                market_state=market_state,
                reason=reason + f" | ATR: {atr_pips:.1f}pips | Target: {target_pips:.1f}pips",
                created_at=datetime.now(),
                expiry=datetime.now() + timedelta(minutes=15)
            )
            
            # Log signal
            risk_pips, target_pips = signal.calculate_pips()
            
            log_callback(f"🎯 {symbol} {direction} SIGNAL ({prediction_source})")
            log_callback(f"   Price: ${current_price:.2f}")
            log_callback(f"   Entry: ${entry:.2f} | SL: ${stop_loss:.2f}")
            log_callback(f"   TP1: ${take_profit_1:.2f} | TP2: ${take_profit_2:.2f} | TP3: ${take_profit_3:.2f}")
            log_callback(f"   Risk: {risk_pips:.1f}pips | Target: {target_pips:.1f}pips")
            log_callback(f"   Confidence: {confidence:.1f}% | RRR: 1:{risk_reward:.1f}")
            log_callback(f"   Position: {position_size:.4f} units")
            log_callback(f"   Reason: {reason}")
            
            self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            log_callback(f"❌ Error generating signal for {symbol}: {str(e)}")
            return None

# =============================================
# ENHANCED GUI WITH ML TESTING PAGE
# =============================================

class MLEnhancedGUI:
    """Enhanced GUI with ML testing metrics page."""
    
    def __init__(self, bot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title(f"🤖 AI SCALPING BOT - {ANALYSIS_MODE} MODE")
        self.root.geometry("1400x900")
        
        # Configure dark theme
        self.setup_styles()
        
        # Create notebook for multiple pages
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create pages
        self.create_trading_page()
        self.create_ml_testing_page()
        
        # Initialize components
        self.start_update_timer()
        
        print("✅ ML Enhanced GUI initialized with testing page")
    
    def setup_styles(self):
        """Setup modern styling."""
        self.style = ttk.Style()
        
        # Configure colors
        bg_color = '#0a0a0a'
        fg_color = '#00ff00'
        panel_bg = '#1a1a1a'
        accent_color = '#00ffff'
        
        self.root.configure(bg=bg_color)
        self.style.configure('TNotebook', background=bg_color)
        self.style.configure('TNotebook.Tab', background=panel_bg, foreground=fg_color)
        
        self.style.configure('Title.TLabel', 
                           background=bg_color, 
                           foreground=accent_color,
                           font=('Arial', 14, 'bold'))
        
        self.style.configure('Panel.TFrame', 
                           background=panel_bg)
        
        self.style.configure('Metric.TLabel',
                           background=panel_bg,
                           foreground=fg_color,
                           font=('Arial', 10))
        
        self.style.configure('Value.TLabel',
                           background=panel_bg,
                           foreground=accent_color,
                           font=('Arial', 10, 'bold'))
        
        self.style.configure('Good.TLabel',
                           background=panel_bg,
                           foreground='#00ff00',
                           font=('Arial', 10, 'bold'))
        
        self.style.configure('Fair.TLabel',
                           background=panel_bg,
                           foreground='#ffff00',
                           font=('Arial', 10, 'bold'))
        
        self.style.configure('Poor.TLabel',
                           background=panel_bg,
                           foreground='#ff4444',
                           font=('Arial', 10, 'bold'))
    
    def create_trading_page(self):
        """Create main trading page."""
        trading_page = ttk.Frame(self.notebook)
        self.notebook.add(trading_page, text="📈 TRADING")
        
        # Main container
        main_container = ttk.Frame(trading_page)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top panel (Stats)
        top_panel = ttk.Frame(main_container)
        top_panel.pack(fill='x', pady=(0, 10))
        
        # Left stats
        left_stats = ttk.LabelFrame(top_panel, text="📊 PERFORMANCE", padding=10)
        left_stats.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right stats
        right_stats = ttk.LabelFrame(top_panel, text="📈 LIVE DATA", padding=10)
        right_stats.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Performance metrics
        self.metric_labels = {}
        metrics_left = [
            ("Today PnL:", "today_pnl", "$0.00"),
            ("Today Pips:", "today_pips", "0.0"),
            ("Win Rate:", "win_rate", "0.0%"),
            ("Active Trades:", "active_trades", "0"),
            ("Avg Trade Duration:", "avg_duration", "0.0m"),
            ("Total Trades:", "total_trades", "0")
        ]
        
        for i, (label, key, default) in enumerate(metrics_left):
            row = ttk.Frame(left_stats)
            row.pack(fill='x', pady=2)
            
            ttk.Label(row, text=label, style='Metric.TLabel', width=20).pack(side='left')
            self.metric_labels[key] = ttk.Label(row, text=default, style='Value.TLabel')
            self.metric_labels[key].pack(side='right')
        
        # Live data metrics
        live_metrics = [
            ("BTC Price:", "btc_price", "$0.00"),
            ("ETH Price:", "eth_price", "$0.00"),
            ("BTC RSI:", "btc_rsi", "0.0"),
            ("ETH RSI:", "eth_rsi", "0.0"),
            ("Analysis Mode:", "analysis_mode", ANALYSIS_MODE),
            ("ML Status:", "ml_status", "🔄 Training")
        ]
        
        for i, (label, key, default) in enumerate(live_metrics):
            row = ttk.Frame(right_stats)
            row.pack(fill='x', pady=2)
            
            ttk.Label(row, text=label, style='Metric.TLabel', width=20).pack(side='left')
            self.metric_labels[key] = ttk.Label(row, text=default, style='Value.TLabel')
            self.metric_labels[key].pack(side='right')
        
        # Middle panel (Graphs)
        middle_panel = ttk.Frame(main_container)
        middle_panel.pack(fill='both', expand=True, pady=(0, 10))
        
        # PnL Graph
        pnl_frame = ttk.LabelFrame(middle_panel, text="📈 PnL PROGRESSION", padding=5)
        pnl_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.fig_pnl = Figure(figsize=(8, 4), dpi=80, facecolor='#1a1a1a')
        self.ax_pnl = self.fig_pnl.add_subplot(111)
        self.canvas_pnl = FigureCanvasTkAgg(self.fig_pnl, pnl_frame)
        self.canvas_pnl.get_tk_widget().pack(fill='both', expand=True)
        
        # Win/Loss Graph
        winloss_frame = ttk.LabelFrame(middle_panel, text="📊 WIN/LOSS DISTRIBUTION", padding=5)
        winloss_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.fig_winloss = Figure(figsize=(8, 4), dpi=80, facecolor='#1a1a1a')
        self.ax_winloss = self.fig_winloss.add_subplot(111)
        self.canvas_winloss = FigureCanvasTkAgg(self.fig_winloss, winloss_frame)
        self.canvas_winloss.get_tk_widget().pack(fill='both', expand=True)
        
        # Bottom panel (Logs & Controls)
        bottom_panel = ttk.Frame(main_container)
        bottom_panel.pack(fill='both', expand=True)
        
        # Left: Controls
        control_frame = ttk.LabelFrame(bottom_panel, text="⚙️ CONTROLS", padding=10)
        control_frame.pack(side='left', fill='both', padx=(0, 5))
        
        # Mode toggle
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(pady=5)
        
        ttk.Label(mode_frame, text="Analysis Mode:", style='Metric.TLabel').pack(side='left', padx=(0, 5))
        
        self.mode_var = tk.StringVar(value=ANALYSIS_MODE)
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, 
                                      values=["ML", "TECHNICAL"], state='readonly', width=15)
        self.mode_combo.pack(side='left')
        self.mode_combo.bind('<<ComboboxSelected>>', self.on_mode_change)
        
        # Control buttons
        self.start_btn = ttk.Button(control_frame, text="▶️ START BOT", 
                                   command=self.start_bot, width=20)
        self.start_btn.pack(pady=5)
        
        self.pause_btn = ttk.Button(control_frame, text="⏸️ PAUSE BOT", 
                                   command=self.pause_bot, width=20,
                                   state='disabled')
        self.pause_btn.pack(pady=5)
        
        ttk.Button(control_frame, text="🔄 TOGGLE MODE", 
                  command=self.toggle_mode, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="🤖 TRAIN ML NOW", 
                  command=self.train_ml_now, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="📡 TEST TELEGRAM", 
                  command=self.test_telegram, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="🗑️ CLEAR LOGS", 
                  command=self.clear_logs, width=20).pack(pady=5)
        
        # Symbol info
        symbol_frame = ttk.Frame(control_frame)
        symbol_frame.pack(pady=10)
        
        ttk.Label(symbol_frame, text="🎯 TRADING PAIRS:", style='Metric.TLabel').pack()
        for symbol in TRADING_PAIRS:
            ttk.Label(symbol_frame, text=f"  {symbol}", 
                     style='Value.TLabel').pack(anchor='w')
        
        # Status info
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(pady=5)
        
        ttk.Label(status_frame, text=f"📡 LIVE DATA: {'✅ ON' if USE_LIVE_DATA else '❌ OFF'}", 
                 style='Value.TLabel').pack()
        
        ttk.Label(status_frame, text=f"📚 HISTORICAL TRAINING: {'✅ ON' if TRAIN_WITH_HISTORICAL else '❌ OFF'}", 
                 style='Value.TLabel').pack()
        
        # Right: Logs
        log_frame = ttk.LabelFrame(bottom_panel, text="📝 LIVE TRADING LOG", padding=5)
        log_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12,
                                                 font=('Consolas', 9),
                                                 bg='#0a0a0a', fg='#00ff00',
                                                 insertbackground='white')
        self.log_text.pack(fill='both', expand=True)
        
        # Initialize graphs
        self.init_graphs()
    
    def create_ml_testing_page(self):
        """Create ML testing and metrics page."""
        testing_page = ttk.Frame(self.notebook)
        self.notebook.add(testing_page, text="🤖 ML TESTING")
        
        # Main container
        main_container = ttk.Frame(testing_page)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(title_frame, text="🤖 MACHINE LEARNING MODEL TESTING", 
                 style='Title.TLabel').pack()
        
        ttk.Label(title_frame, text="Real-time performance metrics and historical backtesting",
                 style='Metric.TLabel').pack()
        
        # Metrics display area
        metrics_container = ttk.Frame(main_container)
        metrics_container.pack(fill='both', expand=True)
        
        # Create notebook for each symbol
        self.metrics_notebook = ttk.Notebook(metrics_container)
        self.metrics_notebook.pack(fill='both', expand=True)
        
        # Create metrics pages for each symbol
        self.metrics_pages = {}
        self.metrics_labels = {}
        
        for symbol in TRADING_PAIRS:
            page = ttk.Frame(self.metrics_notebook)
            self.metrics_notebook.add(page, text=symbol)
            self.metrics_pages[symbol] = page
            
            # Create metrics display for this symbol
            self.create_symbol_metrics_page(symbol, page)
        
        # Controls at bottom
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(controls_frame, text="🔄 REFRESH METRICS", 
                  command=self.refresh_ml_metrics, width=20).pack(side='left', padx=5)
        
        ttk.Button(controls_frame, text="📊 BACKTEST MODELS", 
                  command=self.run_backtest, width=20).pack(side='left', padx=5)
        
        ttk.Button(controls_frame, text="💾 SAVE MODELS", 
                  command=self.save_ml_models, width=20).pack(side='left', padx=5)
        
        ttk.Button(controls_frame, text="📈 TRAINING PROGRESS", 
                  command=self.show_training_progress, width=20).pack(side='left', padx=5)
    
    def create_symbol_metrics_page(self, symbol: str, parent_frame):
        """Create metrics display for a specific symbol."""
        # Create scrollable frame
        canvas = tk.Canvas(parent_frame, bg='#0a0a0a', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Training Status
        status_frame = ttk.LabelFrame(scrollable_frame, text="📊 TRAINING STATUS", padding=10)
        status_frame.pack(fill='x', pady=5, padx=5)
        
        self.metrics_labels[symbol] = {}
        status_metrics = [
            ("Model Status:", "status", "Not Trained"),
            ("Last Trained:", "last_trained", "Never"),
            ("Training Samples:", "training_samples", "0"),
            ("Testing Samples:", "testing_samples", "0"),
            ("Total Samples:", "total_samples", "0"),
            ("Training Progress:", "progress", "0%")
        ]
        
        for label_text, key, default in status_metrics:
            row = ttk.Frame(status_frame)
            row.pack(fill='x', pady=2)
            
            ttk.Label(row, text=label_text, style='Metric.TLabel', width=20).pack(side='left')
            self.metrics_labels[symbol][f"status_{key}"] = ttk.Label(row, text=default, style='Value.TLabel')
            self.metrics_labels[symbol][f"status_{key}"].pack(side='right')
        
        # Ensemble Model Metrics
        ensemble_frame = ttk.LabelFrame(scrollable_frame, text="🏆 ENSEMBLE MODEL PERFORMANCE", padding=10)
        ensemble_frame.pack(fill='x', pady=5, padx=5)
        
        ensemble_metrics = [
            ("Accuracy:", "accuracy", "0.0%"),
            ("Precision:", "precision", "0.0%"),
            ("Recall:", "recall", "0.0%"),
            ("F1-Score:", "f1_score", "0.0%"),
            ("Avg Confidence:", "avg_confidence", "0.0%"),
            ("Model Quality:", "quality", "Poor")
        ]
        
        for label_text, key, default in ensemble_metrics:
            row = ttk.Frame(ensemble_frame)
            row.pack(fill='x', pady=2)
            
            ttk.Label(row, text=label_text, style='Metric.TLabel', width=20).pack(side='left')
            self.metrics_labels[symbol][f"ensemble_{key}"] = ttk.Label(row, text=default, style='Value.TLabel')
            self.metrics_labels[symbol][f"ensemble_{key}"].pack(side='right')
        
        # Random Forest Metrics
        rf_frame = ttk.LabelFrame(scrollable_frame, text="🌲 RANDOM FOREST PERFORMANCE", padding=10)
        rf_frame.pack(fill='x', pady=5, padx=5)
        
        rf_metrics = [
            ("Accuracy:", "accuracy", "0.0%"),
            ("Precision:", "precision", "0.0%"),
            ("Recall:", "recall", "0.0%"),
            ("F1-Score:", "f1_score", "0.0%"),
            ("Avg Confidence:", "avg_confidence", "0.0%")
        ]
        
        for label_text, key, default in rf_metrics:
            row = ttk.Frame(rf_frame)
            row.pack(fill='x', pady=2)
            
            ttk.Label(row, text=label_text, style='Metric.TLabel', width=20).pack(side='left')
            self.metrics_labels[symbol][f"rf_{key}"] = ttk.Label(row, text=default, style='Value.TLabel')
            self.metrics_labels[symbol][f"rf_{key}"].pack(side='right')
        
        # Neural Network Metrics
        nn_frame = ttk.LabelFrame(scrollable_frame, text="🧠 NEURAL NETWORK PERFORMANCE", padding=10)
        nn_frame.pack(fill='x', pady=5, padx=5)
        
        nn_metrics = [
            ("Accuracy:", "accuracy", "0.0%"),
            ("Precision:", "precision", "0.0%"),
            ("Recall:", "recall", "0.0%"),
            ("F1-Score:", "f1_score", "0.0%"),
            ("Avg Confidence:", "avg_confidence", "0.0%")
        ]
        
        for label_text, key, default in nn_metrics:
            row = ttk.Frame(nn_frame)
            row.pack(fill='x', pady=2)
            
            ttk.Label(row, text=label_text, style='Metric.TLabel', width=20).pack(side='left')
            self.metrics_labels[symbol][f"nn_{key}"] = ttk.Label(row, text=default, style='Value.TLabel')
            self.metrics_labels[symbol][f"nn_{key}"].pack(side='right')
        
        # Performance Graph Frame
        graph_frame = ttk.LabelFrame(scrollable_frame, text="📈 MODEL PERFORMANCE VISUALIZATION", padding=10)
        graph_frame.pack(fill='both', expand=True, pady=5, padx=5)
        
        # Create performance graph
        self.fig_metrics = Figure(figsize=(10, 4), dpi=80, facecolor='#1a1a1a')
        self.ax_metrics = self.fig_metrics.add_subplot(111)
        self.canvas_metrics = FigureCanvasTkAgg(self.fig_metrics, graph_frame)
        self.canvas_metrics.get_tk_widget().pack(fill='both', expand=True)
        
        # Recommendation Frame
        rec_frame = ttk.LabelFrame(scrollable_frame, text="💡 RECOMMENDATIONS", padding=10)
        rec_frame.pack(fill='x', pady=5, padx=5)
        
        self.metrics_labels[symbol]["recommendation"] = ttk.Label(
            rec_frame, 
            text="Model needs training with historical data.",
            style='Metric.TLabel',
            wraplength=600
        )
        self.metrics_labels[symbol]["recommendation"].pack(anchor='w')
    
    def init_graphs(self):
        """Initialize graphs with default data."""
        # PnL Graph
        self.ax_pnl.clear()
        self.ax_pnl.set_facecolor('#1a1a1a')
        self.ax_pnl.set_title('PnL Progression (Today)', color='white', pad=20)
        self.ax_pnl.set_xlabel('Trade Sequence', color='white')
        self.ax_pnl.set_ylabel('Cumulative PnL ($)', color='white')
        self.ax_pnl.tick_params(colors='white')
        self.ax_pnl.grid(True, alpha=0.3, color='gray')
        
        # Win/Loss Graph
        self.ax_winloss.clear()
        self.ax_winloss.set_facecolor('#1a1a1a')
        self.ax_winloss.set_title('Win/Loss Distribution', color='white', pad=20)
        self.ax_winloss.tick_params(colors='white')
        
        self.canvas_pnl.draw()
        self.canvas_winloss.draw()
        
        # Metrics Graph
        self.ax_metrics.clear()
        self.ax_metrics.set_facecolor('#1a1a1a')
        self.ax_metrics.set_title('Model Performance Comparison', color='white', pad=20)
        self.ax_metrics.tick_params(colors='white')
        self.ax_metrics.grid(True, alpha=0.3, color='gray')
        self.canvas_metrics.draw()
    
    def update_metrics_graph(self, symbol: str, metrics: Dict):
        """Update metrics graph with model performance."""
        self.ax_metrics.clear()
        self.ax_metrics.set_facecolor('#1a1a1a')
        
        if metrics and 'random_forest' in metrics and 'neural_network' in metrics and 'ensemble' in metrics:
            # Prepare data
            models = ['Random Forest', 'Neural Network', 'Ensemble']
            accuracy = [
                metrics['random_forest']['accuracy'],
                metrics['neural_network']['accuracy'],
                metrics['ensemble']['accuracy']
            ]
            
            f1_scores = [
                metrics['random_forest']['f1_score'],
                metrics['neural_network']['f1_score'],
                metrics['ensemble']['f1_score']
            ]
            
            x = np.arange(len(models))
            width = 0.35
            
            # Create grouped bar chart
            bars1 = self.ax_metrics.bar(x - width/2, accuracy, width, label='Accuracy', color='#00ff00')
            bars2 = self.ax_metrics.bar(x + width/2, f1_scores, width, label='F1-Score', color='#00ffff')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    self.ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 1,
                                        f'{height:.1f}%', ha='center', va='bottom',
                                        color='white', fontsize=8)
            
            # Customize chart
            self.ax_metrics.set_xlabel('Models', color='white')
            self.ax_metrics.set_ylabel('Score (%)', color='white')
            self.ax_metrics.set_title(f'{symbol} Model Performance', color='white', pad=20)
            self.ax_metrics.set_xticks(x)
            self.ax_metrics.set_xticklabels(models, color='white')
            self.ax_metrics.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
            self.ax_metrics.set_ylim([0, 100])
            
            # Add grid
            self.ax_metrics.grid(True, alpha=0.3, color='gray')
        
        else:
            self.ax_metrics.text(0.5, 0.5, 'No metrics available\nTrain model first',
                                ha='center', va='center', color='white', fontsize=12,
                                transform=self.ax_metrics.transAxes)
        
        self.canvas_metrics.draw()
    
    def update_ml_metrics_display(self):
        """Update ML metrics display with current data."""
        try:
            ml_metrics = self.bot.ml_manager.get_all_metrics()
            
            for symbol in TRADING_PAIRS:
                if symbol in ml_metrics:
                    metrics = ml_metrics[symbol]
                    
                    # Update status metrics
                    self.metrics_labels[symbol]["status_status"].config(
                        text="✅ Trained" if metrics.get('testing_samples', 0) > 0 else "🔄 Training"
                    )
                    
                    self.metrics_labels[symbol]["status_last_trained"].config(
                        text=metrics.get('last_trained', 'Never')
                    )
                    
                    self.metrics_labels[symbol]["status_training_samples"].config(
                        text=str(metrics.get('training_samples', 0))
                    )
                    
                    self.metrics_labels[symbol]["status_testing_samples"].config(
                        text=str(metrics.get('testing_samples', 0))
                    )
                    
                    self.metrics_labels[symbol]["status_total_samples"].config(
                        text=str(metrics.get('total_samples', 0))
                    )
                    
                    # Calculate progress
                    total_needed = 1000  # Target samples
                    current = metrics.get('total_samples', 0)
                    progress = min(100, (current / total_needed) * 100)
                    self.metrics_labels[symbol]["status_progress"].config(
                        text=f"{progress:.1f}%"
                    )
                    
                    # Update ensemble metrics with quality coloring
                    if 'ensemble' in metrics:
                        ensemble = metrics['ensemble']
                        
                        # Accuracy with quality color
                        accuracy = ensemble['accuracy']
                        accuracy_style = self.get_quality_style(accuracy)
                        self.metrics_labels[symbol]["ensemble_accuracy"].config(
                            text=f"{accuracy:.1f}%",
                            style=accuracy_style
                        )
                        
                        self.metrics_labels[symbol]["ensemble_precision"].config(
                            text=f"{ensemble['precision']:.1f}%"
                        )
                        
                        self.metrics_labels[symbol]["ensemble_recall"].config(
                            text=f"{ensemble['recall']:.1f}%"
                        )
                        
                        self.metrics_labels[symbol]["ensemble_f1_score"].config(
                            text=f"{ensemble['f1_score']:.1f}%"
                        )
                        
                        self.metrics_labels[symbol]["ensemble_avg_confidence"].config(
                            text=f"{ensemble['avg_confidence']:.1f}%"
                        )
                        
                        # Model quality rating
                        quality = self.get_model_quality(accuracy)
                        self.metrics_labels[symbol]["ensemble_quality"].config(
                            text=quality,
                            style=self.get_quality_style(accuracy)
                        )
                    
                    # Update Random Forest metrics
                    if 'random_forest' in metrics:
                        rf = metrics['random_forest']
                        self.metrics_labels[symbol]["rf_accuracy"].config(
                            text=f"{rf['accuracy']:.1f}%"
                        )
                        self.metrics_labels[symbol]["rf_precision"].config(
                            text=f"{rf['precision']:.1f}%"
                        )
                        self.metrics_labels[symbol]["rf_recall"].config(
                            text=f"{rf['recall']:.1f}%"
                        )
                        self.metrics_labels[symbol]["rf_f1_score"].config(
                            text=f"{rf['f1_score']:.1f}%"
                        )
                        self.metrics_labels[symbol]["rf_avg_confidence"].config(
                            text=f"{rf['avg_confidence']:.1f}%"
                        )
                    
                    # Update Neural Network metrics
                    if 'neural_network' in metrics:
                        nn = metrics['neural_network']
                        self.metrics_labels[symbol]["nn_accuracy"].config(
                            text=f"{nn['accuracy']:.1f}%"
                        )
                        self.metrics_labels[symbol]["nn_precision"].config(
                            text=f"{nn['precision']:.1f}%"
                        )
                        self.metrics_labels[symbol]["nn_recall"].config(
                            text=f"{nn['recall']:.1f}%"
                        )
                        self.metrics_labels[symbol]["nn_f1_score"].config(
                            text=f"{nn['f1_score']:.1f}%"
                        )
                        self.metrics_labels[symbol]["nn_avg_confidence"].config(
                            text=f"{nn['avg_confidence']:.1f}%"
                        )
                    
                    # Update recommendation
                    if metrics.get('testing_samples', 0) >= 100:
                        accuracy = metrics['ensemble']['accuracy'] if 'ensemble' in metrics else 0
                        if accuracy >= 70:
                            rec = "✅ Model ready for live trading"
                        elif accuracy >= 60:
                            rec = "⚠️ Model needs more training"
                        else:
                            rec = "❌ Model requires significant improvement"
                    else:
                        rec = "🔄 Collecting more training data..."
                    
                    self.metrics_labels[symbol]["recommendation"].config(text=rec)
                    
                    # Update graph
                    self.update_metrics_graph(symbol, metrics)
                
                else:
                    # No metrics yet
                    self.metrics_labels[symbol]["status_status"].config(text="❌ Not Trained")
                    self.metrics_labels[symbol]["recommendation"].config(
                        text="Model needs training with historical data. Click 'TRAIN ML NOW'."
                    )
            
            # Update ML status in trading page
            trained_models = sum(1 for symbol in TRADING_PAIRS 
                               if symbol in ml_metrics and ml_metrics[symbol].get('testing_samples', 0) > 0)
            
            if trained_models == len(TRADING_PAIRS):
                self.metric_labels['ml_status'].config(text="✅ Trained")
            elif trained_models > 0:
                self.metric_labels['ml_status'].config(text=f"🔄 {trained_models}/{len(TRADING_PAIRS)}")
            else:
                self.metric_labels['ml_status'].config(text="❌ Not Trained")
                
        except Exception as e:
            print(f"⚠️ Error updating ML metrics: {e}")
    
    def get_quality_style(self, accuracy: float) -> str:
        """Get style based on accuracy."""
        if accuracy >= 70:
            return 'Good.TLabel'
        elif accuracy >= 60:
            return 'Fair.TLabel'
        else:
            return 'Poor.TLabel'
    
    def get_model_quality(self, accuracy: float) -> str:
        """Get quality rating based on accuracy."""
        if accuracy >= 75:
            return "Excellent"
        elif accuracy >= 70:
            return "Good"
        elif accuracy >= 65:
            return "Fair"
        elif accuracy >= 60:
            return "Poor"
        else:
            return "Unreliable"
    
    def on_mode_change(self, event=None):
        """Handle mode change."""
        global ANALYSIS_MODE
        ANALYSIS_MODE = self.mode_var.get()
        self.root.title(f"🤖 AI SCALPING BOT - {ANALYSIS_MODE} MODE")
        self.add_log(f"🔄 Analysis mode changed to: {ANALYSIS_MODE}")
    
    def toggle_mode(self):
        """Toggle between ML and Technical analysis."""
        current_mode = self.mode_var.get()
        new_mode = "TECHNICAL" if current_mode == "ML" else "ML"
        self.mode_var.set(new_mode)
        self.on_mode_change()
    
    def train_ml_now(self):
        """Manually trigger ML training."""
        self.add_log("🤖 Starting manual ML training...")
        
        def train_thread():
            for symbol in TRADING_PAIRS:
                success = self.bot.ml_manager.train_with_historical_data(symbol)
                if success:
                    self.add_log(f"✅ {symbol}: ML model trained successfully")
                else:
                    self.add_log(f"❌ {symbol}: ML training failed")
            
            # Refresh metrics display
            self.refresh_ml_metrics()
        
        # Run in separate thread to avoid blocking GUI
        threading.Thread(target=train_thread, daemon=True).start()
    
    def test_telegram(self):
        """Test Telegram connection."""
        self.add_log("📡 Testing Telegram connection...")
        if self.bot.telegram.test_connection():
            self.add_log("✅ Telegram is working!")
        else:
            self.add_log("❌ Telegram test failed")
    
    def refresh_ml_metrics(self):
        """Refresh ML metrics display."""
        self.add_log("🔄 Refreshing ML metrics...")
        self.update_ml_metrics_display()
        self.add_log("✅ ML metrics refreshed")
    
    def run_backtest(self):
        """Run backtest on ML models."""
        self.add_log("📊 Running backtest on ML models...")
        
        def backtest_thread():
            # This would run a more comprehensive backtest
            # For now, just refresh metrics
            time.sleep(2)
            self.add_log("✅ Backtest completed (simulated)")
            self.refresh_ml_metrics()
        
        threading.Thread(target=backtest_thread, daemon=True).start()
    
    def save_ml_models(self):
        """Save ML models to disk."""
        self.add_log("💾 Saving ML models...")
        
        for symbol in TRADING_PAIRS:
            self.bot.ml_manager.save_models(symbol)
        
        self.add_log("✅ ML models saved successfully")
    
    def show_training_progress(self):
        """Show training progress dialog."""
        self.add_log("📈 Showing training progress...")
        
        # Switch to ML testing page
        self.notebook.select(1)  # Second tab is ML testing
        
        # Refresh metrics
        self.refresh_ml_metrics()
    
    def start_update_timer(self):
        """Start update timer."""
        try:
            self.update_ui()
        except Exception as e:
            print(f"⚠️ UI update error: {e}")
        finally:
            self.root.after(5000, self.start_update_timer)  # Update every 5 seconds
    
    def update_ui(self):
        """Update UI with current stats."""
        try:
            # Update status
            status = "RUNNING" if not self.bot.paused else "PAUSED"
            mode = self.mode_var.get()
            
            # Update trading metrics
            if USE_LIVE_DATA:
                data_fetcher = LiveDataFetcher()
                btc_price = data_fetcher.get_live_price("BTC-USD")
                eth_price = data_fetcher.get_live_price("ETH-USD")
                
                if btc_price:
                    self.metric_labels['btc_price'].config(text=f"${btc_price:.2f}")
                if eth_price:
                    self.metric_labels['eth_price'].config(text=f"${eth_price:.2f}")
            
            # Get performance data
            wins = sum(1 for t in self.bot.trade_manager.trade_history if t['pnl'] > 0)
            losses = sum(1 for t in self.bot.trade_manager.trade_history if t['pnl'] <= 0)
            total_trades = len(self.bot.trade_manager.trade_history)
            
            if total_trades > 0:
                win_rate = (wins / total_trades) * 100
                total_pnl = sum(t['pnl'] for t in self.bot.trade_manager.trade_history)
                total_pips = sum(t['pips'] for t in self.bot.trade_manager.trade_history)
                avg_duration = np.mean([t['duration'] for t in self.bot.trade_manager.trade_history]) if total_trades > 0 else 0
            else:
                win_rate = 0
                total_pnl = 0
                total_pips = 0
                avg_duration = 0
            
            # Update metrics
            self.metric_labels['today_pnl'].config(text=f"${total_pnl:.2f}")
            self.metric_labels['today_pips'].config(text=f"{total_pips:.1f}")
            self.metric_labels['win_rate'].config(text=f"{win_rate:.1f}%")
            self.metric_labels['active_trades'].config(text=str(len(self.bot.trade_manager.active_trades)))
            self.metric_labels['avg_duration'].config(text=f"{avg_duration:.1f}m")
            self.metric_labels['total_trades'].config(text=str(total_trades))
            
            # Update graphs
            pnl_data = [t['pnl'] for t in self.bot.trade_manager.trade_history]
            self.update_pnl_graph(pnl_data)
            self.update_winloss_graph(wins, losses)
            
            # Update ML metrics periodically
            if time.time() % 30 < 5:  # Update every 30 seconds
                self.update_ml_metrics_display()
            
        except Exception as e:
            print(f"⚠️ UI update error: {e}")
    
    def update_pnl_graph(self, pnl_data: List[float]):
        """Update PnL graph."""
        self.ax_pnl.clear()
        self.ax_pnl.set_facecolor('#1a1a1a')
        
        if pnl_data:
            times = np.arange(len(pnl_data))
            cumulative_pnl = np.cumsum(pnl_data)
            
            self.ax_pnl.fill_between(times, cumulative_pnl, alpha=0.3, color='#00ff00')
            self.ax_pnl.plot(times, cumulative_pnl, 'g-', linewidth=2, marker='o', markersize=3)
            
            for i in range(len(times)-1):
                color = '#00ff00' if pnl_data[i+1] > 0 else '#ff4444'
                self.ax_pnl.plot(times[i:i+2], cumulative_pnl[i:i+2], color=color, linewidth=2)
        
        self.ax_pnl.set_title('PnL Progression (Today)', color='white', pad=20)
        self.ax_pnl.set_xlabel('Trade Sequence', color='white')
        self.ax_pnl.set_ylabel('Cumulative PnL ($)', color='white')
        self.ax_pnl.tick_params(colors='white')
        self.ax_pnl.grid(True, alpha=0.3, color='gray')
        
        self.canvas_pnl.draw()
    
    def update_winloss_graph(self, wins: int, losses: int):
        """Update win/loss graph."""
        self.ax_winloss.clear()
        self.ax_winloss.set_facecolor('#1a1a1a')
        
        if wins + losses > 0:
            sizes = [wins, losses]
            colors = ['#00ff00', '#ff4444']
            labels = [f'Wins: {wins}', f'Losses: {losses}']
            explode = (0.1, 0)
            
            wedges, texts, autotexts = self.ax_winloss.pie(
                sizes, explode=explode, colors=colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'color': 'white', 'fontsize': 10}
            )
            
            self.ax_winloss.legend(wedges, labels, title="Results", 
                                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                                 fontsize=9, title_fontsize=10)
        
        self.ax_winloss.set_title('Win/Loss Distribution', color='white', pad=20)
        self.canvas_winloss.draw()
    
    def add_log(self, message: str):
        """Add message to log."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        except:
            pass
    
    def clear_logs(self):
        """Clear logs."""
        try:
            self.log_text.delete(1.0, tk.END)
            self.add_log("📝 Logs cleared")
        except:
            pass
    
    def start_bot(self):
        """Start bot."""
        self.bot.paused = False
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.add_log("▶️ AI Scalping Bot STARTED")
        self.add_log(f"🤖 Analysis Mode: {self.mode_var.get()}")
        self.add_log(f"📡 Live Data: {'ENABLED' if USE_LIVE_DATA else 'DISABLED'}")
        self.add_log(f"📚 Historical Training: {'ENABLED' if TRAIN_WITH_HISTORICAL else 'DISABLED'}")
        self.add_log("🎯 Targeting 30-100 pips per trade")
        self.add_log("⏱️ Trade duration: 5-15 minutes")
    
    def pause_bot(self):
        """Pause bot."""
        self.bot.paused = True
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.add_log("⏸️ Bot PAUSED")
        
        # Save ML models
        self.save_ml_models()
        self.add_log("💾 ML models saved")

# =============================================
# MAIN AI SCALPING BOT
# =============================================

class AIScalpingBot:
    """Main AI-enhanced scalping bot with historical training."""
    
    def __init__(self):
        print("="*70)
        print("🤖 AI ENHANCED SCALPING BOT - MEDIUM RANGE")
        print("="*70)
        
        # Initialize Telegram first
        print("📡 Initializing Telegram...")
        self.telegram = TelegramManager()
        
        # Initialize historical data collector
        print("📊 Initializing historical data collector...")
        self.historical_collector = HistoricalDataCollector()
        
        # Initialize live data fetcher
        print("📈 Initializing live data...")
        self.data_fetcher = LiveDataFetcher()
        
        # Initialize enhanced ML model manager
        print("🤖 Initializing ML models with historical training...")
        self.ml_manager = EnhancedMLModelManager(self.historical_collector)
        
        print("📊 Initializing technical analysis...")
        self.ta_analyzer = EnhancedTechnicalAnalyzer(self.data_fetcher)
        
        print("🎯 Initializing signal generator...")
        self.signal_generator = MLEnhancedSignalGenerator(self.ml_manager, self.ta_analyzer, self.telegram)
        
        print("💰 Initializing trade manager...")
        self.trade_manager = EnhancedScalpingTradeManager(self.ml_manager, self.telegram)
        
        # State
        self.cycle_count = 0
        self.signals_today = 0
        self.paused = True
        self.gui = None
        
        print("✅ AI Bot initialized successfully")
        print(f"   • Analysis Mode: {ANALYSIS_MODE}")
        print(f"   • Live Data: {'ENABLED' if USE_LIVE_DATA else 'DISABLED'}")
        print(f"   • Historical Training: {'ENABLED' if TRAIN_WITH_HISTORICAL else 'DISABLED'}")
        print(f"   • Historical Days: {HISTORICAL_DAYS}")
        print("   • Timeframe: Medium-range scalping (5-15 min)")
        print("   • Target: 30-100 pips per trade")
        print("   • Instruments: BTC & ETH only")
        print("   • Risk: 2% per trade, 1:2+ RRR")
        print("="*70)
    
    def set_gui(self, gui):
        """Set enhanced GUI."""
        self.gui = gui
        self.gui.add_log(f"🤖 AI Scalping Bot Ready - {ANALYSIS_MODE} Mode")
        self.gui.add_log(f"📡 Telegram: @TheUltimateScalperBot")
        self.gui.add_log(f"📊 Live Data: {'ENABLED' if USE_LIVE_DATA else 'DISABLED'}")
        self.gui.add_log(f"📚 Historical Training: {'ENABLED' if TRAIN_WITH_HISTORICAL else 'DISABLED'}")
        self.gui.add_log(f"📈 Training Period: {HISTORICAL_DAYS} days")
        self.gui.add_log("🎯 Target: 30-100 pips per trade (5-15 minute duration)")
        self.gui.add_log("⚖️ Risk Management: 2% per trade, 1:2+ RRR")
        self.gui.add_log("🔄 Check the ML TESTING tab for model performance")
        self.gui.add_log("Press START to begin live trading")
        
        # Show ML testing page initially to see training progress
        self.gui.notebook.select(1)
    
    async def run_cycle(self):
        """Run one trading cycle."""
        if self.paused:
            return
        
        self.cycle_count += 1
        
        if self.gui:
            self.gui.add_log(f"\n⚡ CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Monitor existing trades
            await self.trade_manager.monitor_trades(
                self.gui.add_log if self.gui else print
            )
            
            # Generate signals for each symbol
            for symbol in TRADING_PAIRS:
                # Check max trades per symbol
                active_for_symbol = sum(
                    1 for trade in self.trade_manager.active_trades.values()
                    if trade['signal'].symbol == symbol
                )
                
                if active_for_symbol >= 1:
                    continue
                
                # Generate signal
                signal = await self.signal_generator.generate_signal(
                    symbol,
                    self.gui.add_log if self.gui else print
                )
                
                # Execute if valid
                if signal:
                    if await self.trade_manager.execute_trade(signal, self.gui.add_log if self.gui else print):
                        self.signals_today += 1
                        
        except Exception as e:
            error_msg = f"❌ Cycle error: {str(e)}"
            if self.gui:
                self.gui.add_log(error_msg)
            else:
                print(error_msg)
                traceback.print_exc()
        
        if self.gui:
            self.gui.add_log(f"✅ Cycle completed")
    
    async def run(self):
        """Main loop."""
        print("🚀 Starting AI Scalping Bot...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.gui:
            self.gui.add_log("✅ All systems initialized")
            self.gui.add_log("✅ Live data streaming active")
            self.gui.add_log("✅ Telegram notifications enabled")
            
            if TRAIN_WITH_HISTORICAL:
                self.gui.add_log("✅ Background ML training started")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            if self.gui:
                self.gui.add_log("🛑 Bot stopped by user")
                self.gui.add_log("💾 Saving models...")
            
            # Save ML models
            for symbol in TRADING_PAIRS:
                self.ml_manager.save_models(symbol)
            
            # Send shutdown message
            self.telegram.send_message_sync(
                f"🤖 AI Scalping Bot Shutdown\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total Cycles: {self.cycle_count}\n"
                f"Total Signals: {self.signals_today}"
            )
            
            if self.gui:
                self.gui.add_log("✅ Models saved successfully")
                self.gui.add_log("📡 Shutdown message sent to Telegram")
                
        except Exception as e:
            error_msg = f"❌ Bot error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            if self.gui:
                self.gui.add_log(error_msg)
        finally:
            print("💾 Cleanup completed")

# =============================================
# MAIN ENTRY POINT
# =============================================

def main():
    """Start the AI-enhanced scalping bot."""
    bot = AIScalpingBot()
    
    # Create enhanced GUI
    gui = MLEnhancedGUI(bot)
    bot.set_gui(gui)
    
    # Run bot in thread
    def run_bot():
        try:
            asyncio.run(bot.run())
        except Exception as e:
            print(f"❌ Bot thread error: {e}")
            traceback.print_exc()
    
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Start GUI
    try:
        gui.root.mainloop()
    except Exception as e:
        print(f"❌ GUI error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 AI ENHANCED SCALPING BOT - MEDIUM RANGE STRATEGY")
    print("="*70)
    print(f"\n✨ CURRENT MODE: {ANALYSIS_MODE}")
    
    print("\n🚀 ENHANCED FEATURES:")
    print("1. 📡 LIVE DATA: Real-time Yahoo Finance prices")
    print("2. 📚 HISTORICAL TRAINING: ML models trained with 1-year data")
    print("3. 🏆 ENSEMBLE MODELS: Random Forest + Neural Network")
    print("4. 📊 ML TESTING PAGE: Detailed performance metrics")
    print("5. 🎯 DUAL MODE: ML Enhanced + Pure Technical Analysis")
    
    print("\n📈 ML TRAINING CONFIG:")
    print(f"• Historical Data: {HISTORICAL_DAYS} days")
    print(f"• Auto-retrain: Every {TRAIN_INTERVAL//3600} hours")
    print("• Performance Metrics: Accuracy, Precision, Recall, F1-Score")
    print("• Quality Rating: Excellent/Good/Fair/Poor/Unreliable")
    
    print("\n🎯 TRADING PARAMETERS:")
    print("• Timeframe: Medium-range scalping (5-15 minutes)")
    print("• Target Range: 30-100 pips")
    print("• Instruments: BTC & ETH only")
    print("• Risk per Trade: 2%")
    print("• Minimum RRR: 1:2")
    print("• Minimum Confidence: 70%")
    
    print("\n📡 TELEGRAM BOT:")
    print(f"• Bot: @TheUltimateScalperBot")
    print(f"• Token: {TELEGRAM_BOT_TOKEN[:10]}...")
    
    print("\n🔄 CONTROLS:")
    print("• Tab 1: Trading Dashboard")
    print("• Tab 2: ML Testing & Metrics")
    print("• TRAIN ML NOW: Manual training button")
    print("• TOGGLE MODE: Switch between ML and Technical analysis")
    print("="*70 + "\n")
    
    main()
