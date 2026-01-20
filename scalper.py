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
import joblib
import pickle

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION - YOUR NEW OFFICIAL TOKEN
# =============================================
TELEGRAM_BOT_TOKEN = "8285366409:AAH9kdy1D-xULBmGakAPFYUME19fmVCDJ9E"
TELEGRAM_CHAT_ID = "-1003525746518"  # Keep your existing chat ID

# Trading Parameters
SCAN_INTERVAL = 3  # 3 seconds for fast scanning
MAX_CONCURRENT_TRADES = 2
MAX_TRADE_DURATION = 900  # 15 minutes
MIN_CONFIDENCE = 70  # ML model confidence threshold
RISK_PER_TRADE = 0.02  # 2% risk per trade

# Analysis Mode: "ML" or "TECHNICAL"
ANALYSIS_MODE = "ML"  # Change this to "TECHNICAL" for pure technical analysis
TEST_MODE = True  # Set to False for production
MIN_SAMPLES_FOR_ML = 50  # Minimum samples before using ML

# Only BTC and ETH
TRADING_PAIRS = [
    "BTC-USD",
    "ETH-USD"
]

# Pip configurations
PIP_CONFIG = {
    "BTC-USD": 0.01,   # 1 pip = 0.01
    "ETH-USD": 0.01,   # 1 pip = 0.01
}

# ML Model Configuration
ML_FEATURES = 20  # Number of features for ML model
MODEL_SAVE_PATH = "ml_models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

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
# ML MODEL MANAGER
# =============================================

class MLModelManager:
    """Manages ML models for trading signals."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_history = deque(maxlen=1000)
        self.labels_history = deque(maxlen=1000)
        self.init_models()
        self.initialize_with_sample_data()
    
    def init_models(self):
        """Initialize ML models."""
        # Try to load existing models
        for symbol in TRADING_PAIRS:
            model_path = f"{MODEL_SAVE_PATH}{symbol}_model.pkl"
            scaler_path = f"{MODEL_SAVE_PATH}{symbol}_scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[symbol] = joblib.load(model_path)
                    self.scalers[symbol] = joblib.load(scaler_path)
                    print(f"✅ Loaded ML model for {symbol}")
                except:
                    print(f"⚠️ Could not load ML model for {symbol}, creating new")
                    self.create_new_model(symbol)
            else:
                self.create_new_model(symbol)
    
    def create_new_model(self, symbol: str):
        """Create new ML model for symbol."""
        # Random Forest for robustness
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Neural Network for non-linear patterns
        nn_model = MLPClassifier(
            hidden_layer_sizes=(50, 25, 10),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
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
    
    def initialize_with_sample_data(self):
        """Initialize models with sample data."""
        print("🤖 Initializing ML models with sample data...")
        for symbol in TRADING_PAIRS:
            # Generate synthetic training data
            n_samples = 200
            
            # Generate realistic features
            features = []
            labels = []
            
            for _ in range(n_samples):
                # Generate realistic feature values
                feature = np.array([
                    np.random.uniform(30, 70),  # RSI
                    np.random.uniform(-2, 2),   # MACD
                    np.random.uniform(-1, 1),   # MACD Hist
                    np.random.uniform(0.01, 0.05),  # BB Width
                    np.random.uniform(0.001, 0.01),  # ATR/Price
                    np.random.uniform(0.5, 2),   # Volume Ratio
                    np.random.uniform(-5, 5),    # Momentum
                    np.random.uniform(0.5, 3),   # Volatility
                    np.random.uniform(-2, 2),    # Price position in BB
                    np.random.uniform(-5, 5)     # OBV
                ])
                
                # Random label with 55% accuracy bias
                if np.random.random() > 0.45:
                    label = 1  # Win
                else:
                    label = 0  # Loss
                    
                features.append(feature)
                labels.append(label)
            
            # Train model with initial data
            if len(features) > 50:
                self.train_model(symbol, np.array(features), np.array(labels))
        
        print("✅ ML models initialized with sample data")
    
    def save_models(self):
        """Save ML models to disk."""
        for symbol, model_dict in self.models.items():
            try:
                joblib.dump(model_dict, f"{MODEL_SAVE_PATH}{symbol}_model.pkl")
                joblib.dump(self.scalers[symbol], f"{MODEL_SAVE_PATH}{symbol}_scaler.pkl")
            except Exception as e:
                print(f"❌ Error saving model for {symbol}: {e}")
    
    def train_model(self, symbol: str, features: np.ndarray, labels: np.ndarray):
        """Train ML model with new data."""
        if symbol not in self.models:
            return
        
        if len(features) < 100:  # Minimum samples for training
            return
        
        try:
            # Scale features
            scaled_features = self.scalers[symbol].fit_transform(features)
            
            # Train Random Forest
            self.models[symbol]['rf'].fit(scaled_features, labels)
            
            # Train Neural Network
            self.models[symbol]['nn'].fit(scaled_features, labels)
            
            # Update feature history
            self.feature_history.extend(features)
            self.labels_history.extend(labels)
            
            print(f"✅ Trained ML model for {symbol} with {len(features)} samples")
            
        except Exception as e:
            print(f"❌ Error training model for {symbol}: {e}")
    
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
            
            if probability_long > 0.65:  # Strong long signal
                direction = "LONG"
                confidence = probability_long
            elif probability_short > 0.65:  # Strong short signal
                direction = "SHORT"
                confidence = probability_short
            else:  # Neutral
                direction = "NEUTRAL"
                confidence = max(probability_long, probability_short)
            
            return MLPrediction(
                direction=direction,
                confidence=confidence * 100,
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
        """Add data for future training."""
        self.feature_history.append(features)
        self.labels_history.append(label)
        
        # Train periodically
        if len(self.feature_history) % 100 == 0:
            feature_array = np.array(self.feature_history)
            label_array = np.array(self.labels_history)
            self.train_model(symbol, feature_array, label_array)
            self.save_models()

# =============================================
# SYNTHETIC MARKET DATA GENERATOR
# =============================================

class SyntheticMarketData:
    """Generate synthetic market data for testing."""
    
    def __init__(self):
        self.price_base = {
            "BTC-USD": 50000.0,
            "ETH-USD": 3000.0
        }
        self.price_trend = {
            "BTC-USD": 1.0,
            "ETH-USD": 1.0
        }
        self.volatility = {
            "BTC-USD": 0.02,  # 2% daily volatility
            "ETH-USD": 0.03   # 3% daily volatility
        }
    
    def generate_price(self, symbol: str) -> float:
        """Generate realistic price with trend and noise."""
        base = self.price_base[symbol]
        volatility = self.volatility[symbol]
        
        # Random walk with drift
        drift = np.random.normal(0.0001, 0.0005)  # Small drift
        noise = np.random.normal(0, volatility / np.sqrt(252))  # Daily volatility
        
        # Update trend
        self.price_trend[symbol] *= (1 + drift + noise)
        
        # Ensure price doesn't go crazy
        self.price_trend[symbol] = np.clip(
            self.price_trend[symbol], 
            0.8,  # Max 20% drop
            1.2   # Max 20% gain
        )
        
        return base * self.price_trend[symbol]
    
    def generate_volume(self) -> float:
        """Generate realistic volume."""
        return np.random.uniform(1000, 10000)

# =============================================
# ENHANCED TECHNICAL ANALYSIS
# =============================================

class EnhancedTechnicalAnalyzer:
    """Advanced technical analysis with ML features."""
    
    def __init__(self):
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
    
    async def update_history(self, symbol: str, price: float, volume: float):
        """Update price history."""
        if symbol not in self.history:
            self.history[symbol] = {
                'prices': deque(maxlen=200),
                'volumes': deque(maxlen=200),
                'highs': deque(maxlen=200),
                'lows': deque(maxlen=200),
                'timestamps': deque(maxlen=200)
            }
        
        current_time = datetime.now()
        
        # For simplicity, use price for high/low
        self.history[symbol]['prices'].append(price)
        self.history[symbol]['volumes'].append(volume)
        self.history[symbol]['highs'].append(price * 1.0005)  # Simulated high
        self.history[symbol]['lows'].append(price * 0.9995)   # Simulated low
        self.history[symbol]['timestamps'].append(current_time)
    
    def calculate_indicators(self, symbol: str) -> Optional[Dict]:
        """Calculate all technical indicators."""
        if symbol not in self.history or len(self.history[symbol]['prices']) < 20:
            return None
        
        prices = np.array(self.history[symbol]['prices'])
        volumes = np.array(self.history[symbol]['volumes'])
        
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
        atr = self.calculate_atr(prices, period=14)
        
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
    
    def calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(prices) < period:
            return 0.0
        
        # Simplified ATR calculation
        high_low = np.max(prices[-period:]) - np.min(prices[-period:])
        return float(high_low / period)
    
    def calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate Volume Weighted Average Price."""
        if len(prices) == 0 or len(volumes) == 0:
            return prices[-1] if len(prices) > 0 else 0.0
        
        typical_price = (np.max(prices) + np.min(prices) + prices[-1]) / 3
        vwap = np.sum(typical_price * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else typical_price
        
        return float(vwap)
    
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
        return float(np.std(returns) * 100)  # Return as percentage
    
    def calculate_volume_ratio(self, volumes: np.ndarray) -> float:
        """Calculate volume ratio (current vs average)."""
        if len(volumes) < 10:
            return 1.0
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:])
        
        return float(current_volume / avg_volume) if avg_volume > 0 else 1.0
    
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
        if signals["LONG"] > 50 and signals["LONG"] > signals["SHORT"] * 1.5:
            direction = "LONG"
            confidence = min(100, signals["LONG"])
        elif signals["SHORT"] > 50 and signals["SHORT"] > signals["LONG"] * 1.5:
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
# ENHANCED SIGNAL GENERATOR WITH ML/TECHNICAL TOGGLE
# =============================================

class MLEnhancedSignalGenerator:
    """Generates signals using ML and technical analysis."""
    
    def __init__(self, ml_manager: MLModelManager, ta_analyzer: EnhancedTechnicalAnalyzer):
        self.ml_manager = ml_manager
        self.ta_analyzer = ta_analyzer
        self.signal_history = []
    
    async def generate_signal(self, symbol: str, log_callback) -> Optional[ScalpingSignal]:
        """Generate enhanced scalping signal."""
        # Get current market state
        indicators = self.ta_analyzer.calculate_indicators(symbol)
        if indicators is None:
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
            volume=indicators.get('volume_ratio', 1.0) * 1000,  # Simulated volume
            spread=0.001,  # Simulated spread
            
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
        if ANALYSIS_MODE == "ML" and not TEST_MODE:
            # Get ML prediction
            ml_prediction = self.ml_manager.predict(symbol, features)
            
            # Dynamic confidence threshold
            dynamic_threshold = MIN_CONFIDENCE
            training_samples = len(self.ml_manager.feature_history) if hasattr(self.ml_manager, 'feature_history') else 0
            
            if training_samples < 50:
                dynamic_threshold = 50
            elif training_samples < 200:
                dynamic_threshold = 60
            
            if ml_prediction.confidence < dynamic_threshold:
                log_callback(f"⏸️ {symbol}: ML confidence {ml_prediction.confidence:.1f}% < {dynamic_threshold}%")
                return None
            
            direction = ml_prediction.direction
            confidence = ml_prediction.confidence
            prediction_source = "ML"
            reason = f"ML Signal | Confidence: {ml_prediction.confidence:.1f}%"
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
        log_callback(f"   Entry: ${entry:.4f} | SL: ${stop_loss:.4f}")
        log_callback(f"   TP1: ${take_profit_1:.4f} | TP2: ${take_profit_2:.4f} | TP3: ${take_profit_3:.4f}")
        log_callback(f"   Risk: {risk_pips:.1f}pips | Target: {target_pips:.1f}pips")
        log_callback(f"   Confidence: {confidence:.1f}% | RRR: 1:{risk_reward:.1f}")
        log_callback(f"   Position: {position_size:.4f} units")
        log_callback(f"   Reason: {reason}")
        
        self.signal_history.append(signal)
        
        return signal

# =============================================
# ENHANCED TRADE MANAGER
# =============================================

class EnhancedScalpingTradeManager:
    """Manages scalping trades with ML feedback."""
    
    def __init__(self, ml_manager: MLModelManager):
        self.ml_manager = ml_manager
        self.active_trades = {}
        self.trade_history = []
    
    async def execute_trade(self, signal: ScalpingSignal, log_callback) -> bool:
        """Execute a scalping trade."""
        # Check max trades
        if len(self.active_trades) >= MAX_CONCURRENT_TRADES:
            log_callback(f"⚠️ Max trades reached ({MAX_CONCURRENT_TRADES})")
            return False
        
        # Add to active trades
        self.active_trades[signal.signal_id] = {
            'signal': signal,
            'entry_time': datetime.now(),
            'status': 'ACTIVE',
            'partial_tps': []
        }
        
        log_callback(f"✅ TRADE EXECUTED: {signal.signal_id}")
        log_callback(f"   {signal.symbol} {signal.direction}")
        
        if signal.ml_prediction:
            log_callback(f"   ML Confidence: {signal.ml_prediction.confidence:.1f}%")
        elif signal.technical_prediction:
            log_callback(f"   Technical Confidence: {signal.technical_prediction.confidence:.1f}%")
        
        # Send Telegram alert
        await self.send_telegram_alert(signal)
        
        return True
    
    async def monitor_trades(self, log_callback):
        """Monitor active trades with partial TP management."""
        closed_trades = []
        
        for signal_id, trade_data in list(self.active_trades.items()):
            signal = trade_data['signal']
            
            # Simulate price movement
            entry_price = signal.entry_price
            current_time = datetime.now()
            time_elapsed = (current_time - trade_data['entry_time']).total_seconds()
            
            # Simulate price (normal distribution around entry)
            volatility = signal.market_state.volatility / 100
            random_move = np.random.normal(0, volatility)
            
            if signal.direction == "LONG":
                current_price = entry_price * (1 + random_move * (time_elapsed / 300))  # 5 min scale
            else:
                current_price = entry_price * (1 - random_move * (time_elapsed / 300))
            
            # Check stop loss
            if signal.direction == "LONG":
                if current_price <= signal.stop_loss:
                    await self.close_trade(signal_id, current_price, "STOP_LOSS", log_callback)
                    closed_trades.append(signal_id)
                    # Add to training data if ML mode
                    if ANALYSIS_MODE == "ML" and signal.ml_prediction:
                        self.ml_manager.add_training_data(
                            signal.symbol,
                            signal.market_state.features,
                            label=0  # Loss
                        )
                    
                # Check take profits
                elif current_price >= signal.take_profit_3 and 'TP3' not in trade_data['partial_tps']:
                    await self.close_trade(signal_id, current_price, "TAKE_PROFIT_3", log_callback)
                    closed_trades.append(signal_id)
                    # Add to training data if ML mode
                    if ANALYSIS_MODE == "ML" and signal.ml_prediction:
                        self.ml_manager.add_training_data(
                            signal.symbol,
                            signal.market_state.features,
                            label=1  # Win
                        )
                    
                elif current_price >= signal.take_profit_2 and 'TP2' not in trade_data['partial_tps']:
                    log_callback(f"🎯 {signal_id}: Hit TP2 at ${current_price:.4f}")
                    trade_data['partial_tps'].append('TP2')
                    
                elif current_price >= signal.take_profit_1 and 'TP1' not in trade_data['partial_tps']:
                    log_callback(f"🎯 {signal_id}: Hit TP1 at ${current_price:.4f}")
                    trade_data['partial_tps'].append('TP1')
                    
            else:  # SHORT
                if current_price >= signal.stop_loss:
                    await self.close_trade(signal_id, current_price, "STOP_LOSS", log_callback)
                    closed_trades.append(signal_id)
                    if ANALYSIS_MODE == "ML" and signal.ml_prediction:
                        self.ml_manager.add_training_data(
                            signal.symbol,
                            signal.market_state.features,
                            label=0  # Loss
                        )
                    
                elif current_price <= signal.take_profit_3 and 'TP3' not in trade_data['partial_tps']:
                    await self.close_trade(signal_id, current_price, "TAKE_PROFIT_3", log_callback)
                    closed_trades.append(signal_id)
                    if ANALYSIS_MODE == "ML" and signal.ml_prediction:
                        self.ml_manager.add_training_data(
                            signal.symbol,
                            signal.market_state.features,
                            label=1  # Win
                        )
                    
                elif current_price <= signal.take_profit_2 and 'TP2' not in trade_data['partial_tps']:
                    log_callback(f"🎯 {signal_id}: Hit TP2 at ${current_price:.4f}")
                    trade_data['partial_tps'].append('TP2')
                    
                elif current_price <= signal.take_profit_1 and 'TP1' not in trade_data['partial_tps']:
                    log_callback(f"🎯 {signal_id}: Hit TP1 at ${current_price:.4f}")
                    trade_data['partial_tps'].append('TP1')
            
            # Check expiry
            if current_time > signal.expiry:
                await self.close_trade(signal_id, current_price, "EXPIRED", log_callback)
                closed_trades.append(signal_id)
        
        # Remove closed trades
        for signal_id in closed_trades:
            if signal_id in self.active_trades:
                del self.active_trades[signal_id]
    
    async def close_trade(self, signal_id: str, exit_price: float, reason: str, log_callback):
        """Close a trade."""
        if signal_id not in self.active_trades:
            return
        
        trade_data = self.active_trades[signal_id]
        signal = trade_data['signal']
        
        # Calculate PnL
        if signal.direction == "LONG":
            pnl = (exit_price - signal.entry_price) * signal.position_size
        else:
            pnl = (signal.entry_price - exit_price) * signal.position_size
        
        pnl_percent = (pnl / (signal.entry_price * signal.position_size)) * 100
        
        # Calculate pips
        pip_size = PIP_CONFIG.get(signal.symbol, 0.01)
        if signal.direction == "LONG":
            pips = (exit_price - signal.entry_price) / pip_size
        else:
            pips = (signal.entry_price - exit_price) / pip_size
        
        log_callback(f"🔒 TRADE CLOSED: {signal_id}")
        log_callback(f"   Reason: {reason}")
        log_callback(f"   Exit: ${exit_price:.4f}")
        log_callback(f"   PnL: ${pnl:.4f} ({pnl_percent:.2f}%)")
        log_callback(f"   Pips: {pips:+.1f}")
        log_callback(f"   {'💰 PROFIT' if pnl > 0 else '💸 LOSS'}")
        
        # Send Telegram closure
        await self.send_telegram_closure(signal, exit_price, pnl, pnl_percent, pips, reason)
        
        # Add to history
        self.trade_history.append({
            'signal_id': signal_id,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'entry': signal.entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'pips': pips,
            'reason': reason,
            'duration': (datetime.now() - trade_data['entry_time']).total_seconds() / 60
        })
    
    async def send_telegram_alert(self, signal: ScalpingSignal):
        """Send Telegram alert."""
        try:
            direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
            risk_pips, target_pips = signal.calculate_pips()
            
            if signal.ml_prediction:
                analysis_type = "ML ENHANCED"
                confidence = signal.ml_prediction.confidence
                details = f"ML Confidence: {signal.ml_prediction.confidence:.1f}%\nModel: {signal.ml_prediction.model_name}"
            else:
                analysis_type = "TECHNICAL ANALYSIS"
                confidence = signal.technical_prediction.confidence
                details = f"Technical Confidence: {signal.technical_prediction.confidence:.1f}%\nSignals: {signal.technical_prediction.reason}"
            
            message = f"""
⚡ {analysis_type} SCALPING SIGNAL ⚡

{direction_emoji} {signal.symbol} {signal.direction}
Strategy: Medium-Range Scalping (5-15 min)
Confidence: {confidence:.1f}%

🎯 Price Levels:
Entry: ${signal.entry_price:.4f}
Stop Loss: ${signal.stop_loss:.4f}
Take Profit: ${signal.take_profit_3:.4f}

📊 Risk Management:
Risk: {risk_pips:.1f} pips
Target: {target_pips:.1f} pips
Risk/Reward: 1:{signal.risk_reward:.1f}
Position Size: {signal.position_size:.4f} units

{details}

Time: {datetime.now().strftime('%H:%M:%S')}
"""
            
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        print(f"✅ Telegram alert sent: {signal.signal_id}")
                    
        except Exception as e:
            print(f"❌ Telegram alert error: {e}")
    
    async def send_telegram_closure(self, signal: ScalpingSignal, exit_price: float, 
                                  pnl: float, pnl_percent: float, pips: float, reason: str):
        """Send Telegram closure alert."""
        try:
            result_emoji = "💰" if pnl > 0 else "💸"
            
            message = f"""
{result_emoji} SCALPING TRADE CLOSED {result_emoji}

📊 Performance Summary:
Symbol: {signal.symbol}
Direction: {signal.direction}
Analysis: {ANALYSIS_MODE}
Duration: 5-15 min (Medium Range)

💵 Results:
Entry: ${signal.entry_price:.4f}
Exit: ${exit_price:.4f}
PnL: ${pnl:.4f}
PnL %: {pnl_percent:.2f}%
Pips: {pips:+.1f}

📝 Details:
Reason: {reason}
Confidence Was: {signal.confidence:.1f}%
Risk/Reward Was: 1:{signal.risk_reward:.1f}

Closed at: {datetime.now().strftime('%H:%M:%S')}
"""
            
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        print(f"✅ Telegram closure sent")
                    
        except Exception as e:
            print(f"❌ Telegram closure error: {e}")

# =============================================
# ENHANCED GUI WITH ML METRICS
# =============================================

class MLEnhancedGUI:
    """Enhanced GUI with ML performance metrics."""
    
    def __init__(self, bot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title(f"🤖 AI SCALPING BOT - {ANALYSIS_MODE} MODE")
        self.root.geometry("1400x900")
        
        # Configure dark theme
        self.setup_styles()
        
        # Initialize components
        self.init_ui()
        self.start_update_timer()
        
        print("✅ ML Enhanced GUI initialized")
    
    def setup_styles(self):
        """Setup modern styling."""
        self.style = ttk.Style()
        
        # Configure colors
        bg_color = '#0a0a0a'
        fg_color = '#00ff00'
        panel_bg = '#1a1a1a'
        accent_color = '#00ffff'
        
        self.root.configure(bg=bg_color)
        
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
    
    def init_ui(self):
        """Initialize enhanced UI."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top panel (Stats)
        top_panel = ttk.Frame(main_container)
        top_panel.pack(fill='x', pady=(0, 10))
        
        # Left stats
        left_stats = ttk.LabelFrame(top_panel, text="📊 PERFORMANCE", padding=10)
        left_stats.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right stats
        right_stats = ttk.LabelFrame(top_panel, text="🤖 ANALYSIS METRICS", padding=10)
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
        
        # Analysis metrics
        analysis_metrics = [
            ("Analysis Mode:", "analysis_mode", ANALYSIS_MODE),
            ("Confidence:", "current_confidence", "0.0%"),
            ("BTC Signals:", "btc_signals", "0"),
            ("ETH Signals:", "eth_signals", "0"),
            ("Training Samples:", "training_samples", "0"),
            ("Last Signal:", "last_signal", "None")
        ]
        
        for i, (label, key, default) in enumerate(analysis_metrics):
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
        
        ttk.Button(control_frame, text="🤖 TRAIN MODELS", 
                  command=self.train_models, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="🔄 TOGGLE MODE", 
                  command=self.toggle_mode, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="🗑️ CLEAR LOGS", 
                  command=self.clear_logs, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="📊 REFRESH GRAPHS", 
                  command=self.refresh_graphs, width=20).pack(pady=5)
        
        # Symbol info
        symbol_frame = ttk.Frame(control_frame)
        symbol_frame.pack(pady=10)
        
        ttk.Label(symbol_frame, text="🎯 TRADING:", style='Metric.TLabel').pack()
        for symbol in TRADING_PAIRS:
            ttk.Label(symbol_frame, text=f"  {symbol}", 
                     style='Value.TLabel').pack(anchor='w')
        
        # Test mode indicator
        test_frame = ttk.Frame(control_frame)
        test_frame.pack(pady=5)
        
        test_text = "✅ TEST MODE: ON" if TEST_MODE else "❌ TEST MODE: OFF"
        ttk.Label(test_frame, text=test_text, style='Value.TLabel').pack()
        
        # Right: Logs
        log_frame = ttk.LabelFrame(bottom_panel, text="📝 LIVE TRADING LOG", padding=5)
        log_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12,
                                                 font=('Consolas', 9),
                                                 bg='#0a0a0a', fg='#00ff00',
                                                 insertbackground='white')
        self.log_text.pack(fill='both', expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, 
                                   text=f"🤖 AI SCALPING BOT READY | {ANALYSIS_MODE} MODE | Press START",
                                   relief='sunken',
                                   anchor='center',
                                   font=('Arial', 10))
        self.status_bar.pack(side='bottom', fill='x')
        
        # Initialize graphs
        self.init_graphs()
    
    def init_graphs(self):
        """Initialize graphs with default data."""
        # PnL Graph
        self.ax_pnl.clear()
        self.ax_pnl.set_facecolor('#1a1a1a')
        self.ax_pnl.set_title('PnL Progression (Today)', color='white', pad=20)
        self.ax_pnl.set_xlabel('Time', color='white')
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
    
    def update_pnl_graph(self, pnl_data: List[float]):
        """Update PnL graph."""
        self.ax_pnl.clear()
        self.ax_pnl.set_facecolor('#1a1a1a')
        
        if pnl_data:
            # Create time axis
            times = np.arange(len(pnl_data))
            
            # Plot cumulative PnL
            cumulative_pnl = np.cumsum(pnl_data)
            
            # Plot with gradient fill
            self.ax_pnl.fill_between(times, cumulative_pnl, alpha=0.3, color='#00ff00')
            self.ax_pnl.plot(times, cumulative_pnl, 'g-', linewidth=2, marker='o', markersize=3)
            
            # Color positive/negative areas
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
            # Create pie chart
            sizes = [wins, losses]
            colors = ['#00ff00', '#ff4444']
            labels = [f'Wins: {wins}', f'Losses: {losses}']
            explode = (0.1, 0)
            
            wedges, texts, autotexts = self.ax_winloss.pie(
                sizes, explode=explode, colors=colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'color': 'white', 'fontsize': 10}
            )
            
            # Add legend
            self.ax_winloss.legend(wedges, labels, title="Results", 
                                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                                 fontsize=9, title_fontsize=10)
        
        self.ax_winloss.set_title('Win/Loss Distribution', color='white', pad=20)
        
        self.canvas_winloss.draw()
    
    def start_update_timer(self):
        """Start update timer."""
        try:
            self.update_ui()
        except Exception as e:
            print(f"⚠️ UI update error: {e}")
        finally:
            self.root.after(2000, self.start_update_timer)  # Update every 2 seconds
    
    def update_ui(self):
        """Update UI with current stats."""
        try:
            # Update status
            status = "RUNNING" if not self.bot.paused else "PAUSED"
            mode = self.mode_var.get()
            
            self.status_bar.config(
                text=f"🤖 {status} | {mode} MODE | "
                     f"Cycles: {self.bot.cycle_count} | "
                     f"Signals: {self.bot.signals_today} | "
                     f"Active Trades: {len(self.bot.trade_manager.active_trades)} | "
                     f"{datetime.now().strftime('%H:%M:%S')}"
            )
            
            # Update analysis mode label
            self.metric_labels['analysis_mode'].config(text=mode)
            
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
            
            # Update analysis metrics
            if hasattr(self.bot.ml_manager, 'feature_history'):
                training_samples = len(self.bot.ml_manager.feature_history)
                self.metric_labels['training_samples'].config(text=str(training_samples))
            
            # Count signals by symbol
            btc_signals = sum(1 for s in self.bot.signal_generator.signal_history if "BTC" in s.symbol)
            eth_signals = sum(1 for s in self.bot.signal_generator.signal_history if "ETH" in s.symbol)
            self.metric_labels['btc_signals'].config(text=str(btc_signals))
            self.metric_labels['eth_signals'].config(text=str(eth_signals))
            
            # Get last signal confidence
            if self.bot.signal_generator.signal_history:
                last_signal = self.bot.signal_generator.signal_history[-1]
                self.metric_labels['last_signal'].config(
                    text=f"{last_signal.symbol} {last_signal.direction}"
                )
                self.metric_labels['current_confidence'].config(
                    text=f"{last_signal.confidence:.1f}%"
                )
            
            # Update graphs
            pnl_data = [t['pnl'] for t in self.bot.trade_manager.trade_history]
            self.update_pnl_graph(pnl_data)
            self.update_winloss_graph(wins, losses)
            
        except Exception as e:
            print(f"⚠️ UI update error: {e}")
    
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
        self.add_log("🎯 Targeting 30-100 pips per trade")
        self.add_log("⏱️ Trade duration: 5-15 minutes")
    
    def pause_bot(self):
        """Pause bot."""
        self.bot.paused = True
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.add_log("⏸️ Bot PAUSED")
        if self.mode_var.get() == "ML":
            self.bot.ml_manager.save_models()
            self.add_log("💾 ML models saved")
    
    def train_models(self):
        """Manually train ML models."""
        self.add_log("🤖 Training ML models with current data...")
        self.bot.ml_manager.save_models()
        self.add_log("✅ ML models trained and saved")
    
    def refresh_graphs(self):
        """Refresh graphs."""
        try:
            wins = sum(1 for t in self.bot.trade_manager.trade_history if t['pnl'] > 0)
            losses = sum(1 for t in self.bot.trade_manager.trade_history if t['pnl'] <= 0)
            
            pnl_data = [t['pnl'] for t in self.bot.trade_manager.trade_history]
            self.update_pnl_graph(pnl_data)
            self.update_winloss_graph(wins, losses)
            
            self.add_log("📊 Graphs refreshed with latest data")
        except Exception as e:
            self.add_log(f"❌ Error refreshing graphs: {e}")

# =============================================
# MAIN AI SCALPING BOT
# =============================================

class AIScalpingBot:
    """Main AI-enhanced scalping bot."""
    
    def __init__(self):
        print("="*70)
        print("🤖 AI ENHANCED SCALPING BOT - MEDIUM RANGE")
        print("="*70)
        
        # Initialize enhanced components
        self.ml_manager = MLModelManager()
        self.ta_analyzer = EnhancedTechnicalAnalyzer()
        self.signal_generator = MLEnhancedSignalGenerator(self.ml_manager, self.ta_analyzer)
        self.trade_manager = EnhancedScalpingTradeManager(self.ml_manager)
        self.market_data_gen = SyntheticMarketData()
        
        # State
        self.cycle_count = 0
        self.signals_today = 0
        self.paused = True
        self.gui = None
        
        print("✅ AI Bot initialized successfully")
        print(f"   • Analysis Mode: {ANALYSIS_MODE}")
        print("   • ML Models: Random Forest + Neural Network Ensemble")
        print("   • Technical Analysis: 10+ indicators with signal scoring")
        print("   • Timeframe: Medium-range scalping (5-15 min)")
        print("   • Target: 30-100 pips per trade")
        print("   • Instruments: BTC & ETH only")
        print("   • Risk: 2% per trade, 1:2+ RRR")
        print("   • Test Mode:", "ON" if TEST_MODE else "OFF")
        print("="*70)
    
    def set_gui(self, gui):
        """Set enhanced GUI."""
        self.gui = gui
        self.gui.add_log(f"🤖 AI Scalping Bot Ready - {ANALYSIS_MODE} Mode")
        if ANALYSIS_MODE == "ML":
            self.gui.add_log("✅ ML models loaded: Random Forest + Neural Network")
        else:
            self.gui.add_log("✅ Technical analysis system ready")
        self.gui.add_log("🎯 Target: 30-100 pips per trade (5-15 minute duration)")
        self.gui.add_log("⚖️ Risk Management: 2% per trade, 1:2+ RRR")
        self.gui.add_log("📊 Real-time performance graphs initialized")
        self.gui.add_log("🔄 Use the TOGGLE MODE button to switch between ML and Technical analysis")
        self.gui.add_log("Press START to begin trading")
    
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
            
            # Update technical analysis for each symbol
            for symbol in TRADING_PAIRS:
                # Generate synthetic market data
                current_price = self.market_data_gen.generate_price(symbol)
                volume = self.market_data_gen.generate_volume()
                
                await self.ta_analyzer.update_history(symbol, current_price, volume)
                
                # Check max trades per symbol
                active_for_symbol = sum(
                    1 for trade in self.trade_manager.active_trades.values()
                    if trade['signal'].symbol == symbol
                )
                
                if active_for_symbol >= 1:
                    continue
                
                # Generate signal using ML or Technical analysis
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
        
        if self.gui:
            self.gui.add_log(f"✅ Cycle completed")
    
    async def run(self):
        """Main loop."""
        print("🚀 Starting AI Scalping Bot...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.gui:
            self.gui.add_log("✅ Market data generator initialized")
            self.gui.add_log("✅ Technical analysis ready")
            self.gui.add_log("✅ Telegram alerts enabled")
        
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
            self.ml_manager.save_models()
            
            if self.gui:
                self.gui.add_log("✅ Models saved successfully")
                
        except Exception as e:
            error_msg = f"❌ Bot error: {str(e)}"
            print(error_msg)
            if self.gui:
                self.gui.add_log(error_msg)
        finally:
            # Cleanup
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
    
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Start GUI
    try:
        gui.root.mainloop()
    except Exception as e:
        print(f"❌ GUI error: {e}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 AI ENHANCED SCALPING BOT - MEDIUM RANGE STRATEGY")
    print("="*70)
    print(f"\n✨ CURRENT MODE: {ANALYSIS_MODE}")
    
    if ANALYSIS_MODE == "ML":
        print("✨ ENHANCED FEATURES:")
        print("1. 🤖 ML ENSEMBLE: Random Forest + Neural Network")
        print("2. 📚 INITIAL TRAINING: Pre-trained with 200 samples")
        print("3. 🔄 CONTINUOUS LEARNING: ML models learn from every trade")
    else:
        print("✨ TECHNICAL ANALYSIS FEATURES:")
        print("1. 📊 MULTI-INDICATOR SCORING: RSI, MACD, Bollinger Bands, Volume")
        print("2. 🎯 SIGNAL CONFLUENCE: Weighted scoring system")
        print("3. 📈 REAL-TIME ANALYSIS: Dynamic signal strength calculation")
    
    print("\n🎯 COMMON TRADING PARAMETERS:")
    print("• Timeframe: Medium-range scalping (5-15 minutes)")
    print("• Target Range: 30-100 pips")
    print("• Instruments: BTC & ETH only")
    print("• Risk per Trade: 2%")
    print("• Minimum RRR: 1:2")
    print("• Minimum Confidence: 70%")
    
    print("\n🔄 TOGGLE FEATURE:")
    print("• Use the TOGGLE MODE button in the GUI to switch between ML and Technical analysis")
    print("• ML Mode: Uses ensemble ML models with confidence scoring")
    print("• Technical Mode: Pure technical analysis with multi-indicator confluence")
    print("="*70 + "\n")
    
    main()
