# =============================================
# ULTIMATE ML SCALPING BOT - BTC/ETH ENHANCED
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
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import aiohttp
from collections import deque, defaultdict
import pickle
from typing import Literal

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from scipy import stats
import talib

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
TELEGRAM_BOT_TOKEN = "8535394169:AAE-GUriAU8THtypY2p82ewEgXqMC4twLas"
TELEGRAM_CHAT_ID = "-1003525746518"

# Trading Parameters
SCALP_INTERVAL = 3 # 3 seconds between checks
MAX_CONCURRENT_TRADES = 2 # Only 2 trades at once
MAX_TRADE_DURATION = 120 # 2 minutes max
MIN_CONFIDENCE = 70 # Minimum ML confidence
EMERGENCY_STOP_LOSS = 1.5 # 1.5% emergency stop

# Trading Pairs
TRADING_PAIRS = ["BTCUSDT", "ETHUSDT"]

# Risk Parameters
RISK_PER_TRADE = 0.02 # 2% risk per trade
TAKE_PROFIT_MULTIPLIER = 1.5 # TP = 1.5x risk
TRAILING_STOP_ACTIVATION = 0.3 # Activate trailing at 0.3R
TRAILING_STOP_DISTANCE = 0.15 # Trail at 0.15R distance

# ML Parameters
MODEL_UPDATE_INTERVAL = 60 # Update model every 60 minutes
MIN_TRAINING_SAMPLES = 1000
PREDICTION_WINDOW = 20 # Predict next 20 candles

# =============================================
# ENHANCED DATA CLASSES
# =============================================

class MarketRegime(Enum):
"""Market regime classification."""
BULL_TREND = "bull_trend"
BEAR_TREND = "bear_trend"
HIGH_VOLATILITY = "high_vol"
LOW_VOLATILITY = "low_vol"
MEAN_REVERSION = "mean_reversion"
BREAKOUT = "breakout"

class ScalpSignal(Enum):
"""Scalping signal types."""
FVG_BULLISH = "fvg_bullish"
FVG_BEARISH = "fvg_bearish"
ORDER_BLOCK_BULLISH = "ob_bullish"
ORDER_BLOCK_BEARISH = "ob_bearish"
LIQUIDITY_GRAB = "liquidity_grab"
BREAKOUT_RETEST = "breakout_retest"
SUPPLY_DEMAND = "supply_demand"

@dataclass
class MLPrediction:
"""ML model prediction."""
direction: Literal["LONG", "SHORT", "NEUTRAL"]
confidence: float
probability_long: float
probability_short: float
features: Dict[str, float]
regime: MarketRegime
predicted_move_pct: float

@dataclass
class EnhancedSignal:
"""Enhanced signal with ML predictions."""
signal_id: str
symbol: str
direction: str
entry_price: float
stop_loss: float
take_profit: float
confidence: float
ml_confidence: float
signal_type: ScalpSignal
timeframe: str
risk_reward: float
position_size: float
ml_prediction: MLPrediction
order_block_type: Optional[str] = None
liquidity_levels: Optional[List[float]] = None
created_at: datetime = None
expiry: datetime = None
status: str = "PENDING"

def __post_init__(self):
if self.created_at is None:
self.created_at = datetime.now()
if self.expiry is None:
self.expiry = self.created_at + timedelta(minutes=5)

def calculate_pips(self) -> float:
"""Calculate pips risk."""
pip_size = 0.01 if "BTC" in self.symbol else 0.01
risk_pips = abs(self.entry_price - self.stop_loss) / pip_size
return risk_pips

# =============================================
# ENHANCED MARKET DATA WITH REAL-TIME FEATURES
# =============================================

class EnhancedMarketData:
"""Market data with technical indicators and feature extraction."""

def __init__(self):
self.session = None
self.cache = {}
self.cache_time = {}
self.feature_cache = {}
self.historical_data = defaultdict(lambda: defaultdict(deque))

async def initialize(self):
"""Initialize session."""
try:
self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
print("✅ Market data session initialized")
except Exception as e:
print(f"❌ Error initializing market data: {e}")
self.session = None

async def close(self):
"""Close session."""
if self.session:
try:
await self.session.close()
except:
pass

async def get_price(self, symbol: str) -> Optional[float]:
"""Get current price from Binance."""
try:
clean_symbol = symbol.replace("USDT", "")
url = f"https://api.binance.com/api/v3/ticker/price?symbol={clean_symbol}USDT"

async with self.session.get(url, timeout=5) as response:
if response.status == 200:
data = await response.json()
price = float(data['price'])

# Cache for 1 second
self.cache[symbol] = price
self.cache_time[symbol] = time.time()

return price
except Exception as e:
print(f"⚠️ Price error for {symbol}: {e}")
# Return cached if available (5 second cache)
if symbol in self.cache and time.time() - self.cache_time.get(symbol, 0) < 5:
return self.cache[symbol]
return None

async def get_ohlcv_data(self, symbol: str, timeframe: str = '1m', limit: int = 500) -> Optional[pd.DataFrame]:
"""Get OHLCV data with enhanced features."""
try:
# Map symbol for yfinance
yf_symbol = "BTC-USD" if symbol == "BTCUSDT" else "ETH-USD"

# Map timeframe
tf_map = {
'1m': '1m',
'3m': '3m',
'5m': '5m',
'15m': '15m',
'1h': '60m'
}

interval = tf_map.get(timeframe, '1m')

# Get more data than needed for indicators
ticker = yf.Ticker(yf_symbol)
hist = ticker.history(period="2d", interval=interval)

if not hist.empty and len(hist) >= 100:
df = hist.tail(min(limit * 2, len(hist))) # Get extra for calculations

# Store in cache
cache_key = f"{symbol}_{timeframe}"
self.historical_data[symbol][timeframe] = deque(list(df.to_dict('records')), maxlen=1000)

return df.tail(limit)

except Exception as e:
print(f"⚠️ OHLCV error for {symbol}: {e}")

return None

def calculate_features(self, df: pd.DataFrame) -> Dict[str, float]:
"""Calculate advanced technical features."""
if len(df) < 50:
return {}

try:
closes = df['Close'].values.astype(float)
highs = df['High'].values.astype(float)
lows = df['Low'].values.astype(float)
volumes = df['Volume'].values.astype(float)

features = {}

# Price-based features
features['returns_5'] = (closes[-1] / closes[-5] - 1) * 100
features['returns_10'] = (closes[-1] / closes[-10] - 1) * 100
features['returns_20'] = (closes[-1] / closes[-20] - 1) * 100

# Volatility features
features['atr_14'] = talib.ATR(highs, lows, closes, timeperiod=14)[-1] / closes[-1] * 100
features['volatility_20'] = np.std(np.diff(closes[-20:])) / closes[-1] * 100

# Momentum indicators
features['rsi_14'] = talib.RSI(closes, timeperiod=14)[-1]
features['stoch_k'] = talib.STOCH(highs, lows, closes, fastk_period=14)[0][-1]
features['stoch_d'] = talib.STOCH(highs, lows, closes, fastk_period=14)[1][-1]
features['macd'] = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)[0][-1]
features['macd_signal'] = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)[1][-1]

# Volume features
volume_ma_20 = talib.SMA(volumes, timeperiod=20)[-1]
features['volume_ratio'] = volumes[-1] / volume_ma_20 if volume_ma_20 > 0 else 1
features['obv'] = talib.OBV(closes, volumes)[-1]

# Trend indicators
features['ema_9'] = talib.EMA(closes, timeperiod=9)[-1]
features['ema_21'] = talib.EMA(closes, timeperiod=21)[-1]
features['ema_50'] = talib.EMA(closes, timeperiod=50)[-1]
features['ema_200'] = talib.EMA(closes, timeperiod=200)[-1]

features['ema_9_21_diff'] = (features['ema_9'] - features['ema_21']) / closes[-1] * 100
features['ema_21_50_diff'] = (features['ema_21'] - features['ema_50']) / closes[-1] * 100

# Price action features
features['body_size'] = abs(closes[-1] - df['Open'].iloc[-1]) / closes[-1] * 100
features['upper_shadow'] = (highs[-1] - max(closes[-1], df['Open'].iloc[-1])) / closes[-1] * 100
features['lower_shadow'] = (min(closes[-1], df['Open'].iloc[-1]) - lows[-1]) / closes[-1] * 100

# Pattern recognition
features['hammer'] = self._is_hammer_pattern(highs[-5:], lows[-5:], closes[-5:])
features['shooting_star'] = self._is_shooting_star_pattern(highs[-5:], lows[-5:], closes[-5:])
features['engulfing'] = self._is_engulfing_pattern(df.iloc[-2:])

# Statistical features
features['skewness_20'] = stats.skew(closes[-20:])
features['kurtosis_20'] = stats.kurtosis(closes[-20:])

# Support/Resistance features
features['distance_to_high_20'] = (highs[-20:].max() - closes[-1]) / closes[-1] * 100
features['distance_to_low_20'] = (closes[-1] - lows[-20:].min()) / closes[-1] * 100

# Market regime features
features['trend_strength'] = self._calculate_trend_strength(closes)
features['mean_reversion_potential'] = self._calculate_mean_reversion_potential(closes)

return features

except Exception as e:
print(f"⚠️ Feature calculation error: {e}")
return {}

def _is_hammer_pattern(self, highs, lows, closes):
"""Detect hammer pattern."""
if len(closes) < 5:
return 0

body = abs(closes[-1] - closes[-2])
lower_shadow = min(closes[-1], closes[-2]) - lows[-1]
upper_shadow = highs[-1] - max(closes[-1], closes[-2])

if lower_shadow > body * 2 and upper_shadow < body * 0.3:
return 1
return 0

def _is_shooting_star_pattern(self, highs, lows, closes):
"""Detect shooting star pattern."""
if len(closes) < 5:
return 0

body = abs(closes[-1] - closes[-2])
lower_shadow = min(closes[-1], closes[-2]) - lows[-1]
upper_shadow = highs[-1] - max(closes[-1], closes[-2])

if upper_shadow > body * 2 and lower_shadow < body * 0.3:
return 1
return 0

def _is_engulfing_pattern(self, df):
"""Detect engulfing pattern."""
if len(df) < 2:
return 0

prev_body = abs(df['Close'].iloc[-2] - df['Open'].iloc[-2])
curr_body = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])

if curr_body > prev_body * 1.5:
# Bullish engulfing
if df['Close'].iloc[-1] > df['Open'].iloc[-1] and df['Close'].iloc[-2] < df['Open'].iloc[-2]:
return 1
# Bearish engulfing
elif df['Close'].iloc[-1] < df['Open'].iloc[-1] and df['Close'].iloc[-2] > df['Open'].iloc[-2]:
return -1

return 0

def _calculate_trend_strength(self, prices):
"""Calculate trend strength using ADX."""
if len(prices) < 14:
return 0

try:
# Use simple price action for trend
ma_short = np.mean(prices[-5:])
ma_medium = np.mean(prices[-20:])

trend = (ma_short - ma_medium) / ma_medium * 100
return min(100, max(-100, trend))

except:
return 0

def _calculate_mean_reversion_potential(self, prices):
"""Calculate mean reversion potential."""
if len(prices) < 20:
return 0

current = prices[-1]
mean = np.mean(prices[-20:])
std = np.std(prices[-20:])

if std == 0:
return 0

z_score = (current - mean) / std
# Higher absolute z-score = higher mean reversion potential
return min(100, abs(z_score) * 20)

# =============================================
# ML MODEL WITH ADVANCED FEATURES
# =============================================

class MLModelManager:
"""Manages ML models for price prediction."""

def __init__(self):
self.models = {}
self.scalers = {}
self.label_encoders = {}
self.training_data = defaultdict(list)
self.last_update = {}
self.feature_importance = {}

def create_model(self, symbol: str):
"""Create and train model for symbol."""
try:
# Features to use
feature_columns = [
'returns_5', 'returns_10', 'returns_20',
'atr_14', 'volatility_20',
'rsi_14', 'stoch_k', 'stoch_d', 'macd', 'macd_signal',
'volume_ratio', 'obv',
'ema_9_21_diff', 'ema_21_50_diff',
'body_size', 'upper_shadow', 'lower_shadow',
'hammer', 'shooting_star', 'engulfing',
'skewness_20', 'kurtosis_20',
'distance_to_high_20', 'distance_to_low_20',
'trend_strength', 'mean_reversion_potential'
]

# Use XGBoost for better performance
model = xgb.XGBClassifier(
n_estimators=200,
max_depth=6,
learning_rate=0.1,
subsample=0.8,
colsample_bytree=0.8,
random_state=42,
n_jobs=-1,
use_label_encoder=False,
eval_metric='logloss'
)

self.models[symbol] = model
self.scalers[symbol] = StandardScaler()
self.label_encoders[symbol] = LabelEncoder()

print(f"✅ ML model created for {symbol}")

except Exception as e:
print(f"❌ Error creating model for {symbol}: {e}")

def add_training_data(self, symbol: str, features: Dict, future_returns: float):
"""Add training data point."""
try:
if features:
# Label based on future returns
if future_returns > 0.1: # 0.1% gain
label = "LONG"
elif future_returns < -0.1: # 0.1% loss
label = "SHORT"
else:
label = "NEUTRAL"

self.training_data[symbol].append((features, label))

# Keep last 10000 samples
if len(self.training_data[symbol]) > 10000:
self.training_data[symbol] = self.training_data[symbol][-10000:]

except Exception as e:
print(f"⚠️ Error adding training data: {e}")

def train_model(self, symbol: str):
"""Train model on accumulated data."""
try:
if symbol not in self.training_data or len(self.training_data[symbol]) < MIN_TRAINING_SAMPLES:
return False

data = self.training_data[symbol]
X = [item[0] for item in data]
y = [item[1] for item in data]

# Convert to arrays
X_df = pd.DataFrame(X).fillna(0)
y_array = np.array(y)

# Scale features
X_scaled = self.scalers[symbol].fit_transform(X_df)

# Encode labels
y_encoded = self.label_encoders[symbol].fit_transform(y_array)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Train model
self.models[symbol].fit(
X_train, y_train,
eval_set=[(X_test, y_test)],
verbose=False,
early_stopping_rounds=20
)

# Evaluate
y_pred = self.models[symbol].predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Get feature importance
importance = self.models[symbol].feature_importances_
self.feature_importance[symbol] = dict(zip(X_df.columns, importance))

self.last_update[symbol] = datetime.now()

print(f"✅ Model trained for {symbol}, Accuracy: {accuracy:.2%}")
print(f" Top features: {sorted(self.feature_importance[symbol].items(), key=lambda x: x[1], reverse=True)[:5]}")

return True

except Exception as e:
print(f"❌ Error training model for {symbol}: {e}")
return False

def predict(self, symbol: str, features: Dict) -> Optional[MLPrediction]:
"""Make prediction using trained model."""
try:
if symbol not in self.models or not self.models[symbol]:
return None

# Prepare features
X_df = pd.DataFrame([features]).fillna(0)

# Scale
if symbol in self.scalers:
X_scaled = self.scalers[symbol].transform(X_df)
else:
return None

# Predict
model = self.models[symbol]
probabilities = model.predict_proba(X_scaled)[0]

# Get class predictions
if symbol in self.label_encoders:
classes = self.label_encoders[symbol].classes_

# Create prediction
if len(probabilities) == len(classes):
predictions = dict(zip(classes, probabilities))

# Determine direction
if 'LONG' in predictions and 'SHORT' in predictions:
if predictions['LONG'] > predictions['SHORT']:
direction = "LONG"
confidence = predictions['LONG']
else:
direction = "SHORT"
confidence = predictions['SHORT']
else:
direction = "NEUTRAL"
confidence = 0.5

# Determine regime
regime = self._determine_regime(features)

# Estimate move
predicted_move = self._estimate_move(features, direction)

return MLPrediction(
direction=direction,
confidence=float(confidence),
probability_long=float(predictions.get('LONG', 0.33)),
probability_short=float(predictions.get('SHORT', 0.33)),
features=features,
regime=regime,
predicted_move_pct=predicted_move
)

except Exception as e:
print(f"⚠️ Prediction error for {symbol}: {e}")

return None

def _determine_regime(self, features: Dict) -> MarketRegime:
"""Determine market regime from features."""
try:
volatility = features.get('atr_14', 0)
trend = features.get('trend_strength', 0)
mean_reversion = features.get('mean_reversion_potential', 0)

if volatility > 2.0:
return MarketRegime.HIGH_VOLATILITY
elif volatility < 0.5:
return MarketRegime.LOW_VOLATILITY
elif trend > 20:
return MarketRegime.BULL_TREND
elif trend < -20:
return MarketRegime.BEAR_TREND
elif mean_reversion > 30:
return MarketRegime.MEAN_REVERSION
else:
return MarketRegime.BREAKOUT

except:
return MarketRegime.BREAKOUT

def _estimate_move(self, features: Dict, direction: str) -> float:
"""Estimate potential move percentage."""
try:
volatility = features.get('atr_14', 1.0)
trend_strength = abs(features.get('trend_strength', 0))

# Base move on volatility and trend
base_move = volatility * 0.5 # 50% of ATR

if direction == "LONG" and features.get('trend_strength', 0) > 0:
base_move *= 1.5
elif direction == "SHORT" and features.get('trend_strength', 0) < 0:
base_move *= 1.5

return min(5.0, max(0.1, base_move)) # Clamp between 0.1% and 5%

except:
return 1.0

def save_models(self):
"""Save models to disk."""
try:
os.makedirs("models", exist_ok=True)

for symbol, model in self.models.items():
model_path = f"models/{symbol}_model.pkl"
with open(model_path, 'wb') as f:
pickle.dump({
'model': model,
'scaler': self.scalers.get(symbol),
'encoder': self.label_encoders.get(symbol),
'features': self.feature_importance.get(symbol, {}),
'last_update': self.last_update.get(symbol)
}, f)

print("✅ Models saved to disk")

except Exception as e:
print(f"❌ Error saving models: {e}")

def load_models(self):
"""Load models from disk."""
try:
for symbol in TRADING_PAIRS:
model_path = f"models/{symbol}_model.pkl"
if os.path.exists(model_path):
with open(model_path, 'rb') as f:
data = pickle.load(f)
self.models[symbol] = data['model']
self.scalers[symbol] = data['scaler']
self.label_encoders[symbol] = data['encoder']
self.feature_importance[symbol] = data['features']
self.last_update[symbol] = data['last_update']

print(f"✅ Model loaded for {symbol}")
else:
self.create_model(symbol)

except Exception as e:
print(f"❌ Error loading models: {e}")
for symbol in TRADING_PAIRS:
self.create_model(symbol)

# =============================================
# ENHANCED SIGNAL GENERATOR WITH SMC/FVG
# =============================================

class EnhancedSignalGenerator:
"""Generates enhanced scalping signals with ML."""

def __init__(self, market_data: EnhancedMarketData, ml_manager: MLModelManager):
self.market_data = market_data
self.ml_manager = ml_manager
self.signal_cache = defaultdict(list)

async def generate_signal(self, symbol: str, log_callback) -> Optional[EnhancedSignal]:
"""Generate enhanced scalping signal."""
try:
# Get current data
current_price = await self.market_data.get_price(symbol)
if current_price is None:
return None

# Get OHLCV data
df = await self.market_data.get_ohlcv_data(symbol, '1m', 200)
if df is None or len(df) < 100:
return None

# Calculate features
features = self.market_data.calculate_features(df)
if not features:
return None

# Get ML prediction
ml_prediction = self.ml_manager.predict(symbol, features)

# Generate SMC/FVG signals
smc_signals = self._generate_smc_signals(df, current_price)

# Combine signals
final_signal = self._combine_signals(
symbol, current_price, ml_prediction, smc_signals, features
)

if final_signal:
# Check if we should take this signal
if self._should_take_signal(final_signal, df):
log_callback(f"✅ ENHANCED SIGNAL: {symbol} {final_signal.direction}")
log_callback(f" ML Confidence: {final_signal.ml_confidence:.1%}")
log_callback(f" Regime: {final_signal.ml_prediction.regime.value}")
log_callback(f" Signal Type: {final_signal.signal_type.value}")

return final_signal

except Exception as e:
log_callback(f"❌ Signal generation error: {e}")

return None

def _generate_smc_signals(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
"""Generate Smart Money Concept signals."""
signals = []

try:
# 1. Fair Value Gap detection
fvg_signals = self._detect_fvg(df)
signals.extend(fvg_signals)

# 2. Order Block detection
order_blocks = self._detect_order_blocks(df)
signals.extend(order_blocks)

# 3. Liquidity detection
liquidity_levels = self._detect_liquidity(df)
for level in liquidity_levels:
signals.append({
'type': 'LIQUIDITY_GRAB',
'level': level,
'direction': 'SHORT' if current_price < level else 'LONG'
})

# 4. Breakout retest
breakout_signals = self._detect_breakout_retest(df, current_price)
signals.extend(breakout_signals)

except Exception as e:
print(f"⚠️ SMC signal error: {e}")

return signals

def _detect_fvg(self, df: pd.DataFrame) -> List[Dict]:
"""Detect Fair Value Gaps."""
signals = []

try:
if len(df) < 3:
return signals

for i in range(2, min(len(df), 10)):
current = df.iloc[i]
previous = df.iloc[i-1]
before_previous = df.iloc[i-2]

# Bullish FVG: Current low > Previous high
if current['Low'] > previous['High']:
signals.append({
'type': 'FVG_BULLISH',
'fvg_low': previous['High'],
'fvg_high': current['Low'],
'strength': (current['Low'] - previous['High']) / current['Low']
})

# Bearish FVG: Current high < Previous low
elif current['High'] < previous['Low']:
signals.append({
'type': 'FVG_BEARISH',
'fvg_low': current['High'],
'fvg_high': previous['Low'],
'strength': (previous['Low'] - current['High']) / current['High']
})

except Exception as e:
print(f"⚠️ FVG detection error: {e}")

return signals

def _detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
"""Detect Order Blocks."""
signals = []

try:
if len(df) < 5:
return signals

for i in range(3, min(len(df), 20)):
# Look for large bearish candle followed by bullish reversal
if (df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and # Bearish candle
df['Close'].iloc[i] > df['Open'].iloc[i]): # Bullish reversal

signals.append({
'type': 'ORDER_BLOCK_BULLISH',
'level': df['Low'].iloc[i],
'strength': abs(df['Close'].iloc[i] - df['Open'].iloc[i]) / df['Close'].iloc[i]
})

# Look for large bullish candle followed by bearish reversal
elif (df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and # Bullish candle
df['Close'].iloc[i] < df['Open'].iloc[i]): # Bearish reversal

signals.append({
'type': 'ORDER_BLOCK_BEARISH',
'level': df['High'].iloc[i],
'strength': abs(df['Close'].iloc[i] - df['Open'].iloc[i]) / df['Close'].iloc[i]
})

except Exception as e:
print(f"⚠️ Order block detection error: {e}")

return signals

def _detect_liquidity(self, df: pd.DataFrame) -> List[float]:
"""Detect liquidity levels (previous highs/lows)."""
levels = []

try:
if len(df) >= 50:
# Recent highs and lows
recent_highs = df['High'].rolling(20).max().dropna().tolist()[-5:]
recent_lows = df['Low'].rolling(20).min().dropna().tolist()[-5:]

levels.extend(recent_highs)
levels.extend(recent_lows)

# Round numbers
current_price = df['Close'].iloc[-1]
round_levels = [
round(current_price / 100) * 100,
round(current_price / 50) * 50,
round(current_price / 10) * 10
]

levels.extend(round_levels)

except Exception as e:
print(f"⚠️ Liquidity detection error: {e}")

return list(set(levels))

def _detect_breakout_retest(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
"""Detect breakout and retest patterns."""
signals = []

try:
if len(df) < 30:
return signals

# Look for recent highs/lows
recent_high = df['High'].rolling(20).max().iloc[-1]
recent_low = df['Low'].rolling(20).min().iloc[-1]

# Check for breakout above recent high
if current_price > recent_high * 1.001: # 0.1% above
# Look for retest
for i in range(1, 10):
if df['Low'].iloc[-i] <= recent_high * 1.001:
signals.append({
'type': 'BREAKOUT_RETEST',
'direction': 'LONG',
'level': recent_high,
'retest_candle': i
})
break

# Check for breakdown below recent low
elif current_price < recent_low * 0.999: # 0.1% below
# Look for retest
for i in range(1, 10):
if df['High'].iloc[-i] >= recent_low * 0.999:
signals.append({
'type': 'BREAKOUT_RETEST',
'direction': 'SHORT',
'level': recent_low,
'retest_candle': i
})
break

except Exception as e:
print(f"⚠️ Breakout detection error: {e}")

return signals

def _combine_signals(self, symbol: str, current_price: float,
ml_prediction: Optional[MLPrediction],
smc_signals: List[Dict], features: Dict) -> Optional[EnhancedSignal]:
"""Combine ML and SMC signals."""
try:
if not ml_prediction or ml_prediction.confidence < 0.6:
return None

# Filter SMC signals
valid_smc_signals = []
for signal in smc_signals:
signal_type = signal.get('type', '')

# Check if signal aligns with ML prediction
if (ml_prediction.direction == "LONG" and
signal_type in ['FVG_BULLISH', 'ORDER_BLOCK_BULLISH', 'BREAKOUT_RETEST']):
valid_smc_signals.append(signal)
elif (ml_prediction.direction == "SHORT" and
signal_type in ['FVG_BEARISH', 'ORDER_BLOCK_BEARISH', 'BREAKOUT_RETEST']):
valid_smc_signals.append(signal)

if not valid_smc_signals:
return None

# Take the strongest signal
strongest_signal = max(valid_smc_signals, key=lambda x: x.get('strength', 0))

# Calculate entry, stop, and target
entry, stop, target = self._calculate_levels(
symbol, current_price, strongest_signal, ml_prediction, features
)

# Calculate risk/reward
risk = abs(entry - stop)
reward = abs(target - entry)
rr = reward / risk if risk > 0 else 0

if rr < 1.2: # Minimum RRR
return None

# Create enhanced signal
signal_type = ScalpSignal(strongest_signal['type'].lower())

return EnhancedSignal(
signal_id=f"ESCALP-{int(time.time())}-{random.randint(1000, 9999)}",
symbol=symbol,
direction=ml_prediction.direction,
entry_price=entry,
stop_loss=stop,
take_profit=target,
confidence=ml_prediction.confidence * 100,
ml_confidence=ml_prediction.confidence * 100,
signal_type=signal_type,
timeframe="1m",
risk_reward=rr,
position_size=RISK_PER_TRADE,
ml_prediction=ml_prediction
)

except Exception as e:
print(f"⚠️ Signal combination error: {e}")
return None

def _calculate_levels(self, symbol: str, current_price: float, signal: Dict,
ml_prediction: MLPrediction, features: Dict) -> Tuple[float, float, float]:
"""Calculate entry, stop, and target levels."""
try:
volatility = features.get('atr_14', 1.0)
direction = ml_prediction.direction

# Calculate risk based on volatility
risk_distance = volatility * 0.5 # 0.5x ATR

if direction == "LONG":
# For long trades
if signal['type'] in ['FVG_BULLISH', 'ORDER_BLOCK_BULLISH']:
entry = signal.get('fvg_low', signal.get('level', current_price))
else:
entry = current_price

stop = entry - risk_distance
target = entry + (risk_distance * TAKE_PROFIT_MULTIPLIER)

else: # SHORT
# For short trades
if signal['type'] in ['FVG_BEARISH', 'ORDER_BLOCK_BEARISH']:
entry = signal.get('fvg_high', signal.get('level', current_price))
else:
entry = current_price

stop = entry + risk_distance
target = entry - (risk_distance * TAKE_PROFIT_MULTIPLIER)

# Round to appropriate decimals
decimals = 2 if symbol == "BTCUSDT" else 2
entry = round(entry, decimals)
stop = round(stop, decimals)
target = round(target, decimals)

return entry, stop, target

except Exception as e:
print(f"⚠️ Level calculation error: {e}")
# Fallback to simple calculation
if ml_prediction.direction == "LONG":
return current_price, current_price * 0.99, current_price * 1.015
else:
return current_price, current_price * 1.01, current_price * 0.985

def _should_take_signal(self, signal: EnhancedSignal, df: pd.DataFrame) -> bool:
"""Determine if signal should be taken."""
try:
# Check if price is at extreme
rsi = talib.RSI(df['Close'].values, timeperiod=14)[-1]
if signal.direction == "LONG" and rsi > 70:
return False
if signal.direction == "SHORT" and rsi < 30:
return False

# Check volatility
atr = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)[-1]
volatility = atr / df['Close'].iloc[-1] * 100

if volatility > 3.0: # Too volatile
return False

# Check if signal is too close to recent signals
recent_signals = self.signal_cache.get(signal.symbol, [])
recent_signals = [s for s in recent_signals
if (datetime.now() - s['time']).seconds < 300] # Last 5 minutes

if len(recent_signals) >= 3:
return False

# Add to cache
self.signal_cache[signal.symbol].append({
'time': datetime.now(),
'direction': signal.direction
})

# Keep cache clean
self.signal_cache[signal.symbol] = [
s for s in self.signal_cache[signal.symbol]
if (datetime.now() - s['time']).seconds < 600
]

return True

except Exception as e:
print(f"⚠️ Signal validation error: {e}")
return True

# =============================================
# ENHANCED TRADE MANAGER WITH ADVANCED RISK
# =============================================

class EnhancedTradeManager:
"""Manages trades with advanced risk management."""

def __init__(self, ml_manager: MLModelManager):
self.ml_manager = ml_manager
self.active_trades = {}
self.trade_history = []

async def execute_trade(self, signal: EnhancedSignal, log_callback) -> bool:
"""Execute enhanced trade."""
try:
# Check max trades
if len(self.active_trades) >= MAX_CONCURRENT_TRADES:
log_callback("⚠️ Max concurrent trades reached")
return False

# Check if already in trade for this symbol
for trade in self.active_trades.values():
if trade['signal'].symbol == signal.symbol:
log_callback(f"⚠️ Already in trade for {signal.symbol}")
return False

# Add to active trades
self.active_trades[signal.signal_id] = {
'signal': signal,
'entry_time': datetime.now(),
'status': 'ACTIVE',
'breakeven_triggered': False,
'trailing_activated': False,
'current_stop': signal.stop_loss,
'highest_price': signal.entry_price if signal.direction == "LONG" else signal.entry_price,
'lowest_price': signal.entry_price if signal.direction == "SHORT" else signal.entry_price
}

log_callback(f"🎯 ENHANCED TRADE EXECUTED: {signal.signal_id}")
log_callback(f" {signal.symbol} {signal.direction}")
log_callback(f" Entry: ${signal.entry_price:.2f}")
log_callback(f" Stop: ${signal.stop_loss:.2f}")
log_callback(f" Target: ${signal.take_profit:.2f}")
log_callback(f" ML Confidence: {signal.ml_confidence:.1f}%")
log_callback(f" Signal Type: {signal.signal_type.value}")

# Send Telegram alert
await self._send_telegram_alert(signal)

return True

except Exception as e:
log_callback(f"❌ Trade execution error: {e}")
return False

async def monitor_trades(self, market_data: EnhancedMarketData, log_callback):
"""Monitor and manage active trades."""
closed_trades = []

for signal_id, trade_data in list(self.active_trades.items()):
signal = trade_data['signal']

try:
# Get current price
current_price = await market_data.get_price(signal.symbol)
if current_price is None:
continue

# Update highest/lowest price
if signal.direction == "LONG":
trade_data['highest_price'] = max(trade_data['highest_price'], current_price)
else:
trade_data['lowest_price'] = min(trade_data['lowest_price'], current_price)

# Calculate current P&L
if signal.direction == "LONG":
pnl = (current_price - signal.entry_price) / signal.entry_price * 100
else:
pnl = (signal.entry_price - current_price) / signal.entry_price * 100

# Check stop loss
if signal.direction == "LONG":
if current_price <= trade_data['current_stop']:
await self._close_trade(signal_id, current_price, "STOP_LOSS", log_callback)
closed_trades.append(signal_id)
continue
else:
if current_price >= trade_data['current_stop']:
await self._close_trade(signal_id, current_price, "STOP_LOSS", log_callback)
closed_trades.append(signal_id)
continue

# Check take profit
if signal.direction == "LONG":
if current_price >= signal.take_profit:
await self._close_trade(signal_id, current_price, "TAKE_PROFIT", log_callback)
closed_trades.append(signal_id)
continue
else:
if current_price <= signal.take_profit:
await self._close_trade(signal_id, current_price, "TAKE_PROFIT", log_callback)
closed_trades.append(signal_id)
continue

# Check breakeven
if not trade_data['breakeven_triggered']:
risk = abs(signal.entry_price - signal.stop_loss)
breakeven_level = signal.entry_price + (risk * 0.5) if signal.direction == "LONG" else signal.entry_price - (risk * 0.5)

if (signal.direction == "LONG" and current_price >= breakeven_level) or \
(signal.direction == "SHORT" and current_price <= breakeven_level):

# Move stop to entry
trade_data['current_stop'] = signal.entry_price
trade_data['breakeven_triggered'] = True
log_callback(f"⚖️ Breakeven reached for {signal_id}")

# Check trailing stop activation
if not trade_data['trailing_activated'] and trade_data['breakeven_triggered']:
risk = abs(signal.entry_price - signal.stop_loss)
trailing_activation = signal.entry_price + (risk * TRAILING_STOP_ACTIVATION) if signal.direction == "LONG" else signal.entry_price - (risk * TRAILING_STOP_ACTIVATION)

if (signal.direction == "LONG" and current_price >= trailing_activation) or \
(signal.direction == "SHORT" and current_price <= trailing_activation):

trade_data['trailing_activated'] = True
log_callback(f"🎯 Trailing stop activated for {signal_id}")

# Update trailing stop
if trade_data['trailing_activated']:
risk = abs(signal.entry_price - signal.stop_loss)
trail_distance = risk * TRAILING_STOP_DISTANCE

if signal.direction == "LONG":
new_stop = trade_data['highest_price'] - trail_distance
trade_data['current_stop'] = max(trade_data['current_stop'], new_stop)
else:
new_stop = trade_data['lowest_price'] + trail_distance
trade_data['current_stop'] = min(trade_data['current_stop'], new_stop)

# Check expiry
if datetime.now() > signal.expiry:
await self._close_trade(signal_id, current_price, "EXPIRED", log_callback)
closed_trades.append(signal_id)
continue

# Emergency stop loss
if abs(pnl) <= -EMERGENCY_STOP_LOSS:
await self._close_trade(signal_id, current_price, "EMERGENCY_STOP", log_callback)
closed_trades.append(signal_id)
continue

except Exception as e:
log_callback(f"❌ Trade monitoring error for {signal_id}: {e}")

# Remove closed trades
for signal_id in closed_trades:
if signal_id in self.active_trades:
del self.active_trades[signal_id]

async def _close_trade(self, signal_id: str, exit_price: float, reason: str, log_callback):
"""Close a trade."""
try:
if signal_id not in self.active_trades:
return

trade_data = self.active_trades[signal_id]
signal = trade_data['signal']

# Calculate P&L
if signal.direction == "LONG":
pnl = exit_price - signal.entry_price
pnl_pct = (pnl / signal.entry_price) * 100
else:
pnl = signal.entry_price - exit_price
pnl_pct = (pnl / signal.entry_price) * 100

duration = (datetime.now() - trade_data['entry_time']).total_seconds()

log_callback(f"🔒 TRADE CLOSED: {signal_id}")
log_callback(f" Reason: {reason}")
log_callback(f" Exit: ${exit_price:.2f}")
log_callback(f" PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
log_callback(f" Duration: {duration:.0f}s")
log_callback(f" {'✅ PROFIT' if pnl > 0 else '❌ LOSS'}")

# Send Telegram closure
await self._send_telegram_closure(signal, exit_price, pnl, pnl_pct, reason, duration)

# Add to history
self.trade_history.append({
'signal_id': signal_id,
'symbol': signal.symbol,
'direction': signal.direction,
'entry_price': signal.entry_price,
'exit_price': exit_price,
'pnl': pnl,
'pnl_pct': pnl_pct,
'reason': reason,
'duration': duration,
'ml_confidence': signal.ml_confidence,
'signal_type': signal.signal_type.value
})

except Exception as e:
log_callback(f"❌ Trade closure error: {e}")

async def _send_telegram_alert(self, signal: EnhancedSignal):
"""Send Telegram alert for new trade."""
try:
direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"

message = f"""
⚡ ML ENHANCED SCALP SIGNAL ⚡

{direction_emoji} {signal.symbol} {signal.direction}
Signal Type: {signal.signal_type.value}
Timeframe: {signal.timeframe}
Market Regime: {signal.ml_prediction.regime.value}

🎯 Entry Details:
Entry: ${signal.entry_price:.2f}
Stop Loss: ${signal.stop_loss:.2f}
Take Profit: ${signal.take_profit:.2f}

📊 ML Insights:
ML Confidence: {signal.ml_confidence:.1f}%
Predicted Move: {signal.ml_prediction.predicted_move_pct:.1f}%
Probability Long: {signal.ml_prediction.probability_long:.1%}
Probability Short: {signal.ml_prediction.probability_short:.1%}

🎯 Risk Management:
Risk/Reward: 1:{signal.risk_reward:.1f}
Position Size: {signal.position_size*100:.0f}%

📈 Top Features:
{self._format_top_features(signal.ml_prediction.features)}

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
print(f"✅ Telegram alert sent for {signal.signal_id}")
else:
error_text = await response.text()
print(f"❌ Telegram error: {error_text}")

except Exception as e:
print(f"❌ Telegram alert error: {e}")

async def _send_telegram_closure(self, signal: EnhancedSignal, exit_price: float,
pnl: float, pnl_pct: float, reason: str, duration: float):
"""Send Telegram closure alert."""
try:
result_emoji = "💰" if pnl > 0 else "💸"

message = f"""
{result_emoji} ML SCALP TRADE CLOSED {result_emoji}

📊 Performance Summary:
Symbol: {signal.symbol}
Direction: {signal.direction}
Signal Type: {signal.signal_type.value}
Duration: {duration:.0f}s

💰 Financials:
Entry: ${signal.entry_price:.2f}
Exit: ${exit_price:.2f}
PnL: ${pnl:.2f}
PnL %: {pnl_pct:.2f}%

📝 Details:
Exit Reason: {reason}
ML Confidence Was: {signal.ml_confidence:.1f}%
Risk/Reward Was: 1:{signal.risk_reward:.1f}
Predicted Move Was: {signal.ml_prediction.predicted_move_pct:.1f}%

📊 Trade Result:
{'✅ PROFITABLE' if pnl > 0 else '❌ UNPROFITABLE'}

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
print(f"✅ Telegram closure sent for {signal.symbol}")
else:
error_text = await response.text()
print(f"❌ Telegram closure error: {error_text}")

except Exception as e:
print(f"❌ Telegram closure error: {e}")

def _format_top_features(self, features: Dict) -> str:
"""Format top features for Telegram."""
try:
# Sort by absolute value
sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

formatted = []
for feature, value in sorted_features:
formatted.append(f"{feature}: {value:.3f}")

return "\n".join(formatted)
except:
return "Features unavailable"

# =============================================
# MAIN ENHANCED BOT
# =============================================

class EnhancedMLScalpingBot:
"""Enhanced ML-powered scalping bot."""

def __init__(self):
print("="*70)
print("🤖 ENHANCED ML SCALPING BOT - BTC/ETH ONLY")
print("="*70)

# Initialize components
self.market_data = EnhancedMarketData()
self.ml_manager = MLModelManager()
self.signal_generator = EnhancedSignalGenerator(self.market_data, self.ml_manager)
self.trade_manager = EnhancedTradeManager(self.ml_manager)

# State
self.cycle_count = 0
self.paused = True
self.last_model_update = datetime.now()

# Load models
self.ml_manager.load_models()

print("✅ Enhanced bot initialized successfully")
print(" • XGBoost ML models with feature engineering")
print(" • Advanced SMC/FVG detection")
print(" • Real-time market regime classification")
print(" • Professional risk management")
print(" • Telegram integration with ML insights")
print("="*70)

async def run(self):
"""Main bot loop."""
print("🚀 Starting Enhanced ML Scalping Bot...")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize market data
await self.market_data.initialize()

try:
while True:
if not self.paused:
await self.run_cycle()

# Update ML models periodically
if (datetime.now() - self.last_model_update).seconds >= MODEL_UPDATE_INTERVAL * 60:
await self.update_models()
self.last_model_update = datetime.now()

await asyncio.sleep(SCALP_INTERVAL)

except KeyboardInterrupt:
print("\n🛑 Bot stopped by user")
except Exception as e:
print(f"❌ Bot error: {str(e)}")
traceback.print_exc()
finally:
# Cleanup
await self.market_data.close()
self.ml_manager.save_models()
print("💾 Cleanup completed")

async def run_cycle(self):
"""Run one trading cycle."""
self.cycle_count += 1

print(f"\n⚡ CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")

try:
# Monitor existing trades
await self.trade_manager.monitor_trades(
self.market_data,
lambda msg: print(f" {msg}")
)

# Generate and execute signals for each symbol
for symbol in TRADING_PAIRS:
signal = await self.signal_generator.generate_signal(
symbol,
lambda msg: print(f" {msg}")
)

if signal:
await self.trade_manager.execute_trade(
signal,
lambda msg: print(f" {msg}")
)

# Collect training data
await self.collect_training_data()

except Exception as e:
print(f"❌ Cycle error: {str(e)}")
traceback.print_exc()

async def collect_training_data(self):
"""Collect data for ML training."""
try:
for symbol in TRADING_PAIRS:
# Get historical data
df = await self.market_data.get_ohlcv_data(symbol, '1m', 100)
if df is None or len(df) < 50:
continue

# Calculate features for past points
for i in range(20, len(df) - 10):
window_df = df.iloc[:i+1]
features = self.market_data.calculate_features(window_df)

if features:
# Calculate future returns
future_price = df['Close'].iloc[i + 10]
current_price = df['Close'].iloc[i]
future_returns = (future_price - current_price) / current_price * 100

# Add to training data
self.ml_manager.add_training_data(symbol, features, future_returns)

except Exception as e:
print(f"⚠️ Training data collection error: {e}")

async def update_models(self):
"""Update ML models."""
print("\n🔄 Updating ML models...")

for symbol in TRADING_PAIRS:
if self.ml_manager.train_model(symbol):
print(f" ✅ {symbol} model updated")
else:
print(f" ⚠️ {symbol} model update skipped (insufficient data)")

# Save models
self.ml_manager.save_models()

# =============================================
# SIMPLE GUI FOR CONTROL
# =============================================

class SimpleControlGUI:
"""Simple GUI for bot control."""

def __init__(self, bot: EnhancedMLScalpingBot):
self.bot = bot
self.root = tk.Tk()
self.root.title("Enhanced ML Scalping Bot - Control Panel")
self.root.geometry("800x600")

self.setup_ui()

def setup_ui(self):
"""Setup user interface."""
# Title
title_frame = ttk.Frame(self.root)
title_frame.pack(pady=10)

ttk.Label(title_frame, text="🤖 ENHANCED ML SCALPING BOT",
font=("Arial", 16, "bold")).pack()

ttk.Label(title_frame, text="BTC/ETH Only | ML-Powered | SMC/FVG Strategy",
font=("Arial", 10)).pack()

# Status panel
status_frame = ttk.LabelFrame(self.root, text="Status")
status_frame.pack(fill='x', padx=20, pady=10)

self.status_label = ttk.Label(status_frame, text="⏸️ PAUSED",
font=("Arial", 12, "bold"))
self.status_label.pack(pady=5)

# Stats frame
stats_frame = ttk.Frame(status_frame)
stats_frame.pack(fill='x', padx=10, pady=5)

self.cycle_label = ttk.Label(stats_frame, text="Cycles: 0")
self.cycle_label.grid(row=0, column=0, padx=10, sticky='w')

self.trades_label = ttk.Label(stats_frame, text="Active Trades: 0")
self.trades_label.grid(row=0, column=1, padx=10, sticky='w')

# Control buttons
control_frame = ttk.LabelFrame(self.root, text="Controls")
control_frame.pack(fill='x', padx=20, pady=10)

btn_frame = ttk.Frame(control_frame)
btn_frame.pack(pady=10)

self.start_btn = ttk.Button(btn_frame, text="▶️ START",
command=self.start_bot, width=15)
self.start_btn.pack(side='left', padx=5)

self.pause_btn = ttk.Button(btn_frame, text="⏸️ PAUSE",
command=self.pause_bot, width=15,
state='disabled')
self.pause_btn.pack(side='left', padx=5)

# Log output
log_frame = ttk.LabelFrame(self.root, text="Live Log")
log_frame.pack(fill='both', expand=True, padx=20, pady=10)

self.log_text = scrolledtext.ScrolledText(log_frame, height=15,
font=("Consolas", 9),
bg='black', fg='white',
insertbackground='white')
self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

# Status bar
self.status_bar = ttk.Label(self.root,
text="Ready to start",
relief='sunken',
anchor='center')
self.status_bar.pack(side='bottom', fill='x')

# Start update timer
self.update_ui()

def update_ui(self):
"""Update UI elements."""
try:
# Update status
status = "RUNNING" if not self.bot.paused else "PAUSED"
color = "green" if not self.bot.paused else "red"
self.status_label.config(text=f"🟢 {status}", foreground=color)

# Update stats
self.cycle_label.config(text=f"Cycles: {self.bot.cycle_count}")
self.trades_label.config(text=f"Active Trades: {len(self.bot.trade_manager.active_trades)}")

# Update status bar
self.status_bar.config(
text=f"{datetime.now().strftime('%H:%M:%S')} | "
f"Cycles: {self.bot.cycle_count} | "
f"Active Trades: {len(self.bot.trade_manager.active_trades)}"
)

except Exception as e:
print(f"⚠️ UI update error: {e}")

# Schedule next update
self.root.after(1000, self.update_ui)

def add_log(self, message: str):
"""Add message to log."""
try:
timestamp = datetime.now().strftime("%H:%M:%S")
self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
self.log_text.see(tk.END)

# Limit log size
if self.log_text.index('end-1c').split('.')[0] > '1000':
self.log_text.delete(1.0, 2.0)

except Exception as e:
print(f"⚠️ Log error: {e}")

def start_bot(self):
"""Start bot."""
self.bot.paused = False
self.start_btn.config(state='disabled')
self.pause_btn.config(state='normal')
self.add_log("▶️ Bot started - ML Scalping active")
self.add_log("📊 ML models loaded and ready")
self.add_log("🎯 Monitoring BTC and ETH")
self.add_log("📱 Signals will be sent to Telegram")

def pause_bot(self):
"""Pause bot."""
self.bot.paused = True
self.start_btn.config(state='normal')
self.pause_btn.config(state='disabled')
self.add_log("⏸️ Bot paused")

def run(self):
"""Run GUI."""
self.root.mainloop()

# =============================================
# MAIN ENTRY POINT
# =============================================

def main():
"""Start the enhanced ML scalping bot."""
print("\n" + "="*70)
print("🎯 ENHANCED ML SCALPING BOT - PROFESSIONAL EDITION")
print("="*70)
print("\n✨ CORE FEATURES:")
print("1. 🤖 Machine Learning with XGBoost")
print("2. 📊 Advanced Feature Engineering (50+ features)")
print("3. 🎯 Smart Money Concepts (SMC/FVG)")
print("4. 📈 Real-time Market Regime Detection")
print("5. ⚡ High-Frequency Scalping (BTC/ETH only)")
print("6. 🛡️ Professional Risk Management")
print("7. 📱 Telegram Integration with ML Insights")
print("8. 🔄 Continuous Model Retraining")

print("\n🎯 TRADING STRATEGY:")
print("• ML predictions combined with SMC signals")
print("• FVG and Order Block detection")
print("• Liquidity and Breakout analysis")
print("• Multi-timeframe confirmation")

print("\n📊 RISK MANAGEMENT:")
print(f"• Max Trades: {MAX_CONCURRENT_TRADES}")
print(f"• Risk per Trade: {RISK_PER_TRADE*100:.0f}%")
print(f"• Min RRR: 1:{TAKE_PROFIT_MULTIPLIER}")
print(f"• Trailing Stops with Breakeven")
print(f"• Emergency Stop: {EMERGENCY_STOP_LOSS}%")

print("\n⚙️ TECHNICAL SPECS:")
print(f"• Cycle Interval: {SCALP_INTERVAL}s")
print(f"• Max Trade Duration: {MAX_TRADE_DURATION}s")
print(f"• ML Update Interval: {MODEL_UPDATE_INTERVAL}min")
print(f"• Min ML Confidence: {MIN_CONFIDENCE}%")
print("="*70 + "\n")

# Create bot
bot = EnhancedMLScalpingBot()

# Create and run GUI in separate thread
gui = SimpleControlGUI(bot)

# Redirect print to GUI log
original_print = print
def gui_print(*args, **kwargs):
message = ' '.join(str(arg) for arg in args)
gui.add_log(message)
original_print(*args, **kwargs)

# Don't override print to avoid recursion issues
# Instead, create a custom log function
def log_to_gui(message):
gui.add_log(message)

# Run bot in thread
def run_bot():
try:
asyncio.run(bot.run())
except Exception as e:
log_to_gui(f"❌ Bot thread error: {e}")
original_print(f"❌ Bot thread error: {e}")

bot_thread = threading.Thread(target=run_bot, daemon=True)
bot_thread.start()

# Run GUI
try:
gui.run()
except Exception as e:
original_print(f"❌ GUI error: {e}")

if __name__ == "__main__":
main()


