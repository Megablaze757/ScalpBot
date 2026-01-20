# =============================================
# ULTIMATE SCALPING BOT - FVG + SMART MONEY + MICROSTRUCTURE
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
# CONFIGURATION - ULTIMATE SCALPER
# =============================================
TELEGRAM_BOT_TOKEN = "8285366409:AAH9kdy1D-xULBmGakAPFYUME19fmVCDJ9E"
TELEGRAM_CHAT_ID = "-1003525746518"

# Trading Parameters
SCAN_INTERVAL = 5  # 5 seconds for ultra-fast scanning
MAX_CONCURRENT_TRADES = 3
MAX_TRADE_DURATION = 600  # 10 minutes (ultra scalping)
MIN_CONFIDENCE = 75  # Higher threshold for ultimate scalper
RISK_PER_TRADE = 0.01  # 1% risk per trade for scalping

# Analysis Mode: "SMART_MONEY" or "FVG_MICROSTRUCTURE"
ANALYSIS_MODE = "SMART_MONEY"  # Ultimate scalping mode
USE_LIVE_DATA = True
TRAIN_WITH_HISTORICAL = True
HISTORICAL_DAYS = 90  # 90 days for scalping patterns
TRAIN_INTERVAL = 1800  # Retrain every 30 minutes

# Ultimate Scalping Pairs (Higher volatility)
TRADING_PAIRS = [
    "BTC-USD",
    "ETH-USD"
]

# Yahoo Finance symbols (correct format)
YF_SYMBOLS = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD"
}

# Pip configurations for crypto
PIP_CONFIG = {
    "BTC-USD": 0.01,
    "ETH-USD": 0.01,
}

# Order Block/Imbalance sizes
ORDER_BLOCK_SIZE = 0.002  # 0.2% for order blocks
FVG_SIZE = 0.0015  # 0.15% for Fair Value Gaps

# Model paths
MODEL_SAVE_PATH = "ultimate_models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# =============================================
# ULTIMATE TELEGRAM MANAGER
# =============================================

class UltimateTelegramManager:
    """Telegram manager for ultimate scalping signals."""
    
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.test_connection()
    
    def test_connection(self):
        """Test Telegram connection."""
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
                    self.send_message_sync("🚀 ULTIMATE SCALPING BOT ACTIVATED\n"
                                          f"Mode: {ANALYSIS_MODE}\n"
                                          f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    return True
                    
        except Exception as e:
            print(f"❌ Telegram test error: {e}")
        return False
    
    def send_message_sync(self, message: str) -> bool:
        """Send Telegram message."""
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
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
        return False

# =============================================
# ADVANCED DATA FETCHER
# =============================================

class AdvancedDataFetcher:
    """Advanced data fetching with multiple timeframes."""
    
    def __init__(self):
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 15  # Cache for 15 seconds
        
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get live price."""
        try:
            current_time = time.time()
            if symbol in self.cache and symbol in self.cache_time:
                if current_time - self.cache_time[symbol] < self.cache_duration:
                    return self.cache[symbol]
            
            yf_symbol = YF_SYMBOLS.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if len(data) > 0:
                current_price = data['Close'].iloc[-1]
                self.cache[symbol] = current_price
                self.cache_time[symbol] = current_time
                return current_price
                
        except Exception as e:
            print(f"⚠️ Error fetching data for {symbol}: {e}")
        return None
    
    def get_multiple_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes."""
        timeframes = {}
        try:
            yf_symbol = YF_SYMBOLS.get(symbol, symbol)
            ticker = yf.Ticker(yf_symbol)
            
            # 1-minute for microstructure
            tf1 = ticker.history(period='1d', interval='1m')
            if len(tf1) > 0:
                timeframes['1m'] = tf1
            
            # 5-minute for order blocks
            tf5 = ticker.history(period='5d', interval='5m')
            if len(tf5) > 0:
                timeframes['5m'] = tf5
            
            # 15-minute for FVGs
            tf15 = ticker.history(period='15d', interval='15m')
            if len(tf15) > 0:
                timeframes['15m'] = tf15
            
            # 1-hour for higher timeframe structure
            tf1h = ticker.history(period='30d', interval='1h')
            if len(tf1h) > 0:
                timeframes['1h'] = tf1h
                
        except Exception as e:
            print(f"⚠️ Error fetching multiple timeframes for {symbol}: {e}")
            
        return timeframes

# =============================================
# SMART MONEY CONCEPTS ANALYZER
# =============================================

class SmartMoneyAnalyzer:
    """Analyzes Smart Money Concepts: Order Blocks, Liquidity, FVGs."""
    
    def __init__(self, data_fetcher: AdvancedDataFetcher):
        self.data_fetcher = data_fetcher
        self.order_blocks = {}
        self.fair_value_gaps = {}
        self.liquidity_levels = {}
        
    def analyze_order_blocks(self, symbol: str, timeframe: str = '5m'):
        """Identify order blocks (market structure shifts)."""
        try:
            timeframes = self.data_fetcher.get_multiple_timeframes(symbol)
            if timeframe not in timeframes:
                return []
            
            data = timeframes[timeframe]
            if len(data) < 20:
                return []
            
            highs = data['High'].values
            lows = data['Low'].values
            closes = data['Close'].values
            
            order_blocks = []
            
            # Look for significant candles followed by reversal
            for i in range(2, len(data) - 2):
                # Bullish order block (sell-side liquidity taken)
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and  # Lower low
                    closes[i+1] > closes[i] and closes[i+2] > closes[i+1]):  # Then reversal
                    
                    block = {
                        'type': 'BULLISH_OB',
                        'price_level': lows[i],
                        'timestamp': data.index[i],
                        'size': abs(highs[i] - lows[i]),
                        'timeframe': timeframe
                    }
                    order_blocks.append(block)
                
                # Bearish order block (buy-side liquidity taken)
                elif (highs[i] > highs[i-1] and highs[i] > highs[i-2] and  # Higher high
                      closes[i+1] < closes[i] and closes[i+2] < closes[i+1]):  # Then reversal
                    
                    block = {
                        'type': 'BEARISH_OB',
                        'price_level': highs[i],
                        'timestamp': data.index[i],
                        'size': abs(highs[i] - lows[i]),
                        'timeframe': timeframe
                    }
                    order_blocks.append(block)
            
            self.order_blocks[symbol] = order_blocks
            return order_blocks
            
        except Exception as e:
            print(f"⚠️ Error analyzing order blocks for {symbol}: {e}")
            return []
    
    def analyze_fair_value_gaps(self, symbol: str, timeframe: str = '15m'):
        """Identify Fair Value Gaps (imbalances)."""
        try:
            timeframes = self.data_fetcher.get_multiple_timeframes(symbol)
            if timeframe not in timeframes:
                return []
            
            data = timeframes[timeframe]
            if len(data) < 10:
                return []
            
            highs = data['High'].values
            lows = data['Low'].values
            fvgs = []
            
            # Look for gaps between candles
            for i in range(1, len(data) - 1):
                # Bullish FVG (gap up)
                if lows[i] > highs[i-1]:
                    fvg = {
                        'type': 'BULLISH_FVG',
                        'top': highs[i-1],
                        'bottom': lows[i],
                        'midpoint': (highs[i-1] + lows[i]) / 2,
                        'timestamp': data.index[i],
                        'size': abs(lows[i] - highs[i-1]),
                        'timeframe': timeframe
                    }
                    fvgs.append(fvg)
                
                # Bearish FVG (gap down)
                elif highs[i] < lows[i-1]:
                    fvg = {
                        'type': 'BEARISH_FVG',
                        'top': lows[i-1],
                        'bottom': highs[i],
                        'midpoint': (lows[i-1] + highs[i]) / 2,
                        'size': abs(lows[i-1] - highs[i]),
                        'timestamp': data.index[i],
                        'timeframe': timeframe
                    }
                    fvgs.append(fvg)
            
            self.fair_value_gaps[symbol] = fvgs
            return fvgs
            
        except Exception as e:
            print(f"⚠️ Error analyzing FVGs for {symbol}: {e}")
            return []
    
    def analyze_liquidity_levels(self, symbol: str):
        """Identify liquidity pools (previous highs/lows)."""
        try:
            timeframes = self.data_fetcher.get_multiple_timeframes(symbol)
            if '1h' not in timeframes:
                return []
            
            data = timeframes['1h']
            if len(data) < 50:
                return []
            
            # Recent swing highs and lows
            highs = data['High'].values[-50:]
            lows = data['Low'].values[-50:]
            
            # Find significant swing points
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    swing_highs.append(highs[i])
                
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    swing_lows.append(lows[i])
            
            liquidity_levels = {
                'swing_highs': sorted(swing_highs)[-5:],  # Top 5 recent swing highs
                'swing_lows': sorted(swing_lows)[:5],     # Bottom 5 recent swing lows
                'current_high': max(highs[-20:]),
                'current_low': min(lows[-20:])
            }
            
            self.liquidity_levels[symbol] = liquidity_levels
            return liquidity_levels
            
        except Exception as e:
            print(f"⚠️ Error analyzing liquidity for {symbol}: {e}")
            return []
    
    def analyze_break_of_structure(self, symbol: str, timeframe: str = '5m'):
        """Analyze Break of Structure (BOS) and Change of Character (CHOCH)."""
        try:
            timeframes = self.data_fetcher.get_multiple_timeframes(symbol)
            if timeframe not in timeframes:
                return None
            
            data = timeframes[timeframe]
            if len(data) < 30:
                return None
            
            closes = data['Close'].values
            highs = data['High'].values
            lows = data['Low'].values
            
            # Analyze market structure
            structure = {
                'higher_highs': 0,
                'higher_lows': 0,
                'lower_highs': 0,
                'lower_lows': 0,
                'bos_detected': False,
                'choch_detected': False,
                'trend': 'NEUTRAL'
            }
            
            # Check last 10 swings
            for i in range(20, len(closes) - 5):
                # Check for higher highs
                if highs[i] > highs[i-5] and highs[i] > highs[i-10]:
                    structure['higher_highs'] += 1
                
                # Check for higher lows
                if lows[i] > lows[i-5] and lows[i] > lows[i-10]:
                    structure['higher_lows'] += 1
                
                # Check for lower highs
                if highs[i] < highs[i-5] and highs[i] < highs[i-10]:
                    structure['lower_highs'] += 1
                
                # Check for lower lows
                if lows[i] < lows[i-5] and lows[i] < lows[i-10]:
                    structure['lower_lows'] += 1
            
            # Determine trend
            if structure['higher_highs'] > 3 and structure['higher_lows'] > 3:
                structure['trend'] = 'BULLISH'
            elif structure['lower_highs'] > 3 and structure['lower_lows'] > 3:
                structure['trend'] = 'BEARISH'
            
            # Check for BOS (break of previous swing point)
            if len(closes) > 25:
                recent_high = max(highs[-25:-5])
                recent_low = min(lows[-25:-5])
                
                if highs[-1] > recent_high:
                    structure['bos_detected'] = True
                    structure['bos_type'] = 'BULLISH'
                elif lows[-1] < recent_low:
                    structure['bos_detected'] = True
                    structure['bos_type'] = 'BEARISH'
            
            # Check for CHOCH (change from HH/HL to LH/LL or vice versa)
            if (structure['higher_highs'] > 2 and structure['higher_lows'] > 2 and
                structure['lower_highs'] > 0 and structure['lower_lows'] > 0):
                structure['choch_detected'] = True
                structure['choch_type'] = 'BEARISH_REVERSAL'
            elif (structure['lower_highs'] > 2 and structure['lower_lows'] > 2 and
                  structure['higher_highs'] > 0 and structure['higher_lows'] > 0):
                structure['choch_detected'] = True
                structure['choch_type'] = 'BULLISH_REVERSAL'
            
            return structure
            
        except Exception as e:
            print(f"⚠️ Error analyzing structure for {symbol}: {e}")
            return None
    
    def get_smart_money_signal(self, symbol: str) -> Dict:
        """Generate comprehensive smart money signal."""
        try:
            # Analyze all components
            order_blocks = self.analyze_order_blocks(symbol, '5m')
            fvgs = self.analyze_fair_value_gaps(symbol, '15m')
            liquidity = self.analyze_liquidity_levels(symbol)
            structure = self.analyze_break_of_structure(symbol, '5m')
            
            # Get current price
            current_price = self.data_fetcher.get_live_price(symbol)
            if current_price is None:
                return None
            
            # Find nearest order blocks
            nearest_bullish_ob = None
            nearest_bearish_ob = None
            ob_distance = float('inf')
            
            for ob in order_blocks:
                distance = abs(current_price - ob['price_level'])
                if distance < ob_distance:
                    if ob['type'] == 'BULLISH_OB':
                        nearest_bullish_ob = ob
                    else:
                        nearest_bearish_ob = ob
                    ob_distance = distance
            
            # Find nearest FVG
            nearest_fvg = None
            fvg_distance = float('inf')
            
            for fvg in fvgs:
                if fvg['type'] == 'BULLISH_FVG':
                    if current_price <= fvg['bottom']:  # Price below bullish FVG
                        distance = abs(current_price - fvg['bottom'])
                        if distance < fvg_distance:
                            nearest_fvg = fvg
                            fvg_distance = distance
                else:  # BEARISH_FVG
                    if current_price >= fvg['top']:  # Price above bearish FVG
                        distance = abs(current_price - fvg['top'])
                        if distance < fvg_distance:
                            nearest_fvg = fvg
                            fvg_distance = distance
            
            # Determine signal
            signal = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'order_blocks': order_blocks,
                'fvgs': fvgs,
                'liquidity': liquidity,
                'structure': structure,
                'nearest_bullish_ob': nearest_bullish_ob,
                'nearest_bearish_ob': nearest_bearish_ob,
                'nearest_fvg': nearest_fvg,
                'signal_strength': 0,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'reason': []
            }
            
            # Calculate signal strength
            strength_factors = []
            
            # Order Block proximity (30% weight)
            if nearest_bullish_ob and ob_distance < current_price * 0.005:  # Within 0.5%
                strength_factors.append(30)
                signal['reason'].append(f"Near Bullish OB ({ob_distance/current_price*100:.2f}%)")
                if current_price <= nearest_bullish_ob['price_level'] * 1.001:  # At or below OB
                    strength_factors.append(20)
                    signal['direction'] = 'LONG'
            
            elif nearest_bearish_ob and ob_distance < current_price * 0.005:
                strength_factors.append(30)
                signal['reason'].append(f"Near Bearish OB ({ob_distance/current_price*100:.2f}%)")
                if current_price >= nearest_bearish_ob['price_level'] * 0.999:  # At or above OB
                    strength_factors.append(20)
                    signal['direction'] = 'SHORT'
            
            # FVG proximity (25% weight)
            if nearest_fvg and fvg_distance < current_price * 0.003:  # Within 0.3%
                strength_factors.append(25)
                if nearest_fvg['type'] == 'BULLISH_FVG':
                    signal['reason'].append(f"Near Bullish FVG ({fvg_distance/current_price*100:.2f}%)")
                    if signal['direction'] == 'NEUTRAL':
                        signal['direction'] = 'LONG'
                else:
                    signal['reason'].append(f"Near Bearish FVG ({fvg_distance/current_price*100:.2f}%)")
                    if signal['direction'] == 'NEUTRAL':
                        signal['direction'] = 'SHORT'
            
            # Market Structure (25% weight)
            if structure:
                if structure['trend'] == 'BULLISH' and structure['bos_detected']:
                    strength_factors.append(25)
                    signal['reason'].append("Bullish BOS detected")
                    if signal['direction'] == 'NEUTRAL':
                        signal['direction'] = 'LONG'
                elif structure['trend'] == 'BEARISH' and structure['bos_detected']:
                    strength_factors.append(25)
                    signal['reason'].append("Bearish BOS detected")
                    if signal['direction'] == 'NEUTRAL':
                        signal['direction'] = 'SHORT'
                
                if structure['choch_detected']:
                    strength_factors.append(15)
                    signal['reason'].append(f"{structure['choch_type']} detected")
            
            # Liquidity proximity (20% weight)
            if liquidity:
                # Check distance to recent swing high/low (liquidity)
                for swing_high in liquidity['swing_highs']:
                    if abs(current_price - swing_high) < current_price * 0.002:  # Within 0.2%
                        strength_factors.append(20)
                        signal['reason'].append(f"Near swing high liquidity")
                        signal['direction'] = 'SHORT'  # Expect rejection at swing high
                        break
                
                for swing_low in liquidity['swing_lows']:
                    if abs(current_price - swing_low) < current_price * 0.002:
                        strength_factors.append(20)
                        signal['reason'].append(f"Near swing low liquidity")
                        signal['direction'] = 'LONG'  # Expect bounce at swing low
                        break
            
            # Calculate total confidence
            if strength_factors:
                signal['signal_strength'] = sum(strength_factors)
                signal['confidence'] = min(100, signal['signal_strength'])
            
            # Only return if confidence meets threshold
            if signal['confidence'] >= MIN_CONFIDENCE and signal['direction'] != 'NEUTRAL':
                signal['reason'] = " | ".join(signal['reason'])
                return signal
            
        except Exception as e:
            print(f"⚠️ Error generating smart money signal for {symbol}: {e}")
        
        return None

# =============================================
# MARKET MICROSTRUCTURE ANALYZER
# =============================================

class MicrostructureAnalyzer:
    """Analyzes market microstructure for ultra-scalping."""
    
    def __init__(self, data_fetcher: AdvancedDataFetcher):
        self.data_fetcher = data_fetcher
        self.order_flow = {}
        self.imbalances = {}
        
    def analyze_order_flow_imbalance(self, symbol: str):
        """Analyze order flow imbalance using 1-minute data."""
        try:
            timeframes = self.data_fetcher.get_multiple_timeframes(symbol)
            if '1m' not in timeframes:
                return None
            
            data = timeframes['1m']
            if len(data) < 20:
                return None
            
            closes = data['Close'].values
            volumes = data['Volume'].values if 'Volume' in data.columns else np.ones(len(closes))
            
            # Calculate price changes and volume-weighted moves
            price_changes = np.diff(closes)
            volumes = volumes[1:]  # Align with price changes
            
            # Calculate buying vs selling pressure
            buy_volume = np.sum(volumes[price_changes > 0])
            sell_volume = np.sum(volumes[price_changes < 0])
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                buy_pressure = buy_volume / total_volume
                sell_pressure = sell_volume / total_volume
                imbalance = buy_pressure - sell_pressure
            else:
                buy_pressure = sell_pressure = 0.5
                imbalance = 0
            
            # Analyze recent acceleration
            recent_changes = price_changes[-5:]
            acceleration = np.mean(np.diff(recent_changes)) if len(recent_changes) > 1 else 0
            
            # Detect absorption (large volume without price movement)
            absorption_signals = []
            for i in range(1, len(price_changes)):
                if abs(price_changes[i]) < 0.0001 and volumes[i] > np.mean(volumes) * 1.5:
                    absorption_signals.append({
                        'index': i,
                        'volume': volumes[i],
                        'type': 'ABSORPTION'
                    })
            
            microstructure = {
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'imbalance': imbalance,
                'acceleration': acceleration,
                'absorption_count': len(absorption_signals),
                'volume_trend': 'UP' if volumes[-1] > np.mean(volumes) else 'DOWN',
                'current_momentum': 'BULLISH' if price_changes[-1] > 0 else 'BEARISH'
            }
            
            self.order_flow[symbol] = microstructure
            return microstructure
            
        except Exception as e:
            print(f"⚠️ Error analyzing microstructure for {symbol}: {e}")
            return None
    
    def detect_supply_demand_zones(self, symbol: str):
        """Detect immediate supply and demand zones."""
        try:
            timeframes = self.data_fetcher.get_multiple_timeframes(symbol)
            if '1m' not in timeframes:
                return {'supply_zones': [], 'demand_zones': []}
            
            data = timeframes['1m']
            if len(data) < 30:
                return {'supply_zones': [], 'demand_zones': []}
            
            highs = data['High'].values
            lows = data['Low'].values
            closes = data['Close'].values
            
            supply_zones = []
            demand_zones = []
            
            # Look for rejection candles (wicks)
            for i in range(1, len(data) - 1):
                candle_range = highs[i] - lows[i]
                if candle_range == 0:
                    continue
                
                # Upper wick > 60% of candle (supply zone)
                upper_wick = highs[i] - max(closes[i], closes[i-1])
                if upper_wick > candle_range * 0.6:
                    zone = {
                        'price': highs[i],
                        'strength': upper_wick / candle_range,
                        'timestamp': data.index[i]
                    }
                    supply_zones.append(zone)
                
                # Lower wick > 60% of candle (demand zone)
                lower_wick = min(closes[i], closes[i-1]) - lows[i]
                if lower_wick > candle_range * 0.6:
                    zone = {
                        'price': lows[i],
                        'strength': lower_wick / candle_range,
                        'timestamp': data.index[i]
                    }
                    demand_zones.append(zone)
            
            # Keep only recent zones (last 15 minutes)
            recent_supply = [z for z in supply_zones if 
                           (datetime.now() - z['timestamp']).total_seconds() < 900]
            recent_demand = [z for z in demand_zones if 
                           (datetime.now() - z['timestamp']).total_seconds() < 900]
            
            zones = {
                'supply_zones': sorted(recent_supply, key=lambda x: x['strength'], reverse=True)[:3],
                'demand_zones': sorted(recent_demand, key=lambda x: x['strength'], reverse=True)[:3]
            }
            
            self.imbalances[symbol] = zones
            return zones
            
        except Exception as e:
            print(f"⚠️ Error detecting supply/demand zones for {symbol}: {e}")
            return {'supply_zones': [], 'demand_zones': []}
    
    def get_microstructure_signal(self, symbol: str) -> Dict:
        """Generate microstructure-based scalping signal."""
        try:
            # Get current price
            current_price = self.data_fetcher.get_live_price(symbol)
            if current_price is None:
                return None
            
            # Analyze microstructure
            order_flow = self.analyze_order_flow_imbalance(symbol)
            zones = self.detect_supply_demand_zones(symbol)
            
            if order_flow is None:
                return None
            
            signal = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'order_flow': order_flow,
                'zones': zones,
                'signal_strength': 0,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'reason': []
            }
            
            # Calculate signal based on microstructure
            strength_factors = []
            
            # Order Flow Imbalance (40% weight)
            imbalance = order_flow['imbalance']
            if imbalance > 0.3:  # Strong buying pressure
                strength_factors.append(40)
                signal['reason'].append(f"Strong buy pressure ({imbalance:.2%})")
                signal['direction'] = 'LONG'
            elif imbalance < -0.3:  # Strong selling pressure
                strength_factors.append(40)
                signal['reason'].append(f"Strong sell pressure ({abs(imbalance):.2%})")
                signal['direction'] = 'SHORT'
            
            # Supply/Demand Zone proximity (35% weight)
            # Check distance to nearest supply zone
            nearest_supply = None
            supply_distance = float('inf')
            for zone in zones['supply_zones']:
                distance = abs(current_price - zone['price'])
                if distance < supply_distance:
                    nearest_supply = zone
                    supply_distance = distance
            
            # Check distance to nearest demand zone
            nearest_demand = None
            demand_distance = float('inf')
            for zone in zones['demand_zones']:
                distance = abs(current_price - zone['price'])
                if distance < demand_distance:
                    nearest_demand = zone
                    demand_distance = distance
            
            # Determine which zone is closer
            if nearest_supply and supply_distance < demand_distance:
                if supply_distance < current_price * 0.001:  # Within 0.1%
                    strength_factors.append(35)
                    signal['reason'].append(f"At supply zone ({supply_distance/current_price*100:.3f}%)")
                    if signal['direction'] == 'NEUTRAL':
                        signal['direction'] = 'SHORT'  # Expect rejection at supply
            
            elif nearest_demand and demand_distance < supply_distance:
                if demand_distance < current_price * 0.001:  # Within 0.1%
                    strength_factors.append(35)
                    signal['reason'].append(f"At demand zone ({demand_distance/current_price*100:.3f}%)")
                    if signal['direction'] == 'NEUTRAL':
                        signal['direction'] = 'LONG'  # Expect bounce at demand
            
            # Momentum and Acceleration (25% weight)
            acceleration = order_flow['acceleration']
            momentum = order_flow['current_momentum']
            
            if abs(acceleration) > 0.0001:  # Significant acceleration
                strength_factors.append(25)
                if acceleration > 0 and momentum == 'BULLISH':
                    signal['reason'].append(f"Bullish acceleration ({acceleration:.6f})")
                    if signal['direction'] == 'NEUTRAL':
                        signal['direction'] = 'LONG'
                elif acceleration < 0 and momentum == 'BEARISH':
                    signal['reason'].append(f"Bearish acceleration ({acceleration:.6f})")
                    if signal['direction'] == 'NEUTRAL':
                        signal['direction'] = 'SHORT'
            
            # Absorption detection (bonus 15%)
            if order_flow['absorption_count'] > 0:
                strength_factors.append(15)
                signal['reason'].append(f"{order_flow['absorption_count']} absorption candles")
                # Absorption often precedes reversal
            
            # Calculate total confidence
            if strength_factors:
                signal['signal_strength'] = sum(strength_factors)
                signal['confidence'] = min(100, signal['signal_strength'])
            
            # Only return if confidence meets threshold
            if signal['confidence'] >= MIN_CONFIDENCE and signal['direction'] != 'NEUTRAL':
                signal['reason'] = " | ".join(signal['reason'])
                return signal
            
        except Exception as e:
            print(f"⚠️ Error generating microstructure signal for {symbol}: {e}")
        
        return None

# =============================================
# ULTIMATE SIGNAL GENERATOR
# =============================================

class UltimateSignalGenerator:
    """Generates ultimate scalping signals using multiple strategies."""
    
    def __init__(self, smart_money_analyzer: SmartMoneyAnalyzer, 
                 microstructure_analyzer: MicrostructureAnalyzer,
                 telegram: UltimateTelegramManager):
        self.smart_money = smart_money_analyzer
        self.microstructure = microstructure_analyzer
        self.telegram = telegram
        self.data_fetcher = smart_money_analyzer.data_fetcher
        self.signal_history = []
        
    async def generate_ultimate_signal(self, symbol: str, log_callback) -> Optional[Dict]:
        """Generate ultimate scalping signal."""
        try:
            current_price = self.data_fetcher.get_live_price(symbol)
            if current_price is None:
                return None
            
            signal = None
            
            if ANALYSIS_MODE == "SMART_MONEY":
                # Smart Money Concepts signal
                signal = self.smart_money.get_smart_money_signal(symbol)
                strategy = "SMART_MONEY"
            else:
                # FVG + Microstructure signal
                smart_signal = self.smart_money.get_smart_money_signal(symbol)
                micro_signal = self.microstructure.get_microstructure_signal(symbol)
                
                # Combine signals if both available
                if smart_signal and micro_signal:
                    # Take the stronger signal
                    if smart_signal['confidence'] >= micro_signal['confidence']:
                        signal = smart_signal
                        signal['reason'] += f" | Micro confluence: {micro_signal['reason']}"
                        signal['confidence'] = min(100, (smart_signal['confidence'] + micro_signal['confidence']) / 2)
                    else:
                        signal = micro_signal
                        signal['reason'] += f" | Smart Money confluence: {smart_signal['reason']}"
                        signal['confidence'] = min(100, (smart_signal['confidence'] + micro_signal['confidence']) / 2)
                    strategy = "FVG_MICROSTRUCTURE_COMBINED"
                elif smart_signal:
                    signal = smart_signal
                    strategy = "SMART_MONEY"
                elif micro_signal:
                    signal = micro_signal
                    strategy = "MICROSTRUCTURE"
                else:
                    return None
            
            if signal and signal['confidence'] >= MIN_CONFIDENCE:
                # Calculate precise entry, stop loss, and take profit
                entry, stop_loss, take_profit = self.calculate_scalping_levels(
                    symbol, current_price, signal['direction'], strategy
                )
                
                # Calculate risk/reward
                risk = abs(entry - stop_loss)
                reward = abs(take_profit - entry)
                risk_reward = reward / risk if risk > 0 else 0
                
                # Minimum 1:2.5 RRR for ultimate scalping
                if risk_reward < 2.5:
                    log_callback(f"⏸️ {symbol}: RRR 1:{risk_reward:.1f} < 1:2.5")
                    return None
                
                # Calculate position size (1% risk for scalping)
                pip_size = PIP_CONFIG.get(symbol, 0.01)
                risk_pips = risk / pip_size
                position_size = (RISK_PER_TRADE * 10000) / risk_pips
                
                # Create final signal
                ultimate_signal = {
                    'signal_id': f"ULT-{int(time.time())}-{random.randint(1000, 9999)}",
                    'symbol': symbol,
                    'strategy': strategy,
                    'direction': signal['direction'],
                    'entry_price': round(entry, 4),
                    'stop_loss': round(stop_loss, 4),
                    'take_profit': round(take_profit, 4),
                    'confidence': signal['confidence'],
                    'risk_reward': risk_reward,
                    'position_size': round(position_size, 6),  # Small positions for scalping
                    'risk_pips': round(risk_pips, 1),
                    'target_pips': round(reward / pip_size, 1),
                    'reason': signal['reason'],
                    'current_price': current_price,
                    'created_at': datetime.now(),
                    'expiry': datetime.now() + timedelta(minutes=10),  # 10 minute expiry for scalping
                    'status': 'PENDING'
                }
                
                # Log signal
                log_callback(f"⚡ ULTIMATE {symbol} {signal['direction']} SCALP")
                log_callback(f"   Strategy: {strategy}")
                log_callback(f"   Price: ${current_price:.2f}")
                log_callback(f"   Entry: ${entry:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
                log_callback(f"   Risk: {risk_pips:.1f}pips | Target: {reward/pip_size:.1f}pips")
                log_callback(f"   Confidence: {signal['confidence']:.1f}% | RRR: 1:{risk_reward:.1f}")
                log_callback(f"   Position: {position_size:.6f}")
                log_callback(f"   Reason: {signal['reason']}")
                
                self.signal_history.append(ultimate_signal)
                return ultimate_signal
                
        except Exception as e:
            log_callback(f"❌ Error generating ultimate signal for {symbol}: {str(e)}")
        
        return None
    
    def calculate_scalping_levels(self, symbol: str, current_price: float, 
                                 direction: str, strategy: str) -> Tuple[float, float, float]:
        """Calculate precise scalping levels."""
        pip_size = PIP_CONFIG.get(symbol, 0.01)
        
        if strategy == "SMART_MONEY" or "FVG" in strategy:
            # Tighter stops for smart money concepts
            if direction == "LONG":
                entry = current_price
                stop_loss = entry - (15 * pip_size)  # 15 pip stop for scalping
                take_profit = entry + (40 * pip_size)  # 40 pip target (1:2.67 RRR)
            else:  # SHORT
                entry = current_price
                stop_loss = entry + (15 * pip_size)
                take_profit = entry - (40 * pip_size)
        else:
            # Even tighter for microstructure
            if direction == "LONG":
                entry = current_price
                stop_loss = entry - (10 * pip_size)  # 10 pip stop
                take_profit = entry + (30 * pip_size)  # 30 pip target (1:3 RRR)
            else:  # SHORT
                entry = current_price
                stop_loss = entry + (10 * pip_size)
                take_profit = entry - (30 * pip_size)
        
        return entry, stop_loss, take_profit

# =============================================
# ULTIMATE TRADE MANAGER
# =============================================

class UltimateTradeManager:
    """Manages ultra-scalping trades."""
    
    def __init__(self, telegram: UltimateTelegramManager):
        self.telegram = telegram
        self.active_trades = {}
        self.trade_history = []
        self.scalping_stats = {
            'wins': 0,
            'losses': 0,
            'total_pips': 0,
            'total_pnl': 0,
            'win_streak': 0,
            'loss_streak': 0,
            'current_streak': 0
        }
    
    async def execute_trade(self, signal: Dict, log_callback) -> bool:
        """Execute an ultra-scalping trade."""
        if len(self.active_trades) >= MAX_CONCURRENT_TRADES:
            log_callback(f"⚠️ Max trades reached ({MAX_CONCURRENT_TRADES})")
            return False
        
        self.active_trades[signal['signal_id']] = {
            'signal': signal,
            'entry_time': datetime.now(),
            'status': 'ACTIVE',
            'breakeven_moved': False,
            'partial_tp_moved': False
        }
        
        log_callback(f"✅ ULTIMATE SCALP EXECUTED: {signal['signal_id']}")
        log_callback(f"   {signal['symbol']} {signal['direction']} | {signal['strategy']}")
        log_callback(f"   Confidence: {signal['confidence']:.1f}%")
        
        # Send Telegram alert
        await self.send_ultimate_telegram_alert(signal)
        
        return True
    
    async def send_ultimate_telegram_alert(self, signal: Dict):
        """Send ultimate Telegram alert."""
        try:
            direction_emoji = "🟢" if signal['direction'] == "LONG" else "🔴"
            strategy_emoji = "🎯" if "SMART_MONEY" in signal['strategy'] else "⚡"
            
            message = f"""
{strategy_emoji} <b>ULTIMATE SCALPING SIGNAL</b> {strategy_emoji}

{direction_emoji} <b>{signal['symbol']} {signal['direction']}</b>
Strategy: {signal['strategy']}
Confidence: <b>{signal['confidence']:.1f}%</b>

🎯 <b>Scalping Levels:</b>
Entry: ${signal['entry_price']:.2f}
Stop Loss: ${signal['stop_loss']:.2f}
Take Profit: ${signal['take_profit']:.2f}

📊 <b>Ultra-Scalping Stats:</b>
Risk: {signal['risk_pips']:.1f} pips
Target: {signal['target_pips']:.1f} pips
Risk/Reward: 1:{signal['risk_reward']:.1f}
Position: {signal['position_size']:.6f}

💡 <b>Signal Reason:</b>
{signal['reason']}

⏱️ <b>Timeframe:</b> 5-10 minute scalp
⏰ Entry Window: {datetime.now().strftime('%H:%M:%S')}
"""
            
            self.telegram.send_message_sync(message)
            
        except Exception as e:
            print(f"❌ Telegram alert error: {e}")
    
    async def monitor_trades(self, data_fetcher: AdvancedDataFetcher, log_callback):
        """Monitor active scalping trades with advanced management."""
        closed_trades = []
        
        for signal_id, trade_data in list(self.active_trades.items()):
            signal = trade_data['signal']
            
            try:
                # Get current price
                current_price = data_fetcher.get_live_price(signal['symbol'])
                if current_price is None:
                    continue
                
                elapsed_seconds = (datetime.now() - trade_data['entry_time']).total_seconds()
                
                # Check expiry (10 minutes max)
                if elapsed_seconds > MAX_TRADE_DURATION:
                    await self.close_trade(signal_id, current_price, "EXPIRED", log_callback)
                    closed_trades.append(signal_id)
                    continue
                
                if signal['direction'] == "LONG":
                    # Check stop loss
                    if current_price <= signal['stop_loss']:
                        await self.close_trade(signal_id, current_price, "STOP_LOSS", log_callback)
                        closed_trades.append(signal_id)
                        self.update_stats(False, signal['risk_pips'])
                    
                    # Check take profit
                    elif current_price >= signal['take_profit']:
                        await self.close_trade(signal_id, current_price, "TAKE_PROFIT", log_callback)
                        closed_trades.append(signal_id)
                        self.update_stats(True, signal['target_pips'])
                    
                    # Move to breakeven at 1.5x risk
                    elif (not trade_data['breakeven_moved'] and 
                          current_price >= signal['entry_price'] + (signal['risk_pips'] * 1.5 * PIP_CONFIG.get(signal['symbol'], 0.01))):
                        new_sl = signal['entry_price']
                        trade_data['signal']['stop_loss'] = new_sl
                        trade_data['breakeven_moved'] = True
                        log_callback(f"🔄 {signal_id}: Moved to breakeven at ${current_price:.2f}")
                    
                    # Partial profit at 0.5x target
                    elif (not trade_data['partial_tp_moved'] and 
                          current_price >= signal['entry_price'] + (signal['target_pips'] * 0.5 * PIP_CONFIG.get(signal['symbol'], 0.01))):
                        # Move stop loss to entry + 0.5x risk
                        new_sl = signal['entry_price'] + (signal['risk_pips'] * 0.5 * PIP_CONFIG.get(signal['symbol'], 0.01))
                        trade_data['signal']['stop_loss'] = new_sl
                        trade_data['partial_tp_moved'] = True
                        log_callback(f"🎯 {signal_id}: Partial TP secured at ${current_price:.2f}")
                
                else:  # SHORT
                    if current_price >= signal['stop_loss']:
                        await self.close_trade(signal_id, current_price, "STOP_LOSS", log_callback)
                        closed_trades.append(signal_id)
                        self.update_stats(False, signal['risk_pips'])
                    
                    elif current_price <= signal['take_profit']:
                        await self.close_trade(signal_id, current_price, "TAKE_PROFIT", log_callback)
                        closed_trades.append(signal_id)
                        self.update_stats(True, signal['target_pips'])
                    
                    elif (not trade_data['breakeven_moved'] and 
                          current_price <= signal['entry_price'] - (signal['risk_pips'] * 1.5 * PIP_CONFIG.get(signal['symbol'], 0.01))):
                        new_sl = signal['entry_price']
                        trade_data['signal']['stop_loss'] = new_sl
                        trade_data['breakeven_moved'] = True
                        log_callback(f"🔄 {signal_id}: Moved to breakeven at ${current_price:.2f}")
                    
                    elif (not trade_data['partial_tp_moved'] and 
                          current_price <= signal['entry_price'] - (signal['target_pips'] * 0.5 * PIP_CONFIG.get(signal['symbol'], 0.01))):
                        new_sl = signal['entry_price'] - (signal['risk_pips'] * 0.5 * PIP_CONFIG.get(signal['symbol'], 0.01))
                        trade_data['signal']['stop_loss'] = new_sl
                        trade_data['partial_tp_moved'] = True
                        log_callback(f"🎯 {signal_id}: Partial TP secured at ${current_price:.2f}")
                        
            except Exception as e:
                log_callback(f"❌ Error monitoring trade {signal_id}: {str(e)}")
        
        # Remove closed trades
        for signal_id in closed_trades:
            if signal_id in self.active_trades:
                del self.active_trades[signal_id]
    
    def update_stats(self, win: bool, pips: float):
        """Update scalping statistics."""
        if win:
            self.scalping_stats['wins'] += 1
            self.scalping_stats['total_pips'] += pips
            self.scalping_stats['current_streak'] = max(0, self.scalping_stats['current_streak']) + 1
            self.scalping_stats['win_streak'] = max(self.scalping_stats['win_streak'], self.scalping_stats['current_streak'])
        else:
            self.scalping_stats['losses'] += 1
            self.scalping_stats['total_pips'] -= pips
            self.scalping_stats['current_streak'] = min(0, self.scalping_stats['current_streak']) - 1
            self.scalping_stats['loss_streak'] = min(self.scalping_stats['loss_streak'], self.scalping_stats['current_streak'])
    
    async def close_trade(self, signal_id: str, exit_price: float, reason: str, log_callback):
        """Close a trade."""
        if signal_id not in self.active_trades:
            return
        
        trade_data = self.active_trades[signal_id]
        signal = trade_data['signal']
        
        # Calculate PnL
        if signal['direction'] == "LONG":
            pnl = (exit_price - signal['entry_price']) * signal['position_size']
            pips = (exit_price - signal['entry_price']) / PIP_CONFIG.get(signal['symbol'], 0.01)
        else:
            pnl = (signal['entry_price'] - exit_price) * signal['position_size']
            pips = (signal['entry_price'] - exit_price) / PIP_CONFIG.get(signal['symbol'], 0.01)
        
        pnl_percent = (pnl / (signal['entry_price'] * signal['position_size'])) * 100
        win = pnl > 0
        
        log_callback(f"🔒 ULTIMATE SCALP CLOSED: {signal_id}")
        log_callback(f"   Result: {'💰 WIN' if win else '💸 LOSS'}")
        log_callback(f"   Reason: {reason}")
        log_callback(f"   Exit: ${exit_price:.2f}")
        log_callback(f"   PnL: ${pnl:.4f} ({pnl_percent:.2f}%)")
        log_callback(f"   Pips: {pips:+.1f}")
        log_callback(f"   Duration: {(datetime.now() - trade_data['entry_time']).total_seconds()/60:.1f}min")
        
        # Send Telegram closure
        await self.send_ultimate_telegram_closure(signal, exit_price, pnl, pnl_percent, pips, reason, win)
        
        # Add to history
        self.trade_history.append({
            'signal_id': signal_id,
            'symbol': signal['symbol'],
            'strategy': signal['strategy'],
            'direction': signal['direction'],
            'entry': signal['entry_price'],
            'exit': exit_price,
            'pnl': pnl,
            'pips': pips,
            'reason': reason,
            'duration': (datetime.now() - trade_data['entry_time']).total_seconds() / 60,
            'win': win
        })
    
    async def send_ultimate_telegram_closure(self, signal: Dict, exit_price: float, 
                                           pnl: float, pnl_percent: float, pips: float, 
                                           reason: str, win: bool):
        """Send ultimate closure alert."""
        try:
            result_emoji = "💰" if win else "💸"
            
            # Calculate win rate
            total_trades = self.scalping_stats['wins'] + self.scalping_stats['losses']
            win_rate = (self.scalping_stats['wins'] / total_trades * 100) if total_trades > 0 else 0
            
            message = f"""
{result_emoji} <b>ULTIMATE SCALP CLOSED</b> {result_emoji}

📊 <b>Performance Summary:</b>
Symbol: {signal['symbol']}
Strategy: {signal['strategy']}
Direction: {signal['direction']}
Result: {'WIN' if win else 'LOSS'}

💵 <b>Trade Results:</b>
Entry: ${signal['entry_price']:.2f}
Exit: ${exit_price:.2f}
PnL: ${pnl:.4f}
PnL %: {pnl_percent:.2f}%
Pips: {pips:+.1f}

📈 <b>Overall Stats:</b>
Win Rate: {win_rate:.1f}%
Total Pips: {self.scalping_stats['total_pips']:+.1f}
Current Streak: {abs(self.scalping_stats['current_streak'])} {'Wins' if self.scalping_stats['current_streak'] > 0 else 'Losses'}

📝 <b>Details:</b>
Reason: {reason}
Confidence Was: {signal['confidence']:.1f}%
Risk/Reward Was: 1:{signal['risk_reward']:.1f}

⏰ Closed at: {datetime.now().strftime('%H:%M:%S')}
"""
            
            self.telegram.send_message_sync(message)
            
        except Exception as e:
            print(f"❌ Telegram closure error: {e}")

# =============================================
# ULTIMATE GUI
# =============================================

class UltimateGUI:
    """Ultimate GUI for scalping bot."""
    
    def __init__(self, bot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title(f"⚡ ULTIMATE SCALPING BOT - {ANALYSIS_MODE}")
        self.root.geometry("1400x900")
        
        self.setup_styles()
        self.init_ui()
        self.start_update_timer()
        
        print("✅ Ultimate GUI initialized")
    
    def setup_styles(self):
        """Setup ultimate styling."""
        self.style = ttk.Style()
        
        bg_color = '#0a0a0a'
        fg_color = '#00ff00'
        panel_bg = '#1a1a1a'
        accent_color = '#00ffff'
        
        self.root.configure(bg=bg_color)
        
        self.style.configure('Ultimate.TLabel', 
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
        """Initialize ultimate UI."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top panel
        top_panel = ttk.Frame(main_container)
        top_panel.pack(fill='x', pady=(0, 10))
        
        # Left stats
        left_stats = ttk.LabelFrame(top_panel, text="⚡ ULTIMATE STATS", padding=10)
        left_stats.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right stats
        right_stats = ttk.LabelFrame(top_panel, text="📊 LIVE PERFORMANCE", padding=10)
        right_stats.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Ultimate stats
        self.metric_labels = {}
        ultimate_stats = [
            ("Total Pips:", "total_pips", "+0.0"),
            ("Win Rate:", "win_rate", "0.0%"),
            ("Win Streak:", "win_streak", "0"),
            ("Active Scalps:", "active_scalps", "0"),
            ("Avg Duration:", "avg_duration", "0.0m"),
            ("Today's PnL:", "today_pnl", "$0.00")
        ]
        
        for label, key, default in ultimate_stats:
            row = ttk.Frame(left_stats)
            row.pack(fill='x', pady=2)
            
            ttk.Label(row, text=label, style='Metric.TLabel', width=20).pack(side='left')
            self.metric_labels[key] = ttk.Label(row, text=default, style='Value.TLabel')
            self.metric_labels[key].pack(side='right')
        
        # Performance stats
        perf_stats = [
            ("BTC Price:", "btc_price", "$0.00"),
            ("ETH Price:", "eth_price", "$0.00"),
            ("Strategy:", "strategy", ANALYSIS_MODE),
            ("Confidence:", "confidence", "0.0%"),
            ("Last Signal:", "last_signal", "None"),
            ("Telegram:", "telegram_status", "✅")
        ]
        
        for label, key, default in perf_stats:
            row = ttk.Frame(right_stats)
            row.pack(fill='x', pady=2)
            
            ttk.Label(row, text=label, style='Metric.TLabel', width=20).pack(side='left')
            self.metric_labels[key] = ttk.Label(row, text=default, style='Value.TLabel')
            self.metric_labels[key].pack(side='right')
        
        # Middle panel
        middle_panel = ttk.Frame(main_container)
        middle_panel.pack(fill='both', expand=True, pady=(0, 10))
        
        # PnL Graph
        pnl_frame = ttk.LabelFrame(middle_panel, text="📈 SCALPING PnL", padding=5)
        pnl_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.fig_pnl = Figure(figsize=(8, 4), dpi=80, facecolor='#1a1a1a')
        self.ax_pnl = self.fig_pnl.add_subplot(111)
        self.canvas_pnl = FigureCanvasTkAgg(self.fig_pnl, pnl_frame)
        self.canvas_pnl.get_tk_widget().pack(fill='both', expand=True)
        
        # Win/Loss Graph
        winloss_frame = ttk.LabelFrame(middle_panel, text="📊 WIN/LOSS HEATMAP", padding=5)
        winloss_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.fig_winloss = Figure(figsize=(8, 4), dpi=80, facecolor='#1a1a1a')
        self.ax_winloss = self.fig_winloss.add_subplot(111)
        self.canvas_winloss = FigureCanvasTkAgg(self.fig_winloss, winloss_frame)
        self.canvas_winloss.get_tk_widget().pack(fill='both', expand=True)
        
        # Bottom panel
        bottom_panel = ttk.Frame(main_container)
        bottom_panel.pack(fill='both', expand=True)
        
        # Controls
        control_frame = ttk.LabelFrame(bottom_panel, text="⚙️ ULTIMATE CONTROLS", padding=10)
        control_frame.pack(side='left', fill='both', padx=(0, 5))
        
        # Strategy toggle
        strategy_frame = ttk.Frame(control_frame)
        strategy_frame.pack(pady=5)
        
        ttk.Label(strategy_frame, text="Strategy:", style='Metric.TLabel').pack(side='left', padx=(0, 5))
        
        self.strategy_var = tk.StringVar(value=ANALYSIS_MODE)
        self.strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, 
                                         values=["SMART_MONEY", "FVG_MICROSTRUCTURE"], 
                                         state='readonly', width=20)
        self.strategy_combo.pack(side='left')
        self.strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)
        
        # Control buttons
        self.start_btn = ttk.Button(control_frame, text="⚡ START ULTIMATE", 
                                   command=self.start_bot, width=20)
        self.start_btn.pack(pady=5)
        
        self.pause_btn = ttk.Button(control_frame, text="⏸️ PAUSE SCALPING", 
                                   command=self.pause_bot, width=20,
                                   state='disabled')
        self.pause_btn.pack(pady=5)
        
        ttk.Button(control_frame, text="🔄 SWITCH STRATEGY", 
                  command=self.toggle_strategy, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="📡 TEST TELEGRAM", 
                  command=self.test_telegram, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="📊 REFRESH STATS", 
                  command=self.refresh_stats, width=20).pack(pady=5)
        
        ttk.Button(control_frame, text="🗑️ CLEAR LOGS", 
                  command=self.clear_logs, width=20).pack(pady=5)
        
        # Pairs info
        pairs_frame = ttk.Frame(control_frame)
        pairs_frame.pack(pady=10)
        
        ttk.Label(pairs_frame, text="🎯 SCALPING PAIRS:", style='Metric.TLabel').pack()
        for symbol in TRADING_PAIRS:
            ttk.Label(pairs_frame, text=f"  {symbol}", 
                     style='Value.TLabel').pack(anchor='w')
        
        # Risk info
        risk_frame = ttk.Frame(control_frame)
        risk_frame.pack(pady=5)
        
        ttk.Label(risk_frame, text=f"⚖️ RISK PER TRADE: {RISK_PER_TRADE*100}%", 
                 style='Value.TLabel').pack()
        ttk.Label(risk_frame, text=f"🎯 MIN RRR: 1:2.5", style='Value.TLabel').pack()
        ttk.Label(risk_frame, text=f"⏱️ MAX DURATION: {MAX_TRADE_DURATION//60}min", 
                 style='Value.TLabel').pack()
        
        # Logs
        log_frame = ttk.LabelFrame(bottom_panel, text="📝 ULTIMATE SCALPING LOG", padding=5)
        log_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12,
                                                 font=('Consolas', 9),
                                                 bg='#0a0a0a', fg='#00ff00',
                                                 insertbackground='white')
        self.log_text.pack(fill='both', expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, 
                                   text=f"⚡ ULTIMATE SCALPING BOT READY | {ANALYSIS_MODE} | Press START",
                                   relief='sunken',
                                   anchor='center',
                                   font=('Arial', 10))
        self.status_bar.pack(side='bottom', fill='x')
        
        # Initialize graphs
        self.init_graphs()
    
    def init_graphs(self):
        """Initialize graphs."""
        self.ax_pnl.clear()
        self.ax_pnl.set_facecolor('#1a1a1a')
        self.ax_pnl.set_title('Scalping PnL Progression', color='white', pad=20)
        self.ax_pnl.set_xlabel('Trade Sequence', color='white')
        self.ax_pnl.set_ylabel('Cumulative PnL ($)', color='white')
        self.ax_pnl.tick_params(colors='white')
        self.ax_pnl.grid(True, alpha=0.3, color='gray')
        
        self.ax_winloss.clear()
        self.ax_winloss.set_facecolor('#1a1a1a')
        self.ax_winloss.set_title('Win/Loss Heatmap', color='white', pad=20)
        self.ax_winloss.tick_params(colors='white')
        
        self.canvas_pnl.draw()
        self.canvas_winloss.draw()
    
    def on_strategy_change(self, event=None):
        """Handle strategy change."""
        global ANALYSIS_MODE
        ANALYSIS_MODE = self.strategy_var.get()
        self.root.title(f"⚡ ULTIMATE SCALPING BOT - {ANALYSIS_MODE}")
        self.add_log(f"🔄 Strategy changed to: {ANALYSIS_MODE}")
    
    def toggle_strategy(self):
        """Toggle between strategies."""
        current = self.strategy_var.get()
        new = "FVG_MICROSTRUCTURE" if current == "SMART_MONEY" else "SMART_MONEY"
        self.strategy_var.set(new)
        self.on_strategy_change()
    
    def test_telegram(self):
        """Test Telegram."""
        self.add_log("📡 Testing Telegram...")
        if self.bot.telegram.test_connection():
            self.add_log("✅ Telegram is working!")
            self.metric_labels['telegram_status'].config(text="✅")
        else:
            self.add_log("❌ Telegram test failed")
            self.metric_labels['telegram_status'].config(text="❌")
    
    def refresh_stats(self):
        """Refresh statistics."""
        self.add_log("📊 Refreshing statistics...")
        self.update_ui()
    
    def start_update_timer(self):
        """Start update timer."""
        try:
            self.update_ui()
        except Exception as e:
            print(f"⚠️ UI update error: {e}")
        finally:
            self.root.after(3000, self.start_update_timer)  # Update every 3 seconds
    
    def update_ui(self):
        """Update UI."""
        try:
            status = "RUNNING" if not self.bot.paused else "PAUSED"
            strategy = self.strategy_var.get()
            
            self.status_bar.config(
                text=f"⚡ {status} | {strategy} | "
                     f"Cycles: {self.bot.cycle_count} | "
                     f"Signals: {self.bot.signals_today} | "
                     f"Active: {len(self.bot.trade_manager.active_trades)} | "
                     f"{datetime.now().strftime('%H:%M:%S')}"
            )
            
            # Update prices
            if USE_LIVE_DATA:
                btc_price = self.bot.data_fetcher.get_live_price("BTC-USD")
                eth_price = self.bot.data_fetcher.get_live_price("ETH-USD")
                
                if btc_price:
                    self.metric_labels['btc_price'].config(text=f"${btc_price:.2f}")
                if eth_price:
                    self.metric_labels['eth_price'].config(text=f"${eth_price:.2f}")
            
            # Update stats from trade manager
            stats = self.bot.trade_manager.scalping_stats
            total_trades = stats['wins'] + stats['losses']
            
            if total_trades > 0:
                win_rate = (stats['wins'] / total_trades) * 100
                avg_duration = np.mean([t['duration'] for t in self.bot.trade_manager.trade_history]) if self.bot.trade_manager.trade_history else 0
                total_pnl = sum(t['pnl'] for t in self.bot.trade_manager.trade_history)
            else:
                win_rate = 0
                avg_duration = 0
                total_pnl = 0
            
            self.metric_labels['total_pips'].config(text=f"{stats['total_pips']:+.1f}")
            self.metric_labels['win_rate'].config(text=f"{win_rate:.1f}%")
            self.metric_labels['win_streak'].config(text=str(abs(stats['current_streak']) if stats['current_streak'] > 0 else 0))
            self.metric_labels['active_scalps'].config(text=str(len(self.bot.trade_manager.active_trades)))
            self.metric_labels['avg_duration'].config(text=f"{avg_duration:.1f}m")
            self.metric_labels['today_pnl'].config(text=f"${total_pnl:.4f}")
            
            # Update last signal
            if self.bot.signal_generator.signal_history:
                last_signal = self.bot.signal_generator.signal_history[-1]
                self.metric_labels['last_signal'].config(
                    text=f"{last_signal['symbol']} {last_signal['direction']}"
                )
                self.metric_labels['confidence'].config(
                    text=f"{last_signal['confidence']:.1f}%"
                )
            
            # Update graphs
            pnl_data = [t['pnl'] for t in self.bot.trade_manager.trade_history]
            self.update_pnl_graph(pnl_data)
            
            wins = stats['wins']
            losses = stats['losses']
            self.update_winloss_graph(wins, losses)
            
        except Exception as e:
            print(f"⚠️ UI update error: {e}")
    
    def update_pnl_graph(self, pnl_data: List[float]):
        """Update PnL graph."""
        self.ax_pnl.clear()
        self.ax_pnl.set_facecolor('#1a1a1a')
        
        if pnl_data:
            times = np.arange(len(pnl_data))
            cumulative = np.cumsum(pnl_data)
            
            # Color based on profit/loss
            colors = ['#00ff00' if p > 0 else '#ff4444' for p in pnl_data]
            self.ax_pnl.bar(times, pnl_data, color=colors, alpha=0.7)
            self.ax_pnl.plot(times, cumulative, 'w-', linewidth=2, alpha=0.8)
            
            # Mark breakeven line
            self.ax_pnl.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        
        self.ax_pnl.set_title('Scalping PnL Progression', color='white', pad=20)
        self.ax_pnl.set_xlabel('Trade #', color='white')
        self.ax_pnl.set_ylabel('PnL ($)', color='white')
        self.ax_pnl.tick_params(colors='white')
        self.ax_pnl.grid(True, alpha=0.2, color='gray')
        
        self.canvas_pnl.draw()
    
    def update_winloss_graph(self, wins: int, losses: int):
        """Update win/loss graph."""
        self.ax_winloss.clear()
        self.ax_winloss.set_facecolor('#1a1a1a')
        
        if wins + losses > 0:
            # Create heatmap-like visualization
            data = np.array([[wins, 0], [0, losses]])
            im = self.ax_winloss.imshow(data, cmap='RdYlGn', aspect='auto')
            
            # Add text labels
            for i in range(2):
                for j in range(2):
                    if data[i, j] > 0:
                        color = 'white' if data[i, j] > (wins + losses) / 4 else 'black'
                        self.ax_winloss.text(j, i, f'{data[i, j]}', 
                                           ha='center', va='center', 
                                           color=color, fontsize=20, fontweight='bold')
            
            self.ax_winloss.set_xticks([0, 1])
            self.ax_winloss.set_xticklabels(['Wins', 'Losses'], color='white')
            self.ax_winloss.set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=self.ax_winloss)
        
        self.ax_winloss.set_title('Win/Loss Heatmap', color='white', pad=20)
        self.canvas_winloss.draw()
    
    def add_log(self, message: str):
        """Add message to log."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
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
        self.add_log("⚡ ULTIMATE SCALPING BOT ACTIVATED")
        self.add_log(f"🎯 Strategy: {self.strategy_var.get()}")
        self.add_log(f"⚖️ Risk: {RISK_PER_TRADE*100}% per trade")
        self.add_log(f"🎯 Target: 1:2.5+ RRR")
        self.add_log(f"⏱️ Duration: 5-10 minute scalps")
        self.add_log("📱 Telegram alerts: ACTIVE")
    
    def pause_bot(self):
        """Pause bot."""
        self.bot.paused = True
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.add_log("⏸️ Scalping PAUSED")
        self.add_log("💾 Statistics saved")

# =============================================
# ULTIMATE SCALPING BOT
# =============================================

class UltimateScalpingBot:
    """Ultimate scalping bot with Smart Money and Microstructure."""
    
    def __init__(self):
        print("="*70)
        print("⚡ ULTIMATE SCALPING BOT - SMART MONEY + MICROSTRUCTURE")
        print("="*70)
        
        # Initialize Telegram
        print("📡 Initializing Telegram...")
        self.telegram = UltimateTelegramManager()
        
        # Initialize data fetcher
        print("📊 Initializing data fetcher...")
        self.data_fetcher = AdvancedDataFetcher()
        
        # Initialize analyzers
        print("🧠 Initializing Smart Money analyzer...")
        self.smart_money = SmartMoneyAnalyzer(self.data_fetcher)
        
        print("⚡ Initializing Microstructure analyzer...")
        self.microstructure = MicrostructureAnalyzer(self.data_fetcher)
        
        # Initialize signal generator
        print("🎯 Initializing Ultimate signal generator...")
        self.signal_generator = UltimateSignalGenerator(
            self.smart_money, self.microstructure, self.telegram
        )
        
        # Initialize trade manager
        print("💰 Initializing Ultimate trade manager...")
        self.trade_manager = UltimateTradeManager(self.telegram)
        
        # State
        self.cycle_count = 0
        self.signals_today = 0
        self.paused = True
        self.gui = None
        
        print("✅ Ultimate Scalping Bot initialized!")
        print(f"   • Strategy: {ANALYSIS_MODE}")
        print(f"   • Risk per trade: {RISK_PER_TRADE*100}%")
        print(f"   • Minimum RRR: 1:2.5")
        print(f"   • Max duration: {MAX_TRADE_DURATION//60} minutes")
        print(f"   • Pairs: {', '.join(TRADING_PAIRS)}")
        print(f"   • Telegram: @TheUltimateScalperBot")
        print("="*70)
    
    def set_gui(self, gui):
        """Set GUI."""
        self.gui = gui
        self.gui.add_log("⚡ ULTIMATE SCALPING BOT READY")
        self.gui.add_log(f"🎯 Strategy: {ANALYSIS_MODE}")
        self.gui.add_log(f"📱 Telegram: @TheUltimateScalperBot")
        self.gui.add_log(f"⚖️ Risk: {RISK_PER_TRADE*100}% per trade")
        self.gui.add_log(f"🎯 Target RRR: 1:2.5+")
        self.gui.add_log(f"⏱️ Scalp duration: 5-10 minutes")
        self.gui.add_log("Press START to begin ultimate scalping!")
    
    async def run_cycle(self):
        """Run one scalping cycle."""
        if self.paused:
            return
        
        self.cycle_count += 1
        
        if self.gui:
            self.gui.add_log(f"\n⚡ CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        try:
            # Monitor existing trades
            await self.trade_manager.monitor_trades(self.data_fetcher, 
                                                   self.gui.add_log if self.gui else print)
            
            # Generate signals for each symbol
            for symbol in TRADING_PAIRS:
                # Check max trades per symbol
                active_for_symbol = sum(
                    1 for trade in self.trade_manager.active_trades.values()
                    if trade['signal']['symbol'] == symbol
                )
                
                if active_for_symbol >= 1:
                    continue
                
                # Generate ultimate signal
                signal = await self.signal_generator.generate_ultimate_signal(
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
            self.gui.add_log(f"✅ Cycle {self.cycle_count} completed")
    
    async def run(self):
        """Main loop."""
        print("🚀 Starting Ultimate Scalping Bot...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.gui:
            self.gui.add_log("✅ All systems initialized")
            self.gui.add_log("✅ Smart Money analyzer ready")
            self.gui.add_log("✅ Microstructure analyzer ready")
            self.gui.add_log("✅ Telegram notifications active")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            if self.gui:
                self.gui.add_log("🛑 Ultimate Scalping stopped")
                self.gui.add_log("💾 Statistics saved")
            
            # Send shutdown message
            self.telegram.send_message_sync(
                f"🛑 Ultimate Scalping Bot Shutdown\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total Cycles: {self.cycle_count}\n"
                f"Total Signals: {self.signals_today}\n"
                f"Total Pips: {self.trade_manager.scalping_stats['total_pips']:+.1f}"
            )
                
        except Exception as e:
            error_msg = f"❌ Bot error: {str(e)}"
            print(error_msg)
            if self.gui:
                self.gui.add_log(error_msg)

# =============================================
# MAIN ENTRY POINT
# =============================================

def main():
    """Start the ultimate scalping bot."""
    bot = UltimateScalpingBot()
    
    # Create GUI
    gui = UltimateGUI(bot)
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
    print("⚡ ULTIMATE SCALPING BOT - SMART MONEY + MICROSTRUCTURE")
    print("="*70)
    print(f"\n🎯 STRATEGY: {ANALYSIS_MODE}")
    
    print("\n🚀 ULTIMATE FEATURES:")
    print("1. 🧠 SMART MONEY CONCEPTS: Order Blocks, FVGs, Liquidity")
    print("2. ⚡ MARKET MICROSTRUCTURE: Order flow, Supply/Demand zones")
    print("3. 🎯 BREAK OF STRUCTURE (BOS) detection")
    print("4. 💧 LIQUIDITY SWEEP identification")
    print("5. ⚖️ ULTRA-TIGHT RISK: 1% per trade, 1:2.5+ RRR")
    print("6. ⏱️ FAST SCALPING: 5-10 minute trades")
    print("7. 📱 INSTANT TELEGRAM: Real-time alerts")
    
    print("\n🧠 SMART MONEY COMPONENTS:")
    print("• Order Blocks (OB): Market structure shifts")
    print("• Fair Value Gaps (FVG): Price imbalances")
    print("• Liquidity Pools: Previous swing highs/lows")
    print("• Break of Structure (BOS): Trend continuation")
    print("• Change of Character (CHOCH): Trend reversal")
    
    print("\n⚡ MICROSTRUCTURE COMPONENTS:")
    print("• Order Flow Imbalance: Buying vs Selling pressure")
    print("• Supply/Demand Zones: Immediate rejection areas")
    print("• Absorption Detection: Large volume without movement")
    print("• Momentum Acceleration: Speed of price movement")
    
    print("\n🎯 TRADING PARAMETERS:")
    print(f"• Strategy: {ANALYSIS_MODE}")
    print(f"• Risk per trade: {RISK_PER_TRADE*100}%")
    print(f"• Minimum RRR: 1:2.5")
    print(f"• Max trade duration: {MAX_TRADE_DURATION//60} minutes")
    print(f"• Scan interval: {SCAN_INTERVAL} seconds")
    print(f"• Minimum confidence: {MIN_CONFIDENCE}%")
    
    print("\n📡 TELEGRAM:")
    print(f"• Bot: @TheUltimateScalperBot")
    print(f"• Chat ID: {TELEGRAM_CHAT_ID}")
    
    print("\n🔄 CONTROLS:")
    print("• START ULTIMATE: Begin scalping")
    print("• SWITCH STRATEGY: Toggle between Smart Money and FVG+Microstructure")
    print("• TEST TELEGRAM: Verify notifications")
    print("• REFRESH STATS: Update performance metrics")
    print("="*70 + "\n")
    
    main()
