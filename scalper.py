# =============================================
# ULTIMATE SMART MONEY SCALPING BOT
# =============================================
# Advanced features:
# 1. Smart Money Concepts (SMC)
# 2. Fair Value Gaps (FVGs)
# 3. Market Microstructure Analysis
# 4. Order Flow Analysis
# 5. Binance API + CoinGecko API
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
import hmac
import hashlib
import urllib.parse
import sqlite3
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass
from enum import Enum
import aiohttp
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import talib

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
TELEGRAM_BOT_TOKEN = "8285366409:AAH9kdy1D-xULBmGakAPFYUME19fmVCDJ9E"
TELEGRAM_CHAT_ID = "-1003525746518"

# API Configuration
USE_BINANCE_API = True
USE_COINGECKO_API = True

# Binance API (Add your keys here)
BINANCE_API_KEY = ""  # Your Binance API Key
BINANCE_API_SECRET = ""  # Your Binance API Secret
BINANCE_TESTNET = True  # Use testnet for safety

# CoinGecko API
COINGECKO_API_KEY = ""  # Optional: Get from coingecko.com

# Trading Parameters
SCAN_INTERVAL = 5  # 5 seconds for ultra-fast scalping
MAX_CONCURRENT_TRADES = 3
MAX_TRADE_DURATION = 600  # 10 minutes max for scalping
MIN_CONFIDENCE = 75  # Higher threshold for smart money
RISK_PER_TRADE = 0.01  # 1% risk for scalping
MAX_DAILY_RISK = 0.05  # 5% max daily risk

# Strategy Mode
STRATEGY_MODE = "SMART_MONEY"  # Options: SMART_MONEY, TRADITIONAL, ML_ENHANCED

# Trading Pairs (Binance format)
TRADING_PAIRS = [
    "BTCUSDT",
    "ETHUSDT"
]

# Timeframes for multi-timeframe analysis
TIMEFRAMES = ["1m", "5m", "15m"]  # Multi-timeframe analysis

# Smart Money Parameters
FVG_THRESHOLD = 0.002  # 0.2% for FVG detection
ORDER_BLOCK_PERCENTILE = 0.95  # 95th percentile for order blocks
LIQUIDITY_POOL_DISTANCE = 0.01  # 1% distance for liquidity pools
MARKET_STRUCTURE_BREAK_CONFIRMATION = 2  # Need 2 candles confirmation

# Risk Management
TRAILING_STOP_PERCENT = 0.003  # 0.3% trailing stop
PARTIAL_TP_LEVELS = [0.005, 0.01, 0.015]  # 0.5%, 1%, 1.5% targets
POSITION_SCALING = True  # Scale into positions

# File Paths
MODEL_SAVE_PATH = "smart_money_models/"
DATA_CACHE_PATH = "market_data/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

# =============================================
# API CLIENTS
# =============================================

class BinanceClient:
    """Binance API client for real market data."""
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': api_key if api_key else ''
        })
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 2  # 2 seconds cache for scalping
        
    def _generate_signature(self, params):
        """Generate HMAC SHA256 signature."""
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 500):
        """Get candlestick data."""
        cache_key = f"{symbol}_{interval}_{limit}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache and cache_key in self.cache_time:
            if current_time - self.cache_time[cache_key] < self.cache_duration:
                return self.cache[cache_key]
        
        try:
            endpoint = "/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert to numeric
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                
                # Cache result
                self.cache[cache_key] = df
                self.cache_time[cache_key] = current_time
                
                return df
                
        except Exception as e:
            print(f"⚠️ Binance API error for {symbol}: {e}")
            
        return None
    
    def get_order_book(self, symbol: str, limit: int = 20):
        """Get order book data."""
        try:
            endpoint = "/fapi/v1/depth"
            params = {'symbol': symbol, 'limit': limit}
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=3)
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            print(f"⚠️ Binance order book error: {e}")
            
        return None
    
    def get_ticker(self, symbol: str):
        """Get ticker data."""
        try:
            endpoint = "/fapi/v1/ticker/24hr"
            params = {'symbol': symbol}
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=3)
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            print(f"⚠️ Binance ticker error: {e}")
            
        return None
    
    def get_funding_rate(self, symbol: str):
        """Get funding rate."""
        try:
            endpoint = "/fapi/v1/fundingRate"
            params = {'symbol': symbol}
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return data[0]
                    
        except Exception as e:
            print(f"⚠️ Binance funding rate error: {e}")
            
        return None

class CoinGeckoClient:
    """CoinGecko API client for additional market data."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'x-cg-demo-api-key': api_key})
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 30  # 30 seconds cache
        
    def get_coin_data(self, coin_id: str):
        """Get coin data from CoinGecko."""
        cache_key = f"coin_{coin_id}"
        current_time = time.time()
        
        if cache_key in self.cache and cache_key in self.cache_time:
            if current_time - self.cache_time[cache_key] < self.cache_duration:
                return self.cache[cache_key]
        
        try:
            endpoint = f"/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'false'
            }
            
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.cache[cache_key] = data
                self.cache_time[cache_key] = current_time
                return data
                
        except Exception as e:
            print(f"⚠️ CoinGecko API error: {e}")
            
        return None
    
    def get_market_chart(self, coin_id: str, days: int = 1):
        """Get market chart data."""
        cache_key = f"chart_{coin_id}_{days}"
        current_time = time.time()
        
        if cache_key in self.cache and cache_key in self.cache_time:
            if current_time - self.cache_time[cache_key] < self.cache_duration:
                return self.cache[cache_key]
        
        try:
            endpoint = f"/coins/{coin_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': days}
            
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.cache[cache_key] = data
                self.cache_time[cache_key] = current_time
                return data
                
        except Exception as e:
            print(f"⚠️ CoinGecko chart error: {e}")
            
        return None

# =============================================
# MARKET MICROSTRUCTURE ANALYZER
# =============================================

class MarketMicrostructureAnalyzer:
    """Analyzes market microstructure for smart money detection."""
    
    def __init__(self):
        self.order_flow_data = defaultdict(lambda: deque(maxlen=1000))
        self.volume_profile = defaultdict(lambda: defaultdict(float))
        self.time_sales = defaultdict(lambda: deque(maxlen=500))
        
    def analyze_order_imbalance(self, order_book: Dict) -> Dict:
        """Analyze order book imbalance."""
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {}
        
        bids = order_book['bids']
        asks = order_book['asks']
        
        # Calculate bid/ask volumes
        bid_volume = sum(float(bid[1]) for bid in bids)
        ask_volume = sum(float(ask[1]) for ask in asks)
        
        # Calculate imbalance
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Calculate weighted average prices
        weighted_bid = sum(float(bid[0]) * float(bid[1]) for bid in bids) / bid_volume if bid_volume > 0 else 0
        weighted_ask = sum(float(ask[0]) * float(ask[1]) for ask in asks) / ask_volume if ask_volume > 0 else 0
        
        # Spread analysis
        spread = weighted_ask - weighted_bid if weighted_ask > 0 and weighted_bid > 0 else 0
        spread_percent = (spread / weighted_bid * 100) if weighted_bid > 0 else 0
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            'weighted_bid': weighted_bid,
            'weighted_ask': weighted_ask,
            'spread': spread,
            'spread_percent': spread_percent,
            'total_volume': total_volume,
            'imbalance_strength': 'STRONG_BID' if imbalance > 0.3 else 
                                 'STRONG_ASK' if imbalance < -0.3 else 
                                 'NEUTRAL'
        }
    
    def detect_liquidity_pools(self, price_data: pd.DataFrame, current_price: float) -> List[Dict]:
        """Detect liquidity pools (high volume nodes)."""
        if price_data is None or len(price_data) < 20:
            return []
        
        pools = []
        
        # Calculate volume profile
        price_col = 'close' if 'close' in price_data.columns else price_data.columns[4]
        volume_col = 'volume' if 'volume' in price_data.columns else price_data.columns[5]
        
        # Group prices into buckets
        price_min = price_data[price_col].min()
        price_max = price_data[price_col].max()
        bucket_size = (price_max - price_min) / 20
        
        if bucket_size <= 0:
            return []
        
        # Create volume profile
        volume_profile = {}
        for idx, row in price_data.iterrows():
            price = row[price_col]
            volume = row[volume_col]
            bucket = round(price / bucket_size) * bucket_size
            
            if bucket in volume_profile:
                volume_profile[bucket] += volume
            else:
                volume_profile[bucket] = volume
        
        # Find high volume nodes (liquidity pools)
        if volume_profile:
            max_volume = max(volume_profile.values())
            threshold = max_volume * 0.5  # 50% of max volume
            
            for price_level, volume in volume_profile.items():
                if volume >= threshold:
                    distance_pct = abs(price_level - current_price) / current_price
                    
                    pools.append({
                        'price': price_level,
                        'volume': volume,
                        'distance_pct': distance_pct,
                        'type': 'SUPPLY' if price_level > current_price else 'DEMAND',
                        'strength': 'STRONG' if volume >= max_volume * 0.8 else 'MODERATE'
                    })
        
        return sorted(pools, key=lambda x: x['volume'], reverse=True)[:5]  # Top 5 pools
    
    def analyze_market_structure(self, price_data: pd.DataFrame) -> Dict:
        """Analyze market structure (higher highs, lower lows)."""
        if price_data is None or len(price_data) < 50:
            return {}
        
        price_col = 'close' if 'close' in price_data.columns else price_data.columns[4]
        high_col = 'high' if 'high' in price_data.columns else price_data.columns[2]
        low_col = 'low' if 'low' in price_data.columns else price_data.columns[3]
        
        prices = price_data[price_col].values
        highs = price_data[high_col].values
        lows = price_data[low_col].values
        
        # Detect swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(prices) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append({'index': i, 'price': highs[i]})
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append({'index': i, 'price': lows[i]})
        
        # Determine market structure
        structure = "RANGING"
        bias = "NEUTRAL"
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for Higher Highs (HH) and Higher Lows (HL) - Uptrend
            last_high = swing_highs[-1]['price']
            prev_high = swing_highs[-2]['price']
            last_low = swing_lows[-1]['price']
            prev_low = swing_lows[-2]['price']
            
            if last_high > prev_high and last_low > prev_low:
                structure = "UPTREND"
                bias = "BULLISH"
            # Check for Lower Highs (LH) and Lower Lows (LL) - Downtrend
            elif last_high < prev_high and last_low < prev_low:
                structure = "DOWNTREND"
                bias = "BEARISH"
            # Check for market structure break
            elif last_high > prev_high and last_low < prev_low:
                structure = "STRUCTURE_BREAK_BULLISH"
                bias = "BULLISH_BREAK"
            elif last_high < prev_high and last_low > prev_low:
                structure = "STRUCTURE_BREAK_BEARISH"
                bias = "BEARISH_BREAK"
        
        return {
            'structure': structure,
            'bias': bias,
            'swing_highs': swing_highs[-3:] if swing_highs else [],
            'swing_lows': swing_lows[-3:] if swing_lows else [],
            'current_high': highs[-1] if len(highs) > 0 else 0,
            'current_low': lows[-1] if len(lows) > 0 else 0,
            'support': swing_lows[-1]['price'] if swing_lows else lows[-1] if len(lows) > 0 else 0,
            'resistance': swing_highs[-1]['price'] if swing_highs else highs[-1] if len(highs) > 0 else 0
        }

# =============================================
# SMART MONEY ANALYZER
# =============================================

class SmartMoneyAnalyzer:
    """Implements Smart Money Concepts (SMC) strategies."""
    
    def __init__(self):
        self.fvg_cache = defaultdict(lambda: deque(maxlen=100))
        self.order_blocks = defaultdict(lambda: deque(maxlen=50))
        self.breakers = defaultdict(lambda: deque(maxlen=50))
        self.equilibrium = defaultdict(lambda: deque(maxlen=100))
        
    def detect_fair_value_gaps(self, candle_data: pd.DataFrame) -> List[Dict]:
        """Detect Fair Value Gaps (FVGs)."""
        if candle_data is None or len(candle_data) < 3:
            return []
        
        fvgs = []
        
        for i in range(1, len(candle_data) - 1):
            prev_candle = candle_data.iloc[i-1]
            current_candle = candle_data.iloc[i]
            next_candle = candle_data.iloc[i+1]
            
            # Bullish FVG (price gap up)
            if prev_candle['low'] > current_candle['high']:
                fvg_size = prev_candle['low'] - current_candle['high']
                fvg_percent = fvg_size / current_candle['high']
                
                if fvg_percent >= FVG_THRESHOLD:
                    fvgs.append({
                        'type': 'BULLISH_FVG',
                        'start': current_candle['high'],
                        'end': prev_candle['low'],
                        'size': fvg_size,
                        'size_percent': fvg_percent * 100,
                        'timestamp': current_candle.name if hasattr(current_candle, 'name') else i,
                        'age': len(candle_data) - i,
                        'status': 'UNFILLED' if next_candle['low'] > current_candle['high'] else 'FILLING' if next_candle['low'] <= prev_candle['low'] else 'FILLED'
                    })
            
            # Bearish FVG (price gap down)
            elif prev_candle['high'] < current_candle['low']:
                fvg_size = current_candle['low'] - prev_candle['high']
                fvg_percent = fvg_size / prev_candle['high']
                
                if fvg_percent >= FVG_THRESHOLD:
                    fvgs.append({
                        'type': 'BEARISH_FVG',
                        'start': prev_candle['high'],
                        'end': current_candle['low'],
                        'size': fvg_size,
                        'size_percent': fvg_percent * 100,
                        'timestamp': current_candle.name if hasattr(current_candle, 'name') else i,
                        'age': len(candle_data) - i,
                        'status': 'UNFILLED' if next_candle['high'] < current_candle['low'] else 'FILLING' if next_candle['high'] >= prev_candle['high'] else 'FILLED'
                    })
        
        return fvgs
    
    def identify_order_blocks(self, candle_data: pd.DataFrame) -> List[Dict]:
        """Identify Order Blocks (Smart Money accumulation/distribution)."""
        if candle_data is None or len(candle_data) < 10:
            return []
        
        order_blocks = []
        
        for i in range(3, len(candle_data) - 3):
            current_candle = candle_data.iloc[i]
            prev_candle = candle_data.iloc[i-1]
            
            # Bullish Order Block (after a drop, strong bullish candle)
            if (prev_candle['close'] < prev_candle['open'] and  # Previous bearish candle
                current_candle['close'] > current_candle['open'] and  # Current bullish candle
                current_candle['close'] > prev_candle['high'] and  # Engulfing pattern
                current_candle['volume'] > candle_data['volume'].rolling(5).mean().iloc[i]):  # Above average volume
                
                order_blocks.append({
                    'type': 'BULLISH_OB',
                    'high': current_candle['high'],
                    'low': current_candle['low'],
                    'mid': (current_candle['high'] + current_candle['low']) / 2,
                    'timestamp': current_candle.name if hasattr(current_candle, 'name') else i,
                    'volume': current_candle['volume'],
                    'strength': 'STRONG' if current_candle['volume'] > candle_data['volume'].rolling(10).mean().iloc[i] * 1.5 else 'MODERATE'
                })
            
            # Bearish Order Block (after a rally, strong bearish candle)
            elif (prev_candle['close'] > prev_candle['open'] and  # Previous bullish candle
                  current_candle['close'] < current_candle['open'] and  # Current bearish candle
                  current_candle['close'] < prev_candle['low'] and  # Engulfing pattern
                  current_candle['volume'] > candle_data['volume'].rolling(5).mean().iloc[i]):  # Above average volume
                
                order_blocks.append({
                    'type': 'BEARISH_OB',
                    'high': current_candle['high'],
                    'low': current_candle['low'],
                    'mid': (current_candle['high'] + current_candle['low']) / 2,
                    'timestamp': current_candle.name if hasattr(current_candle, 'name') else i,
                    'volume': current_candle['volume'],
                    'strength': 'STRONG' if current_candle['volume'] > candle_data['volume'].rolling(10).mean().iloc[i] * 1.5 else 'MODERATE'
                })
        
        return order_blocks[-5:]  # Last 5 order blocks
    
    def detect_breaker_blocks(self, candle_data: pd.DataFrame) -> List[Dict]:
        """Detect Breaker Blocks (failed order blocks)."""
        if candle_data is None or len(candle_data) < 10:
            return []
        
        breakers = []
        order_blocks = self.identify_order_blocks(candle_data)
        
        for ob in order_blocks:
            ob_index = ob['timestamp']
            if isinstance(ob_index, int) and ob_index < len(candle_data) - 5:
                # Check if price has broken through the order block
                subsequent_data = candle_data.iloc[ob_index+1:ob_index+6]
                
                if ob['type'] == 'BULLISH_OB':
                    # Check if price dropped below bullish OB low
                    if subsequent_data['low'].min() < ob['low']:
                        breakers.append({
                            'type': 'BEARISH_BREAKER',
                            'original_ob': ob,
                            'break_price': subsequent_data['low'].min(),
                            'break_time': subsequent_data.index[subsequent_data['low'] == subsequent_data['low'].min()][0] if len(subsequent_data) > 0 else ob_index + 1,
                            'strength': 'STRONG' if (ob['low'] - subsequent_data['low'].min()) / ob['low'] > 0.01 else 'MODERATE'
                        })
                
                elif ob['type'] == 'BEARISH_OB':
                    # Check if price rose above bearish OB high
                    if subsequent_data['high'].max() > ob['high']:
                        breakers.append({
                            'type': 'BULLISH_BREAKER',
                            'original_ob': ob,
                            'break_price': subsequent_data['high'].max(),
                            'break_time': subsequent_data.index[subsequent_data['high'] == subsequent_data['high'].max()][0] if len(subsequent_data) > 0 else ob_index + 1,
                            'strength': 'STRONG' if (subsequent_data['high'].max() - ob['high']) / ob['high'] > 0.01 else 'MODERATE'
                        })
        
        return breakers
    
    def calculate_equilibrium(self, candle_data: pd.DataFrame) -> Dict:
        """Calculate market equilibrium (fair price)."""
        if candle_data is None or len(candle_data) < 20:
            return {}
        
        # Use VWAP as equilibrium
        typical_price = (candle_data['high'] + candle_data['low'] + candle_data['close']) / 3
        vwap = (typical_price * candle_data['volume']).sum() / candle_data['volume'].sum()
        
        # Calculate standard deviation bands
        std_dev = typical_price.std()
        
        return {
            'vwap': vwap,
            'upper_band': vwap + std_dev,
            'lower_band': vwap - std_dev,
            'deviation_percent': (candle_data['close'].iloc[-1] - vwap) / vwap * 100,
            'zone': 'ABOVE_EQUILIBRIUM' if candle_data['close'].iloc[-1] > vwap else 
                   'BELOW_EQUILIBRIUM' if candle_data['close'].iloc[-1] < vwap else 
                   'AT_EQUILIBRIUM'
        }
    
    def generate_smc_signal(self, symbol: str, price_data: pd.DataFrame, 
                           order_book: Dict, current_price: float) -> Dict:
        """Generate Smart Money Concept trading signal."""
        
        # Get all SMC components
        fvgs = self.detect_fair_value_gaps(price_data)
        order_blocks = self.identify_order_blocks(price_data)
        breaker_blocks = self.detect_breaker_blocks(price_data)
        equilibrium = self.calculate_equilibrium(price_data)
        
        # Analyze price relative to SMC levels
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': current_price,
            'fvgs': fvgs,
            'order_blocks': order_blocks,
            'breaker_blocks': breaker_blocks,
            'equilibrium': equilibrium,
            'signal': 'NEUTRAL',
            'confidence': 0,
            'reason': [],
            'levels': {}
        }
        
        # Check for FVG reactions
        active_fvgs = [fvg for fvg in fvgs if fvg['status'] in ['UNFILLED', 'FILLING']]
        
        for fvg in active_fvgs:
            if fvg['type'] == 'BULLISH_FVG' and current_price <= fvg['end']:
                # Price entering bullish FVG
                signal['reason'].append(f"Entering BULLISH_FVG ({fvg['size_percent']:.2f}%)")
                signal['levels']['fvg_entry'] = fvg['start']
                signal['levels']['fvg_target'] = fvg['end']
                
                if len(signal['reason']) >= 2:
                    signal['signal'] = 'LONG'
                    signal['confidence'] = min(85, signal['confidence'] + 40)
            
            elif fvg['type'] == 'BEARISH_FVG' and current_price >= fvg['start']:
                # Price entering bearish FVG
                signal['reason'].append(f"Entering BEARISH_FVG ({fvg['size_percent']:.2f}%)")
                signal['levels']['fvg_entry'] = fvg['start']
                signal['levels']['fvg_target'] = fvg['end']
                
                if len(signal['reason']) >= 2:
                    signal['signal'] = 'SHORT'
                    signal['confidence'] = min(85, signal['confidence'] + 40)
        
        # Check for Order Block reactions
        recent_obs = [ob for ob in order_blocks if ob.get('age', 0) < 20]
        
        for ob in recent_obs:
            ob_zone_low = ob['low'] * 0.998  # 0.2% buffer
            ob_zone_high = ob['high'] * 1.002  # 0.2% buffer
            
            if ob_zone_low <= current_price <= ob_zone_high:
                if ob['type'] == 'BULLISH_OB':
                    signal['reason'].append(f"At BULLISH_OrderBlock (Strength: {ob['strength']})")
                    signal['levels']['ob_support'] = ob['low']
                    
                    if ob['strength'] == 'STRONG':
                        signal['confidence'] = min(90, signal['confidence'] + 45)
                    else:
                        signal['confidence'] = min(80, signal['confidence'] + 35)
                        
                    if signal['signal'] == 'NEUTRAL':
                        signal['signal'] = 'LONG'
                
                elif ob['type'] == 'BEARISH_OB':
                    signal['reason'].append(f"At BEARISH_OrderBlock (Strength: {ob['strength']})")
                    signal['levels']['ob_resistance'] = ob['high']
                    
                    if ob['strength'] == 'STRONG':
                        signal['confidence'] = min(90, signal['confidence'] + 45)
                    else:
                        signal['confidence'] = min(80, signal['confidence'] + 35)
                        
                    if signal['signal'] == 'NEUTRAL':
                        signal['signal'] = 'SHORT'
        
        # Check for Breaker Block reactions (failed breakouts)
        for breaker in breaker_blocks:
            breaker_zone = breaker['break_price'] * 0.995 if breaker['type'] == 'BEARISH_BREAKER' else breaker['break_price'] * 1.005
            
            if (breaker['type'] == 'BEARISH_BREAKER' and current_price >= breaker_zone) or \
               (breaker['type'] == 'BULLISH_BREAKER' and current_price <= breaker_zone):
                
                signal['reason'].append(f"Rejecting {breaker['type']} (Strength: {breaker['strength']})")
                signal['levels']['breaker_level'] = breaker['break_price']
                
                if breaker['strength'] == 'STRONG':
                    signal['confidence'] = min(95, signal['confidence'] + 50)
                else:
                    signal['confidence'] = min(85, signal['confidence'] + 40)
                
                # Breaker blocks often lead to reversals
                if breaker['type'] == 'BEARISH_BREAKER':
                    signal['signal'] = 'LONG'  # Failed bearish break -> bullish
                else:
                    signal['signal'] = 'SHORT'  # Failed bullish break -> bearish
        
        # Check equilibrium
        if 'deviation_percent' in equilibrium:
            deviation = equilibrium['deviation_percent']
            
            if deviation > 1.5:  # 1.5% above equilibrium
                signal['reason'].append(f"Overbought vs VWAP ({deviation:.2f}%)")
                signal['confidence'] = max(0, signal['confidence'] - 20)
                if signal['signal'] == 'LONG':
                    signal['signal'] = 'NEUTRAL'  # Cancel long if overbought
                    
            elif deviation < -1.5:  # 1.5% below equilibrium
                signal['reason'].append(f"Oversold vs VWAP ({deviation:.2f}%)")
                signal['confidence'] = max(0, signal['confidence'] - 20)
                if signal['signal'] == 'SHORT':
                    signal['signal'] = 'NEUTRAL'  # Cancel short if oversold
        
        # Final confidence adjustment
        if len(signal['reason']) >= 3 and signal['confidence'] >= 70:
            signal['confidence'] = min(95, signal['confidence'] + 10)
        
        return signal

# =============================================
# ENHANCED TECHNICAL ANALYZER WITH SMC
# =============================================

class EnhancedTechnicalAnalyzer:
    """Combines traditional TA with Smart Money Concepts."""
    
    def __init__(self, binance_client: BinanceClient = None):
        self.binance_client = binance_client
        self.smart_money = SmartMoneyAnalyzer()
        self.microstructure = MarketMicrostructureAnalyzer()
        
    async def analyze_symbol(self, symbol: str) -> Dict:
        """Comprehensive analysis of a symbol."""
        try:
            # Get multi-timeframe data
            tf_data = {}
            for tf in TIMEFRAMES:
                data = self.binance_client.get_klines(symbol, tf, 100)
                if data is not None:
                    tf_data[tf] = data
            
            if not tf_data:
                return {}
            
            # Get current price from 1m data
            current_price = tf_data['1m']['close'].iloc[-1] if '1m' in tf_data else 0
            
            # Get order book for microstructure
            order_book = self.binance_client.get_order_book(symbol, 20)
            
            # Analyze Smart Money Concepts on 5m timeframe
            smc_signal = {}
            if '5m' in tf_data:
                smc_signal = self.smart_money.generate_smc_signal(
                    symbol, tf_data['5m'], order_book, current_price
                )
            
            # Analyze microstructure
            microstructure = {}
            if order_book:
                microstructure = self.microstructure.analyze_order_imbalance(order_book)
                
                # Add liquidity pool analysis
                if '1m' in tf_data:
                    liquidity_pools = self.microstructure.detect_liquidity_pools(
                        tf_data['1m'], current_price
                    )
                    microstructure['liquidity_pools'] = liquidity_pools
            
            # Analyze market structure
            market_structure = {}
            if '15m' in tf_data:
                market_structure = self.microstructure.analyze_market_structure(tf_data['15m'])
            
            # Traditional technical indicators on 1m
            ta_indicators = {}
            if '1m' in tf_data:
                ta_indicators = self.calculate_ta_indicators(tf_data['1m'])
            
            # Combine all analyses
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'timeframes': list(tf_data.keys()),
                'smart_money': smc_signal,
                'microstructure': microstructure,
                'market_structure': market_structure,
                'technical_indicators': ta_indicators,
                'composite_signal': self.generate_composite_signal(
                    smc_signal, microstructure, market_structure, ta_indicators
                )
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return {}
    
    def calculate_ta_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate traditional technical indicators."""
        if data is None or len(data) < 20:
            return {}
        
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values
        
        # RSI
        rsi = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_value = macd[-1] if not np.isnan(macd[-1]) else 0
        macd_hist_value = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
        bb_width = ((bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]) * 100 if not np.isnan(bb_middle[-1]) else 0
        
        # ATR
        atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1] if len(closes) >= 14 else 0
        
        # Volume indicators
        volume_sma = talib.SMA(volumes, timeperiod=20)[-1] if len(volumes) >= 20 else volumes[-1]
        volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1
        
        # Support/Resistance
        recent_high = np.max(highs[-20:]) if len(highs) >= 20 else highs[-1]
        recent_low = np.min(lows[-20:]) if len(lows) >= 20 else lows[-1]
        
        return {
            'rsi': rsi,
            'macd': macd_value,
            'macd_histogram': macd_hist_value,
            'bb_upper': bb_upper[-1] if not np.isnan(bb_upper[-1]) else recent_high,
            'bb_middle': bb_middle[-1] if not np.isnan(bb_middle[-1]) else np.mean(closes[-20:]),
            'bb_lower': bb_lower[-1] if not np.isnan(bb_lower[-1]) else recent_low,
            'bb_width': bb_width,
            'atr': atr,
            'atr_percent': (atr / closes[-1]) * 100 if closes[-1] > 0 else 0,
            'volume_ratio': volume_ratio,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'price_position': (closes[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5,
            'trend_strength': abs(macd_hist_value / closes[-1]) * 1000 if closes[-1] > 0 else 0
        }
    
    def generate_composite_signal(self, smc_signal: Dict, microstructure: Dict, 
                                 market_structure: Dict, ta_indicators: Dict) -> Dict:
        """Generate composite trading signal from all analyses."""
        
        composite = {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'reasons': [],
            'entry_zones': [],
            'targets': [],
            'stops': [],
            'risk_reward': 0
        }
        
        # Weighted scoring system
        scores = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0}
        max_score = 0
        
        # 1. Smart Money Concepts (40% weight)
        if smc_signal and 'signal' in smc_signal and smc_signal['signal'] != 'NEUTRAL':
            sm_weight = 40
            if smc_signal['signal'] == 'LONG':
                scores['LONG'] += sm_weight * (smc_signal.get('confidence', 0) / 100)
                composite['reasons'].append(f"SMC: {', '.join(smc_signal.get('reason', []))}")
            elif smc_signal['signal'] == 'SHORT':
                scores['SHORT'] += sm_weight * (smc_signal.get('confidence', 0) / 100)
                composite['reasons'].append(f"SMC: {', '.join(smc_signal.get('reason', []))}")
        
        # 2. Market Structure (25% weight)
        if market_structure:
            ms_weight = 25
            bias = market_structure.get('bias', 'NEUTRAL')
            
            if bias in ['BULLISH', 'BULLISH_BREAK']:
                scores['LONG'] += ms_weight * 0.8
                composite['reasons'].append(f"Market Structure: {bias}")
            elif bias in ['BEARISH', 'BEARISH_BREAK']:
                scores['SHORT'] += ms_weight * 0.8
                composite['reasons'].append(f"Market Structure: {bias}")
        
        # 3. Microstructure/Order Flow (20% weight)
        if microstructure:
            mf_weight = 20
            imbalance = microstructure.get('imbalance_strength', 'NEUTRAL')
            
            if imbalance == 'STRONG_BID':
                scores['LONG'] += mf_weight * 0.7
                composite['reasons'].append(f"Order Flow: Bid Dominance")
            elif imbalance == 'STRONG_ASK':
                scores['SHORT'] += mf_weight * 0.7
                composite['reasons'].append(f"Order Flow: Ask Dominance")
        
        # 4. Technical Indicators (15% weight)
        if ta_indicators:
            ta_weight = 15
            
            # RSI based
            rsi = ta_indicators.get('rsi', 50)
            if rsi < 35:
                scores['LONG'] += ta_weight * 0.6
                composite['reasons'].append(f"RSI Oversold: {rsi:.1f}")
            elif rsi > 65:
                scores['SHORT'] += ta_weight * 0.6
                composite['reasons'].append(f"RSI Overbought: {rsi:.1f}")
            
            # MACD histogram
            macd_hist = ta_indicators.get('macd_histogram', 0)
            if macd_hist > 0:
                scores['LONG'] += ta_weight * 0.4
            elif macd_hist < 0:
                scores['SHORT'] += ta_weight * 0.4
        
        # Determine final signal
        max_score = max(scores.values())
        
        if max_score >= 50:  # Minimum threshold
            if scores['LONG'] > scores['SHORT'] and scores['LONG'] > scores['NEUTRAL']:
                composite['signal'] = 'LONG'
                composite['confidence'] = min(95, scores['LONG'])
            elif scores['SHORT'] > scores['LONG'] and scores['SHORT'] > scores['NEUTRAL']:
                composite['signal'] = 'SHORT'
                composite['confidence'] = min(95, scores['SHORT'])
        
        # Calculate targets if we have a signal
        if composite['signal'] != 'NEUTRAL' and composite['confidence'] >= MIN_CONFIDENCE:
            composite = self.calculate_trade_levels(composite, smc_signal, ta_indicators)
        
        return composite
    
    def calculate_trade_levels(self, composite: Dict, smc_signal: Dict, ta_indicators: Dict) -> Dict:
        """Calculate trade entry, targets, and stops."""
        
        if composite['signal'] == 'LONG':
            # Entry zone (within 0.1% of current price)
            current_price = ta_indicators.get('bb_middle', 0) or smc_signal.get('current_price', 0)
            atr = ta_indicators.get('atr', 0)
            
            entry_low = current_price * 0.999
            entry_high = current_price * 1.001
            composite['entry_zones'] = [entry_low, entry_high]
            
            # Stop loss (below recent low or 1.5x ATR)
            recent_low = ta_indicators.get('recent_low', current_price * 0.99)
            stop_atr = current_price - (atr * 1.5)
            stop_loss = min(recent_low, stop_atr)
            composite['stops'] = [stop_loss]
            
            # Take profit levels
            tp1 = current_price + (atr * 2)
            tp2 = current_price + (atr * 3)
            tp3 = current_price + (atr * 4)
            composite['targets'] = [tp1, tp2, tp3]
            
            # Risk/Reward
            risk = current_price - stop_loss
            reward = tp3 - current_price
            composite['risk_reward'] = reward / risk if risk > 0 else 0
            
        elif composite['signal'] == 'SHORT':
            # Entry zone
            current_price = ta_indicators.get('bb_middle', 0) or smc_signal.get('current_price', 0)
            atr = ta_indicators.get('atr', 0)
            
            entry_low = current_price * 0.999
            entry_high = current_price * 1.001
            composite['entry_zones'] = [entry_low, entry_high]
            
            # Stop loss
            recent_high = ta_indicators.get('recent_high', current_price * 1.01)
            stop_atr = current_price + (atr * 1.5)
            stop_loss = max(recent_high, stop_atr)
            composite['stops'] = [stop_loss]
            
            # Take profit levels
            tp1 = current_price - (atr * 2)
            tp2 = current_price - (atr * 3)
            tp3 = current_price - (atr * 4)
            composite['targets'] = [tp1, tp2, tp3]
            
            # Risk/Reward
            risk = stop_loss - current_price
            reward = current_price - tp3
            composite['risk_reward'] = reward / risk if risk > 0 else 0
        
        return composite

# =============================================
# SIGNAL GENERATOR
# =============================================

class UltimateSignalGenerator:
    """Generates ultimate scalping signals with SMC."""
    
    def __init__(self, binance_client: BinanceClient, analyzer: EnhancedTechnicalAnalyzer):
        self.binance_client = binance_client
        self.analyzer = analyzer
        self.signal_history = deque(maxlen=100)
        self.last_signal_time = {}
        
    async def generate_signal(self, symbol: str, log_callback) -> Optional[Dict]:
        """Generate ultimate scalping signal."""
        
        # Rate limiting: don't generate signals too frequently
        current_time = time.time()
        if symbol in self.last_signal_time:
            if current_time - self.last_signal_time[symbol] < 30:  # 30 seconds cooldown
                return None
        
        try:
            # Perform comprehensive analysis
            analysis = await self.analyzer.analyze_symbol(symbol)
            
            if not analysis or 'composite_signal' not in analysis:
                return None
            
            signal_data = analysis['composite_signal']
            
            # Check minimum requirements
            if signal_data['signal'] == 'NEUTRAL':
                log_callback(f"⏸️ {symbol}: No clear signal (Confidence: {signal_data['confidence']:.1f}%)")
                return None
            
            if signal_data['confidence'] < MIN_CONFIDENCE:
                log_callback(f"⏸️ {symbol}: Confidence {signal_data['confidence']:.1f}% < {MIN_CONFIDENCE}%")
                return None
            
            if signal_data.get('risk_reward', 0) < 1.5:
                log_callback(f"⏸️ {symbol}: RRR {signal_data['risk_reward']:.1f} < 1.5")
                return None
            
            # Create signal object
            signal = {
                'signal_id': f"SIG-{int(time.time())}-{random.randint(1000, 9999)}",
                'symbol': symbol,
                'direction': signal_data['signal'],
                'entry_zones': signal_data.get('entry_zones', []),
                'stop_loss': signal_data.get('stops', [0])[0],
                'take_profits': signal_data.get('targets', []),
                'confidence': signal_data['confidence'],
                'risk_reward': signal_data.get('risk_reward', 0),
                'reasons': signal_data.get('reasons', []),
                'analysis': analysis,
                'created_at': datetime.now(),
                'expiry': datetime.now() + timedelta(seconds=MAX_TRADE_DURATION)
            }
            
            # Calculate position size
            current_price = analysis.get('current_price', 0)
            stop_loss = signal['stop_loss']
            
            if signal['direction'] == 'LONG':
                risk_amount = current_price - stop_loss
            else:
                risk_amount = stop_loss - current_price
            
            risk_percent = risk_amount / current_price if current_price > 0 else 0
            position_size = (RISK_PER_TRADE / risk_percent) if risk_percent > 0 else 0
            
            signal['position_size'] = position_size
            signal['risk_percent'] = risk_percent * 100
            
            # Log the signal
            self.log_signal(signal, log_callback)
            
            # Update last signal time
            self.last_signal_time[symbol] = current_time
            
            # Add to history
            self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            log_callback(f"❌ Error generating signal for {symbol}: {str(e)}")
            return None
    
    def log_signal(self, signal: Dict, log_callback):
        """Log signal details."""
        direction_emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
        
        log_callback(f"🎯 {signal['symbol']} {direction_emoji} {signal['direction']} SIGNAL")
        log_callback(f"   Signal ID: {signal['signal_id']}")
        log_callback(f"   Confidence: {signal['confidence']:.1f}%")
        log_callback(f"   RRR: 1:{signal['risk_reward']:.1f}")
        
        if signal['entry_zones']:
            log_callback(f"   Entry Zone: ${signal['entry_zones'][0]:.2f} - ${signal['entry_zones'][-1]:.2f}")
        
        log_callback(f"   Stop Loss: ${signal['stop_loss']:.2f}")
        
        if signal['take_profits']:
            tp_text = " | ".join([f"TP{i+1}: ${tp:.2f}" for i, tp in enumerate(signal['take_profits'][:3])])
            log_callback(f"   {tp_text}")
        
        log_callback(f"   Position Size: {signal['position_size']:.4f}")
        log_callback(f"   Risk: {signal['risk_percent']:.2f}%")
        
        if signal['reasons']:
            log_callback(f"   Reasons: {' | '.join(signal['reasons'][:3])}")

# =============================================
# ADVANCED TRADE MANAGER
# =============================================

class AdvancedTradeManager:
    """Manages trades with advanced risk management."""
    
    def __init__(self, binance_client: BinanceClient, telegram_manager):
        self.binance_client = binance_client
        self.telegram_manager = telegram_manager
        self.active_trades = {}
        self.trade_history = []
        self.daily_pnl = 0
        self.daily_trades = 0
        self.max_daily_trades = 20
        
    async def execute_trade(self, signal: Dict, log_callback) -> bool:
        """Execute a trade."""
        
        # Check max concurrent trades
        if len(self.active_trades) >= MAX_CONCURRENT_TRADES:
            log_callback(f"⚠️ Max concurrent trades reached ({MAX_CONCURRENT_TRADES})")
            return False
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            log_callback(f"⚠️ Daily trade limit reached ({self.max_daily_trades})")
            return False
        
        # Check daily risk limit
        if abs(self.daily_pnl) > MAX_DAILY_RISK:
            log_callback(f"⚠️ Daily risk limit reached ({self.daily_pnl:.2f}%)")
            return False
        
        # Create trade object
        trade_id = f"TRADE-{int(time.time())}-{random.randint(1000, 9999)}"
        
        trade = {
            'trade_id': trade_id,
            'signal': signal,
            'entry_price': signal.get('entry_zones', [0])[0],  # Use lower entry zone
            'stop_loss': signal['stop_loss'],
            'take_profits': signal['take_profits'],
            'position_size': signal['position_size'],
            'direction': signal['direction'],
            'status': 'OPEN',
            'entry_time': datetime.now(),
            'partial_closes': [],
            'trailing_stop': None
        }
        
        # Add to active trades
        self.active_trades[trade_id] = trade
        self.daily_trades += 1
        
        # Log trade
        log_callback(f"✅ TRADE EXECUTED: {trade_id}")
        log_callback(f"   {signal['symbol']} {signal['direction']}")
        log_callback(f"   Entry: ${trade['entry_price']:.2f}")
        log_callback(f"   Stop Loss: ${trade['stop_loss']:.2f}")
        log_callback(f"   Position: {trade['position_size']:.4f}")
        
        # Send Telegram alert
        await self.send_telegram_alert(signal, trade)
        
        return True
    
    async def monitor_trades(self, log_callback):
        """Monitor and manage active trades."""
        closed_trades = []
        
        for trade_id, trade in list(self.active_trades.items()):
            try:
                symbol = trade['signal']['symbol']
                current_data = self.binance_client.get_klines(symbol, '1m', 5)
                
                if current_data is None or len(current_data) == 0:
                    continue
                
                current_price = current_data['close'].iloc[-1]
                direction = trade['direction']
                
                # Update trailing stop for profitable trades
                if direction == 'LONG' and current_price > trade['entry_price']:
                    new_trailing_stop = current_price * (1 - TRAILING_STOP_PERCENT)
                    if trade['trailing_stop'] is None or new_trailing_stop > trade['trailing_stop']:
                        trade['trailing_stop'] = new_trailing_stop
                
                elif direction == 'SHORT' and current_price < trade['entry_price']:
                    new_trailing_stop = current_price * (1 + TRAILING_STOP_PERCENT)
                    if trade['trailing_stop'] is None or new_trailing_stop < trade['trailing_stop']:
                        trade['trailing_stop'] = new_trailing_stop
                
                # Check stop loss (including trailing stop)
                stop_loss_price = trade['stop_loss']
                if trade['trailing_stop'] is not None:
                    if direction == 'LONG':
                        stop_loss_price = max(stop_loss_price, trade['trailing_stop'])
                    else:
                        stop_loss_price = min(stop_loss_price, trade['trailing_stop'])
                
                # Check for stop loss hit
                if (direction == 'LONG' and current_price <= stop_loss_price) or \
                   (direction == 'SHORT' and current_price >= stop_loss_price):
                    
                    await self.close_trade(trade_id, current_price, 'STOP_LOSS', log_callback)
                    closed_trades.append(trade_id)
                    continue
                
                # Check for take profit hits
                take_profits = trade['take_profits']
                for i, tp in enumerate(take_profits):
                    tp_level = f'TP{i+1}'
                    
                    if tp_level in trade['partial_closes']:
                        continue
                    
                    if (direction == 'LONG' and current_price >= tp) or \
                       (direction == 'SHORT' and current_price <= tp):
                        
                        # Partial close
                        close_percentage = PARTIAL_TP_LEVELS[i] if i < len(PARTIAL_TP_LEVELS) else 0.5
                        await self.partial_close(trade_id, tp, close_percentage, tp_level, log_callback)
                        trade['partial_closes'].append(tp_level)
                        
                        # If all TPs hit, close trade
                        if len(trade['partial_closes']) >= len(take_profits):
                            await self.close_trade(trade_id, current_price, 'ALL_TP_HIT', log_callback)
                            closed_trades.append(trade_id)
                
                # Check expiry
                if datetime.now() > trade['signal']['expiry']:
                    await self.close_trade(trade_id, current_price, 'EXPIRED', log_callback)
                    closed_trades.append(trade_id)
                    
            except Exception as e:
                log_callback(f"❌ Error monitoring trade {trade_id}: {str(e)}")
        
        # Remove closed trades
        for trade_id in closed_trades:
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
    
    async def partial_close(self, trade_id: str, price: float, percentage: float, 
                           level: str, log_callback):
        """Partially close a trade."""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        
        # Calculate PnL for this partial close
        if trade['direction'] == 'LONG':
            pnl = (price - trade['entry_price']) * trade['position_size'] * percentage
        else:
            pnl = (trade['entry_price'] - price) * trade['position_size'] * percentage
        
        pnl_percent = (pnl / (trade['entry_price'] * trade['position_size'])) * 100
        
        log_callback(f"🎯 {trade_id}: Partial close at {level}")
        log_callback(f"   Price: ${price:.2f}")
        log_callback(f"   PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
        
        # Update position size
        trade['position_size'] *= (1 - percentage)
        
        # Update daily PnL
        self.daily_pnl += pnl_percent
    
    async def close_trade(self, trade_id: str, price: float, reason: str, log_callback):
        """Close a trade completely."""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        
        # Calculate final PnL
        if trade['direction'] == 'LONG':
            pnl = (price - trade['entry_price']) * trade['position_size']
        else:
            pnl = (trade['entry_price'] - price) * trade['position_size']
        
        pnl_percent = (pnl / (trade['entry_price'] * trade['position_size'])) * 100
        
        # Calculate pips
        pip_size = 0.01  # For crypto
        if trade['direction'] == 'LONG':
            pips = (price - trade['entry_price']) / pip_size
        else:
            pips = (trade['entry_price'] - price) / pip_size
        
        log_callback(f"🔒 TRADE CLOSED: {trade_id}")
        log_callback(f"   Reason: {reason}")
        log_callback(f"   Exit: ${price:.2f}")
        log_callback(f"   PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
        log_callback(f"   Pips: {pips:+.1f}")
        log_callback(f"   {'💰 PROFIT' if pnl > 0 else '💸 LOSS'}")
        
        # Update daily PnL
        self.daily_pnl += pnl_percent
        
        # Send Telegram closure
        await self.send_telegram_closure(trade, price, pnl, pnl_percent, pips, reason)
        
        # Add to history
        self.trade_history.append({
            'trade_id': trade_id,
            'symbol': trade['signal']['symbol'],
            'direction': trade['direction'],
            'entry': trade['entry_price'],
            'exit': price,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'pips': pips,
            'reason': reason,
            'duration': (datetime.now() - trade['entry_time']).total_seconds() / 60
        })
    
    async def send_telegram_alert(self, signal: Dict, trade: Dict):
        """Send Telegram alert for new trade."""
        try:
            direction_emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
            
            message = f"""
⚡ ULTIMATE SCALPING SIGNAL ⚡

{direction_emoji} <strong>{signal['symbol']} {signal['direction']}</strong>
Trade ID: {trade['trade_id']}
Confidence: <strong>{signal['confidence']:.1f}%</strong>

🎯 <strong>Trade Levels:</strong>
Entry: ${trade['entry_price']:.2f}
Stop Loss: ${trade['stop_loss']:.2f}
Take Profit: ${trade['take_profits'][-1]:.2f}

📊 <strong>Risk Management:</strong>
Position Size: {trade['position_size']:.4f}
Risk/Reward: 1:{signal['risk_reward']:.1f}
Risk: {signal.get('risk_percent', 0):.2f}%

💡 <strong>Signal Reasons:</strong>
{chr(10).join(signal['reasons'][:3])}

Time: {datetime.now().strftime('%H:%M:%S')}
"""
            
            self.telegram_manager.send_message_sync(message)
            
        except Exception as e:
            print(f"❌ Telegram alert error: {e}")
    
    async def send_telegram_closure(self, trade: Dict, exit_price: float, 
                                  pnl: float, pnl_percent: float, pips: float, reason: str):
        """Send Telegram closure alert."""
        try:
            result_emoji = "💰" if pnl > 0 else "💸"
            
            message = f"""
{result_emoji} <strong>TRADE CLOSED</strong> {result_emoji}

📊 <strong>Performance Summary:</strong>
Symbol: {trade['signal']['symbol']}
Direction: {trade['direction']}
Trade ID: {trade['trade_id']}
Duration: Scalping Trade

💵 <strong>Results:</strong>
Entry: ${trade['entry_price']:.2f}
Exit: ${exit_price:.2f}
PnL: ${pnl:.2f}
PnL %: {pnl_percent:.2f}%
Pips: {pips:+.1f}

📝 <strong>Details:</strong>
Reason: {reason}
Confidence Was: {trade['signal']['confidence']:.1f}%

Closed at: {datetime.now().strftime('%H:%M:%S')}
"""
            
            self.telegram_manager.send_message_sync(message)
            
        except Exception as e:
            print(f"❌ Telegram closure error: {e}")

# =============================================
# TELEGRAM MANAGER (SAME AS BEFORE)
# =============================================

class TelegramManager:
    """Manages Telegram notifications."""
    
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
                    print(f"⚠️ Telegram rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ Telegram error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        
            except Exception as e:
                print(f"⚠️ Telegram network error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return False

# =============================================
# ULTIMATE SCALPING BOT
# =============================================

class UltimateScalpingBot:
    """The ultimate scalping bot with Smart Money Concepts."""
    
    def __init__(self):
        print("="*70)
        print("🤖 ULTIMATE SMART MONEY SCALPING BOT")
        print("="*70)
        
        # Initialize Telegram
        print("📡 Initializing Telegram...")
        self.telegram = TelegramManager()
        
        # Initialize Binance client
        print("💰 Initializing Binance API...")
        self.binance_client = BinanceClient(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_API_SECRET,
            testnet=BINANCE_TESTNET
        )
        
        # Initialize CoinGecko client
        if USE_COINGECKO_API:
            print("📊 Initializing CoinGecko API...")
            self.coingecko_client = CoinGeckoClient(api_key=COINGECKO_API_KEY)
        
        # Initialize analyzers
        print("🧠 Initializing Smart Money Analyzer...")
        self.analyzer = EnhancedTechnicalAnalyzer(self.binance_client)
        
        print("🎯 Initializing Signal Generator...")
        self.signal_generator = UltimateSignalGenerator(self.binance_client, self.analyzer)
        
        print("💰 Initializing Trade Manager...")
        self.trade_manager = AdvancedTradeManager(self.binance_client, self.telegram)
        
        # State
        self.cycle_count = 0
        self.signals_today = 0
        self.paused = True
        self.gui = None
        
        print("✅ Ultimate Scalping Bot initialized successfully")
        print(f"   • Strategy Mode: {STRATEGY_MODE}")
        print(f"   • Smart Money Concepts: ENABLED")
        print(f"   • Market Microstructure: ENABLED")
        print(f"   • Fair Value Gaps: ENABLED")
        print(f"   • Timeframes: {', '.join(TIMEFRAMES)}")
        print(f"   • Trading Pairs: {', '.join(TRADING_PAIRS)}")
        print(f"   • Risk per Trade: {RISK_PER_TRADE*100}%")
        print(f"   • Max Daily Risk: {MAX_DAILY_RISK*100}%")
        print("="*70)
    
    def set_gui(self, gui):
        """Set GUI."""
        self.gui = gui
        self.gui.add_log(f"🤖 Ultimate Scalping Bot Ready")
        self.gui.add_log(f"🎯 Strategy: {STRATEGY_MODE}")
        self.gui.add_log(f"📡 Telegram: @TheUltimateScalperBot")
        self.gui.add_log(f"💰 Binance API: {'✅ Connected' if self.binance_client.api_key else '⚠️ Public Only'}")
        self.gui.add_log(f"🧠 Smart Money Concepts: ENABLED")
        self.gui.add_log(f"📊 Market Microstructure: ENABLED")
        self.gui.add_log("🎯 Press START to begin ultimate scalping")
    
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
                # Check active trades for this symbol
                active_for_symbol = sum(
                    1 for trade in self.trade_manager.active_trades.values()
                    if trade['signal']['symbol'] == symbol
                )
                
                if active_for_symbol >= 1:  # Max 1 trade per symbol
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
        print("🚀 Starting Ultimate Scalping Bot...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.gui:
            self.gui.add_log("✅ All systems initialized")
            self.gui.add_log("✅ Smart Money Analysis active")
            self.gui.add_log("✅ Telegram notifications enabled")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            if self.gui:
                self.gui.add_log("🛑 Bot stopped by user")
            
            # Send shutdown message
            self.telegram.send_message_sync(
                f"🤖 Ultimate Scalping Bot Shutdown\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total Cycles: {self.cycle_count}\n"
                f"Total Signals: {self.signals_today}"
            )
            
            if self.gui:
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
# SIMPLE GUI FOR THE BOT
# =============================================

class SimpleScalpingGUI:
    """Simple GUI for the scalping bot."""
    
    def __init__(self, bot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("🤖 ULTIMATE SMART MONEY SCALPING BOT")
        self.root.geometry("1200x800")
        
        # Configure dark theme
        self.setup_styles()
        self.init_ui()
        
        print("✅ Simple GUI initialized")
    
    def setup_styles(self):
        """Setup styling."""
        self.root.configure(bg='#0a0a0a')
    
    def init_ui(self):
        """Initialize UI."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="🤖 ULTIMATE SMART MONEY SCALPING BOT",
                              font=('Arial', 16, 'bold'),
                              bg='#0a0a0a', fg='#00ff00')
        title_label.pack(pady=10)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(main_frame, text="📊 LIVE STATS", padding=10)
        stats_frame.pack(fill='x', pady=10)
        
        # Stats labels
        self.stats_labels = {}
        stats_data = [
            ("Cycles:", "cycles", "0"),
            ("Signals Today:", "signals", "0"),
            ("Active Trades:", "active_trades", "0"),
            ("Daily PnL:", "daily_pnl", "0.00%"),
            ("Daily Trades:", "daily_trades", "0/20")
        ]
        
        for label_text, key, default in stats_data:
            frame = ttk.Frame(stats_frame)
            frame.pack(fill='x', pady=2)
            
            ttk.Label(frame, text=label_text, width=15).pack(side='left')
            self.stats_labels[key] = ttk.Label(frame, text=default, 
                                              font=('Arial', 10, 'bold'))
            self.stats_labels[key].pack(side='right')
        
        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="⚙️ CONTROLS", padding=10)
        controls_frame.pack(fill='x', pady=10)
        
        # Control buttons
        self.start_btn = ttk.Button(controls_frame, text="▶️ START BOT", 
                                   command=self.start_bot, width=20)
        self.start_btn.pack(pady=5)
        
        self.pause_btn = ttk.Button(controls_frame, text="⏸️ PAUSE BOT", 
                                   command=self.pause_bot, width=20,
                                   state='disabled')
        self.pause_btn.pack(pady=5)
        
        ttk.Button(controls_frame, text="📡 TEST TELEGRAM", 
                  command=self.test_telegram, width=20).pack(pady=5)
        
        ttk.Button(controls_frame, text="🗑️ CLEAR LOGS", 
                  command=self.clear_logs, width=20).pack(pady=5)
        
        # Trading pairs info
        pairs_frame = ttk.LabelFrame(main_frame, text="🎯 TRADING PAIRS", padding=10)
        pairs_frame.pack(fill='x', pady=10)
        
        for symbol in TRADING_PAIRS:
            ttk.Label(pairs_frame, text=f"• {symbol}", 
                     font=('Arial', 10)).pack(anchor='w')
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="📝 LIVE TRADING LOG", padding=5)
        log_frame.pack(fill='both', expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20,
                                                 font=('Consolas', 9),
                                                 bg='#0a0a0a', fg='#00ff00',
                                                 insertbackground='white')
        self.log_text.pack(fill='both', expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, 
                                   text="🤖 Ultimate Scalping Bot Ready | Press START",
                                   relief='sunken',
                                   anchor='center',
                                   font=('Arial', 10))
        self.status_bar.pack(side='bottom', fill='x')
        
        # Start update timer
        self.start_update_timer()
    
    def start_update_timer(self):
        """Start update timer."""
        try:
            self.update_ui()
        except Exception as e:
            print(f"⚠️ UI update error: {e}")
        finally:
            self.root.after(2000, self.start_update_timer)
    
    def update_ui(self):
        """Update UI with current stats."""
        try:
            # Update status
            status = "RUNNING" if not self.bot.paused else "PAUSED"
            
            self.status_bar.config(
                text=f"🤖 {status} | "
                     f"Cycles: {self.bot.cycle_count} | "
                     f"Signals: {self.bot.signals_today} | "
                     f"Active Trades: {len(self.bot.trade_manager.active_trades)} | "
                     f"{datetime.now().strftime('%H:%M:%S')}"
            )
            
            # Update stats
            self.stats_labels['cycles'].config(text=str(self.bot.cycle_count))
            self.stats_labels['signals'].config(text=str(self.bot.signals_today))
            self.stats_labels['active_trades'].config(
                text=str(len(self.bot.trade_manager.active_trades))
            )
            self.stats_labels['daily_pnl'].config(
                text=f"{self.bot.trade_manager.daily_pnl:.2f}%"
            )
            self.stats_labels['daily_trades'].config(
                text=f"{self.bot.trade_manager.daily_trades}/20"
            )
            
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
        self.add_log("▶️ Ultimate Scalping Bot STARTED")
        self.add_log(f"🎯 Strategy: {STRATEGY_MODE}")
        self.add_log("🧠 Smart Money Concepts: ACTIVE")
        self.add_log("📊 Market Microstructure: MONITORING")
        self.add_log("🎯 Looking for high-probability setups...")
    
    def pause_bot(self):
        """Pause bot."""
        self.bot.paused = True
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.add_log("⏸️ Bot PAUSED")
    
    def test_telegram(self):
        """Test Telegram."""
        self.add_log("📡 Testing Telegram...")
        if self.bot.telegram.test_connection():
            self.add_log("✅ Telegram is working!")
        else:
            self.add_log("❌ Telegram test failed")

# =============================================
# MAIN ENTRY POINT
# =============================================

def main():
    """Start the ultimate scalping bot."""
    bot = UltimateScalpingBot()
    
    # Create GUI
    gui = SimpleScalpingGUI(bot)
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
    print("🤖 ULTIMATE SMART MONEY SCALPING BOT")
    print("="*70)
    
    print("\n🚀 ULTIMATE FEATURES:")
    print("1. 🧠 SMART MONEY CONCEPTS (SMC)")
    print("   • Fair Value Gaps (FVGs)")
    print("   • Order Blocks (OBs)")
    print("   • Breaker Blocks")
    print("   • Market Structure Analysis")
    
    print("\n2. 📊 MARKET MICROSTRUCTURE")
    print("   • Order Flow Analysis")
    print("   • Liquidity Pool Detection")
    print("   • Bid/Ask Imbalance")
    print("   • Volume Profile")
    
    print("\n3. 📡 REAL-TIME DATA")
    print("   • Binance Futures API")
    print("   • CoinGecko API")
    print("   • Multi-timeframe Analysis (1m/5m/15m)")
    
    print("\n4. ⚡ ADVANCED SCALPING")
    print("   • Ultra-fast 5-second scanning")
    print("   • Trailing Stops")
    print("   • Partial Take Profits")
    print("   • Position Scaling")
    print("   • Daily Risk Limits")
    
    print("\n🎯 TRADING PARAMETERS:")
    print(f"• Strategy: {STRATEGY_MODE}")
    print(f"• Pairs: {', '.join(TRADING_PAIRS)}")
    print(f"• Risk per Trade: {RISK_PER_TRADE*100}%")
    print(f"• Max Daily Risk: {MAX_DAILY_RISK*100}%")
    print(f"• Minimum Confidence: {MIN_CONFIDENCE}%")
    print(f"• Minimum RRR: 1:1.5")
    
    print("\n📡 TELEGRAM BOT:")
    print(f"• Bot: @TheUltimateScalperBot")
    
    print("\n⚠️ IMPORTANT:")
    print("• Add your Binance API keys for full functionality")
    print("• Start with small position sizes")
    print("• Monitor performance closely")
    print("="*70 + "\n")
    
    main()
