"""
Data Acquisition Module - With Rate Limiting and Caching
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import streamlit as st
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

class StockDataLoader:
    """A comprehensive data loader for stock market data with rate limiting"""
    
    # Class-level cache for ticker data
    _cache = {}
    _last_request_time = 0
    _min_request_interval = 2  # Minimum seconds between requests
    
    def __init__(self, ticker="RELIANCE.NS", start_date="2020-01-01", end_date=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data = None
        
        # Fix Yahoo Finance headers
        self._fix_yfinance_headers()
    
    def _fix_yfinance_headers(self):
        """Fix Yahoo Finance headers to avoid blocking"""
        try:
            yf.set_tz_cache_location(None)
            
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Cache-Control': 'max-age=0',
            })
            yf.Ticker._session = session
        except Exception as e:
            print(f"⚠️ Could not set custom headers: {e}")
    
    @staticmethod
    def _rate_limit():
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - StockDataLoader._last_request_time
        if time_since_last < StockDataLoader._min_request_interval:
            time.sleep(StockDataLoader._min_request_interval - time_since_last)
        StockDataLoader._last_request_time = time.time()
    
    def fetch_yahoo_data(self, max_retries=2, delay=3):
        """Fetch historical stock data with rate limiting and retries"""
        
        # Check cache first
        cache_key = f"{self.ticker}_{self.start_date}_{self.end_date}"
        if cache_key in StockDataLoader._cache:
            cached_data = StockDataLoader._cache[cache_key]
            if cached_data is not None and not cached_data.empty:
                print(f"✅ Using cached data for {self.ticker}")
                self.data = cached_data.copy()
                return self.data
        
        print(f"Fetching data for {self.ticker}...")
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self._rate_limit()
                
                stock = yf.Ticker(self.ticker)
                
                # Try to get info first
                try:
                    info = stock.info
                except:
                    pass
                
                # Fetch history
                self.data = stock.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d",
                    auto_adjust=True
                )
                
                if self.data.empty:
                    # Try without auto_adjust
                    self._rate_limit()
                    self.data = stock.history(
                        start=self.start_date,
                        end=self.end_date,
                        interval="1d",
                        auto_adjust=False
                    )
                
                if not self.data.empty:
                    # Cache the data
                    StockDataLoader._cache[cache_key] = self.data.copy()
                    print(f"✅ Successfully fetched {len(self.data)} rows of data")
                    return self.data
                
                print(f"⚠️ No data found for {self.ticker}")
                return None
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    wait_time = delay * (attempt + 1)
                    print(f"⏳ Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts")
                    return None
        
        return None
    
    def fetch_with_fallback(self):
        """Fetch data with fallback methods and progressive delays"""
        
        # Method 1: Try normal fetch
        data = self.fetch_yahoo_data()
        if data is not None and not data.empty:
            return data
        
        # Method 2: Try alternative ticker format
        alt_ticker = self.ticker.replace('.', '-') if '.' in self.ticker else self.ticker
        if alt_ticker != self.ticker:
            print(f"Trying alternative format: {alt_ticker}")
            self.ticker = alt_ticker
            time.sleep(2)
            data = self.fetch_yahoo_data()
            if data is not None and not data.empty:
                return data
        
        # Method 3: Try with shorter date range
        print("Trying with shorter date range...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        self.start_date = start_date.strftime("%Y-%m-%d")
        time.sleep(2)
        data = self.fetch_yahoo_data()
        if data is not None and not data.empty:
            return data
        
        print(f"❌ All methods failed for {self.ticker}")
        return None
    
    def add_technical_indicators(self, df=None):
        """Add technical indicators to the dataset"""
        if df is None:
            df = self.data.copy()
        
        if df is None or df.empty:
            print("No data to add indicators to")
            return df
        
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Drop NaN values
        df = df.dropna()
        
        print(f"Added technical indicators. New shape: {df.shape}")
        return df
    
    def get_ready_data(self, add_indicators=True):
        """Get data ready for modeling with caching"""
        if self.data is None or self.data.empty:
            self.data = self.fetch_with_fallback()
        
        if self.data is None or self.data.empty:
            print("❌ No data available")
            return None
        
        if add_indicators:
            self.data = self.add_technical_indicators()
        
        return self.data

def download_with_retry(ticker, start_date, end_date, max_retries=3):
    """Download stock data with retry logic"""
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=30
            )
            if not data.empty:
                return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None