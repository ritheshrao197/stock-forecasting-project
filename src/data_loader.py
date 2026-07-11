"""
Data Acquisition Module - Fixed for Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import warnings
warnings.filterwarnings('ignore')

class StockDataLoader:
    """A comprehensive data loader for stock market data with Yahoo Finance fixes"""
    
    def __init__(self, ticker="AAPL", start_date="2020-01-01", end_date=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data = None
        
        # Fix Yahoo Finance headers
        self._fix_yfinance_headers()
    
    def _fix_yfinance_headers(self):
        """Fix Yahoo Finance headers to avoid blocking"""
        try:
            # Override yfinance session headers
            yf.set_tz_cache_location(None)
            
            # Create a custom session with proper headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            })
            
            # Set the session for yfinance
            yf.Ticker._session = session
        except Exception as e:
            print(f"⚠️ Could not set custom headers: {e}")
    
    def fetch_yahoo_data(self, max_retries=3, delay=2):
        """
        Fetch historical stock data from Yahoo Finance with retries
        
        Parameters:
        -----------
        max_retries : int
            Number of retry attempts
        delay : int
            Delay between retries in seconds
        """
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        
        for attempt in range(max_retries):
            try:
                # Try with different parameters
                stock = yf.Ticker(self.ticker)
                
                # Try to get info first to verify ticker exists
                try:
                    info = stock.info
                    if not info or 'longName' not in info:
                        print(f"⚠️ Could not get info for {self.ticker}")
                except:
                    print(f"⚠️ Could not get info for {self.ticker}")
                
                # Fetch history with retry
                try:
                    self.data = stock.history(
                        start=self.start_date,
                        end=self.end_date,
                        interval="1d",
                        auto_adjust=True
                    )
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    else:
                        raise
                
                if self.data.empty:
                    # Try without auto_adjust
                    print("Trying without auto_adjust...")
                    self.data = stock.history(
                        start=self.start_date,
                        end=self.end_date,
                        interval="1d",
                        auto_adjust=False
                    )
                
                if self.data.empty:
                    print(f"⚠️ No data found for {self.ticker}")
                    return None
                
                print(f"✅ Successfully fetched {len(self.data)} rows of data")
                return self.data
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay * (attempt + 1)} seconds...")
                    time.sleep(delay * (attempt + 1))
                else:
                    print(f"❌ Failed to fetch data for {self.ticker} after {max_retries} attempts")
                    return None
        
        return None
    
    def fetch_with_fallback(self):
        """
        Fetch data with multiple fallback methods
        """
        # Method 1: Try normal fetch
        data = self.fetch_yahoo_data()
        if data is not None and not data.empty:
            return data
        
        # Method 2: Try alternative ticker format
        print(f"Trying alternative ticker format...")
        alt_ticker = self.ticker.replace('.', '-') if '.' in self.ticker else self.ticker
        if alt_ticker != self.ticker:
            self.ticker = alt_ticker
            data = self.fetch_yahoo_data()
            if data is not None and not data.empty:
                return data
        
        # Method 3: Try with different date range
        print("Trying with a shorter date range...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        self.start_date = start_date.strftime("%Y-%m-%d")
        data = self.fetch_yahoo_data()
        if data is not None and not data.empty:
            return data
        
        # Method 4: Use alternative data source
        print("Trying alternative data sources...")
        try:
            # Try using alpha_vantage or other sources
            import yfinance as yf
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                threads=False
            )
            if data is not None and not data.empty:
                self.data = data
                return data
        except:
            pass
        
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
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
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
        
        # Price range
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        # Drop NaN values created by rolling windows
        df = df.dropna()
        
        print(f"Added technical indicators. New shape: {df.shape}")
        return df
    
    def get_ready_data(self, add_indicators=True):
        """Get data ready for modeling"""
        if self.data is None or self.data.empty:
            print("Fetching data...")
            self.data = self.fetch_with_fallback()
        
        if self.data is None or self.data.empty:
            print("❌ No data available")
            return None
        
        if add_indicators:
            self.data = self.add_technical_indicators()
        
        return self.data


# Alternative: Direct download function with retry
def download_stock_data(ticker, start_date, end_date, max_retries=3):
    """Download stock data with retry logic"""
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False
            )
            if not data.empty:
                return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


if __name__ == "__main__":
    # Test the data loader
    loader = StockDataLoader(ticker="AAPL", start_date="2022-01-01")
    data = loader.get_ready_data(add_indicators=True)
    if data is not None:
        print(data.head())
        print(f"\nData shape: {data.shape}")
    else:
        print("Failed to load data")