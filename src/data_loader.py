"""
Data Acquisition Module
Fetches stock data from Yahoo Finance
"""

import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class StockDataLoader:
    """A comprehensive data loader for stock market data
    
    Attributes:
        ticker: Stock ticker symbol
        start_date: Start date for data fetch
        end_date: End date for data fetch
        data: Fetched stock data
    """
    
    def __init__(self, ticker: str = "^GSPC", start_date: str = "2018-01-01", end_date: str = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data = None
        
    def fetch_yahoo_data(self):
        """Fetch historical stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
                
            print(f"Successfully fetched {len(self.data)} rows of data")
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def add_technical_indicators(self, df=None):
        """Add technical indicators to the dataset"""
        if df is None:
            df = self.data.copy()
        
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
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
        if self.data is None:
            self.fetch_yahoo_data()
        
        if add_indicators:
            self.data = self.add_technical_indicators()
        
        return self.data

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to validate
            
        Returns:
        --------
        bool: True if data passes validation
        """
        if df is None or df.empty:
            logger.error("Data is empty or None")
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        
        if df['Close'].isnull().sum() > len(df) * 0.1:
            logger.warning("More than 10% null values in Close column")
            return False
        
        logger.info("Data validation passed")
        return True


if __name__ == "__main__":
    # Test the data loader
    loader = StockDataLoader(ticker="AAPL", start_date="2020-01-01")
    data = loader.get_ready_data(add_indicators=True)
    print(data.head())
    print(f"\nData shape: {data.shape}")