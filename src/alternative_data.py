"""
Alternative Data Sources for Stock Data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings('ignore')

class AlphaVantageLoader:
    """Alternative data loader using Alpha Vantage API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or 'demo'  # Use 'demo' for free tier
        self.base_url = 'https://www.alphavantage.co/query'
    
    def fetch_data(self, ticker, start_date, end_date):
        """Fetch data from Alpha Vantage"""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                print(f"⚠️ No data found for {ticker} from Alpha Vantage")
                return None
            
            # Convert to DataFrame
            ts_data = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'Dividend', 'Split']
            df = df.astype(float)
            
            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]
            
            return df
            
        except Exception as e:
            print(f"Error fetching from Alpha Vantage: {e}")
            return None

def get_stock_data_fallback(ticker, start_date, end_date):
    """Try multiple data sources"""
    
    # Method 1: Try yfinance with proper headers
    try:
        import yfinance as yf
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        if not data.empty:
            return data
    except:
        pass
    
    # Method 2: Try Alpha Vantage
    try:
        loader = AlphaVantageLoader()
        data = loader.fetch_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            return data
    except:
        pass
    
    # Method 3: Try Yahoo Finance without auto_adjust
    try:
        import yfinance as yf
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        if not data.empty:
            return data
    except:
        pass
    
    print(f"❌ Failed to fetch data for {ticker} from all sources")
    return None