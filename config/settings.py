"""
Configuration settings for the Stock Market Forecasting app
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# App settings
APP_TITLE = "Stock Market Price Forecasting"
APP_ICON = "📈"
APP_LAYOUT = "wide"

# Default settings
DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_LOOKBACK = 60
DEFAULT_EPOCHS = 20

# Ticker categories
TICKER_CATEGORIES = {
    "📊 US Stock Indices": {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "NASDAQ Composite",
    },
    "🏢 US Tech Companies": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "TSLA": "Tesla Inc.",
    },
    "🏦 US Financial Companies": {
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America",
        "WFC": "Wells Fargo & Co.",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
    },
    "🛒 US Retail & Consumer": {
        "WMT": "Walmart Inc.",
        "TGT": "Target Corporation",
        "COST": "Costco Wholesale",
        "HD": "Home Depot Inc.",
    },
    "💊 US Healthcare": {
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "UNH": "UnitedHealth Group",
    },
    "📈 ETFs & Funds": {
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market",
        "VOO": "Vanguard S&P 500 ETF",
    }
}

def get_ticker_options():
    """Get flattened ticker options for dropdown"""
    options = []
    for category, tickers in TICKER_CATEGORIES.items():
        for ticker, name in tickers.items():
            options.append({
                "label": f"{name} ({ticker})",
                "value": ticker,
                "category": category
            })
    return sorted(options, key=lambda x: x["label"])