"""
Configuration settings for the Stock Market Forecasting app
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
STATIC_DIR = BASE_DIR / 'static'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True)

# App settings
APP_TITLE = "Stock Market Price Forecasting"
APP_ICON = "📈"
APP_LAYOUT = "wide"

# Model settings
DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_LOOKBACK = 60
DEFAULT_EPOCHS = 20

# Logging settings
MAX_LOG_ENTRIES = 200
LOG_AUTO_REFRESH_INTERVAL = 3  # seconds

# Ticker categories (moved from main app)
TICKER_CATEGORIES = {
    "📊 US Stock Indices": {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "NASDAQ Composite",
        "^RUT": "Russell 2000",
        "^VIX": "CBOE Volatility Index",
        "^TNX": "10-Year Treasury Yield"
    },
    "🏢 US Tech Companies": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "TSLA": "Tesla Inc.",
        "NFLX": "Netflix Inc.",
        "INTC": "Intel Corporation",
        "AMD": "Advanced Micro Devices"
    },
    "🏦 US Financial Companies": {
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America",
        "WFC": "Wells Fargo & Co.",
        "C": "Citigroup Inc.",
        "GS": "Goldman Sachs Group Inc.",
        "MS": "Morgan Stanley",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc."
    },
    "🛒 US Retail & Consumer": {
        "WMT": "Walmart Inc.",
        "TGT": "Target Corporation",
        "COST": "Costco Wholesale",
        "HD": "Home Depot Inc.",
        "MCD": "McDonald's Corporation",
        "NKE": "Nike Inc.",
        "SBUX": "Starbucks Corporation"
    },
    "💊 US Healthcare": {
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "UNH": "UnitedHealth Group",
        "MRK": "Merck & Co.",
        "ABBV": "AbbVie Inc.",
        "LLY": "Eli Lilly and Company"
    },
    "⛽ US Energy": {
        "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation",
        "COP": "ConocoPhillips",
        "EOG": "EOG Resources Inc.",
        "SLB": "Schlumberger Limited"
    },
    "✈️ US Transportation": {
        "UAL": "United Airlines",
        "DAL": "Delta Air Lines",
        "AAL": "American Airlines",
        "LUV": "Southwest Airlines",
        "UPS": "United Parcel Service",
        "FDX": "FedEx Corporation"
    },
    "🏭 US Industrial": {
        "BA": "Boeing Company",
        "CAT": "Caterpillar Inc.",
        "GE": "General Electric",
        "HON": "Honeywell International",
        "MMM": "3M Company",
        "LMT": "Lockheed Martin"
    },
    "📡 US Telecommunications": {
        "T": "AT&T Inc.",
        "VZ": "Verizon Communications",
        "TMUS": "T-Mobile US Inc.",
        "CMCSA": "Comcast Corporation"
    },
    "🌍 International Stocks": {
        "BABA": "Alibaba Group",
        "TCS.NS": "Tata Consultancy Services",
        "RELIANCE.NS": "Reliance Industries",
        "SAP": "SAP SE",
        "ASML": "ASML Holding",
        "TSM": "Taiwan Semiconductor"
    },
    "💰 Cryptocurrency Related": {
        "COIN": "Coinbase Global Inc.",
        "MSTR": "MicroStrategy Inc.",
        "RIOT": "Riot Platforms Inc.",
        "MARA": "Marathon Digital Holdings"
    },
    "📈 ETFs & Funds": {
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market",
        "VOO": "Vanguard S&P 500 ETF",
        "BND": "Vanguard Total Bond Market",
        "GLD": "SPDR Gold Trust"
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