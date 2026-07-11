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
DEFAULT_TICKER = "RELIANCE.NS"  # Changed to Indian stock
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_LOOKBACK = 60
DEFAULT_EPOCHS = 20

# Indian Stock Categories
TICKER_CATEGORIES = {
    "📊 Indian Indices": {
        "^NSEI": "NIFTY 50",
        "^BSESN": "SENSEX",
        "^NSEBANK": "NIFTY BANK",
        "^CNXIT": "NIFTY IT",
        "^CNXPHARMA": "NIFTY PHARMA",
        "^CNXFMCG": "NIFTY FMCG",
        "^CNXAUTO": "NIFTY AUTO"
    },
    "🏢 Indian IT & Technology": {
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys Ltd",
        "WIPRO.NS": "Wipro Ltd",
        "HCLTECH.NS": "HCL Technologies",
        "TECHM.NS": "Tech Mahindra",
        "LTIM.NS": "LTIMindtree",
        "MPHASIS.NS": "Mphasis Ltd",
        "COFORGE.NS": "Coforge Ltd",
        "PERSISTENT.NS": "Persistent Systems"
    },
    "🏦 Indian Banking & Finance": {
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "SBIN.NS": "State Bank of India",
        "KOTAKBANK.NS": "Kotak Mahindra Bank",
        "AXISBANK.NS": "Axis Bank",
        "INDUSINDBK.NS": "IndusInd Bank",
        "YESBANK.NS": "Yes Bank",
        "BANKBARODA.NS": "Bank of Baroda",
        "PNB.NS": "Punjab National Bank",
        "FEDERALBNK.NS": "Federal Bank"
    },
    "🛢️ Indian Energy & Oil": {
        "RELIANCE.NS": "Reliance Industries",
        "ONGC.NS": "Oil & Natural Gas Corp",
        "IOCL.NS": "Indian Oil Corporation",
        "BPCL.NS": "Bharat Petroleum",
        "HPCL.NS": "Hindustan Petroleum",
        "GAIL.NS": "GAIL (India) Ltd",
        "POWERGRID.NS": "Power Grid Corporation",
        "NTPC.NS": "NTPC Ltd",
        "ADANIGREEN.NS": "Adani Green Energy",
        "ADANIPOWER.NS": "Adani Power"
    },
    "🏭 Indian Manufacturing & Auto": {
        "TATAMOTORS.NS": "Tata Motors",
        "MARUTI.NS": "Maruti Suzuki India",
        "M&M.NS": "Mahindra & Mahindra",
        "BAJAJ-AUTO.NS": "Bajaj Auto",
        "HEROMOTOCO.NS": "Hero MotoCorp",
        "TATASTEEL.NS": "Tata Steel",
        "JSWSTEEL.NS": "JSW Steel",
        "HINDALCO.NS": "Hindalco Industries",
        "ASIANPAINT.NS": "Asian Paints",
        "ULTRACEMCO.NS": "UltraTech Cement"
    },
    "🏥 Indian Healthcare & Pharma": {
        "SUNPHARMA.NS": "Sun Pharmaceutical",
        "DRREDDY.NS": "Dr. Reddy's Laboratories",
        "CIPLA.NS": "Cipla Ltd",
        "DIVISLAB.NS": "Divi's Laboratories",
        "BIOCON.NS": "Biocon Ltd",
        "APOLLOHOSP.NS": "Apollo Hospitals",
        "FORTIS.NS": "Fortis Healthcare"
    },
    "🛒 Indian FMCG & Consumer": {
        "ITC.NS": "ITC Ltd",
        "HINDUNILVR.NS": "Hindustan Unilever",
        "NESTLEIND.NS": "Nestle India",
        "BRITANNIA.NS": "Britannia Industries",
        "DABUR.NS": "Dabur India",
        "MARICO.NS": "Marico Ltd",
        "TITAN.NS": "Titan Company",
        "TRENT.NS": "Trent Ltd"
    },
    "📡 Indian Telecom & Media": {
        "BHARTIARTL.NS": "Bharti Airtel",
        "JIOFIN.NS": "Jio Financial Services",
        "IDEA.NS": "Vodafone Idea",
        "SUNTV.NS": "Sun TV Network",
        "PVRINOX.NS": "PVR INOX"
    },
    "🏗️ Indian Infrastructure & Real Estate": {
        "L&T.NS": "Larsen & Toubro",
        "ADANIPORTS.NS": "Adani Ports",
        "NCC.NS": "NCC Ltd",
        "DLF.NS": "DLF Ltd",
        "GODREJCP.NS": "Godrej Consumer"
    },
    "🌍 International Stocks": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation"
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