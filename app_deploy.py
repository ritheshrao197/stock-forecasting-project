"""
Stock Market Price Forecasting - Interactive Web UI
Unified Version - Supports Both Indian & International Stocks
"""

# ============================================
# SUPPRESS UNWANTED LOGS AND WARNINGS
# ============================================

import os
import sys
import warnings

# Suppress TensorFlow logs (set before importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=warning, 2=error, 3=none
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization message

# Suppress general warnings
warnings.filterwarnings('ignore')

# Suppress other logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ============================================
# IMPORTS
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import random

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Stock Market Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Status Box Styles */
    .status-box {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
        border-left: 4px solid #9e9e9e;
        background-color: #f5f5f5;
        color: #333;
    }
    .status-box .status-icon {
        font-size: 1.3rem;
    }
    .status-box.status-idle {
        border-left-color: #9e9e9e;
        background-color: #f5f5f5;
        color: #555;
    }
    .status-box.status-processing {
        border-left-color: #ff6b35;
        background-color: #fff3e0;
        color: #e65100;
        animation: pulse 1.5s ease-in-out infinite;
    }
    .status-box.status-complete {
        border-left-color: #4caf50;
        background-color: #e8f5e9;
        color: #1b5e20;
    }
    .status-box.status-error {
        border-left-color: #f44336;
        background-color: #ffebee;
        color: #b71c1c;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Loading spinner animation */
    .spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 107, 53, 0.3);
        border-radius: 50%;
        border-top-color: #ff6b35;
        animation: spin 0.8s ease-in-out infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Processing dots animation */
    .processing-dots::after {
        content: '';
        animation: dots 1.5s steps(4, end) infinite;
    }
    @keyframes dots {
        0% { content: ''; }
        25% { content: '.'; }
        50% { content: '..'; }
        75% { content: '...'; }
        100% { content: ''; }
    }
    
    /* Progress bar styling */
    .progress-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
        height: 24px;
        position: relative;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
        width: 0%;
    }
    .progress-text {
        position: absolute;
        width: 100%;
        text-align: center;
        color: #333;
        font-size: 0.75rem;
        font-weight: bold;
        line-height: 24px;
    }
    
    /* Metric card */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #145a8a;
        color: white;
    }
    .stButton button:disabled {
        background-color: #ccc;
        color: #666;
    }
    
    .india-badge {
        background-color: #ff6b35;
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 8px;
    }
    .us-badge {
        background-color: #1f77b4;
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 8px;
    }
    
    section[data-testid="stSidebar"] {
        height: 100vh !important;
        overflow-y: auto !important;
    }
    section[data-testid="stSidebar"] > div {
        height: 100% !important;
        overflow-y: auto !important;
        padding-bottom: 1rem !important;
    }
    .stSubheader {
        margin-top: 0.3rem !important;
        margin-bottom: 0.1rem !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    .stTextInput {
        margin-bottom: 0.1rem !important;
    }
    .stSelectbox {
        margin-top: 0.1rem !important;
        margin-bottom: 0.2rem !important;
    }
    .stButton button {
        padding: 0.3rem 0.5rem !important;
        font-size: 0.85rem !important;
    }
    .stCheckbox {
        margin-bottom: 0.1rem !important;
    }
    .stSlider {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    .category-label {
        font-size: 0.7rem;
        color: #888;
        margin-top: -0.2rem;
        margin-bottom: 0.2rem;
        font-style: italic;
    }
    footer {
        display: none !important;
    }
    .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar {
        width: 4px;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar-track {
        background: transparent;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 2px;
    }
    hr {
        margin: 0.3rem 0 !important;
    }
    .log-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.85rem;
        height: 250px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
        border: 1px solid #333;
    }
    .log-container::-webkit-scrollbar {
        width: 8px;
    }
    .log-container::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    .log-container::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 4px;
    }
    .log-container::-webkit-scrollbar-thumb:hover {
        background: #777;
    }
    .log-info { color: #4fc3f7; }
    .log-success { color: #81c784; }
    .log-warning { color: #ffb74d; }
    .log-error { color: #e57373; }
    .log-header { color: #ce93d8; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================
# TRY IMPORT MODELS WITH ERROR HANDLING
# ============================================

try:
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# ============================================
# TICKER DATABASE - UNIFIED
# ============================================

INDIAN_TICKERS = {
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
    }
}

INTERNATIONAL_TICKERS = {
    "📊 US Indices": {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "NASDAQ Composite",
        "^RUT": "Russell 2000",
        "^VIX": "CBOE Volatility Index"
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
    "📈 ETFs & Funds": {
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market",
        "VOO": "Vanguard S&P 500 ETF",
        "BND": "Vanguard Total Bond Market",
        "GLD": "SPDR Gold Trust"
    }
}

def get_ticker_options(market_type="India"):
    """Get ticker options based on market type"""
    options = []
    
    if market_type == "India":
        categories = INDIAN_TICKERS
    else:
        categories = INTERNATIONAL_TICKERS
    
    for category, tickers in categories.items():
        for ticker, name in tickers.items():
            options.append({
                "label": f"{name} ({ticker})",
                "value": ticker,
                "category": category,
                "market": market_type
            })
    
    return sorted(options, key=lambda x: x["label"])

def get_exchange_info(ticker):
    """Get exchange information for a ticker"""
    if ticker.endswith('.NS'):
        return "🇮🇳 NSE (National Stock Exchange)"
    elif ticker.endswith('.BO'):
        return "🇮🇳 BSE (Bombay Stock Exchange)"
    elif ticker.startswith('^NSE') or ticker.startswith('^BSESN'):
        return "🇮🇳 Indian Index"
    else:
        return "🌍 International Stock"

def get_currency_symbol(ticker):
    """Get currency symbol based on ticker"""
    if ticker.endswith('.NS') or ticker.endswith('.BO') or ticker.startswith('^NSE') or ticker.startswith('^BSESN'):
        return "₹"
    else:
        return "$"

# ============================================
# STATUS NOTIFICATION FUNCTION WITH PROGRESS
# ============================================

def render_status_notification():
    """Render animated status notification with progress"""
    
    if st.session_state.forecast_running:
        # Get current process name
        process_name = st.session_state.get('current_process', 'Training models')
        progress = st.session_state.get('progress', 0)
        
        # Processing status with spinner animation and progress bar
        st.markdown(f"""
        <div class="status-box status-processing">
            <span class="status-icon">⏳</span>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>
                        <strong>Processing</strong> 
                        <span class="processing-dots">{process_name}</span>
                    </span>
                    <span style="display: inline-block; margin-left: 10px;">
                        <span class="spinner"></span>
                    </span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {progress}%;">
                        {int(progress)}%
                    </div>
                    <div class="progress-text">{int(progress)}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    elif st.session_state.models_trained and st.session_state.forecast_completed:
        st.markdown("""
        <div class="status-box status-complete">
            <span class="status-icon">✅</span>
            <span>
                <strong>Forecast Complete!</strong> 
                Results are ready to view.
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    elif st.session_state.get('forecast_error', False):
        st.markdown("""
        <div class="status-box status-error">
            <span class="status-icon">❌</span>
            <span>
                <strong>Forecast Failed</strong> 
                Please check the logs for details.
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="status-box status-idle">
            <span class="status-icon">💡</span>
            <span>
                <strong>Ready</strong> 
                Configure settings and click 'Run Forecast' to start analysis.
            </span>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# CACHING FOR RATE LIMITING
# ============================================

@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def fetch_stock_data_with_cache(ticker, start_date, end_date):
    """Fetch stock data with caching to prevent rate limiting"""
    try:
        time.sleep(random.uniform(0.5, 1.5))
        
        stock = yf.Ticker(ticker)
        data = stock.history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True
        )
        
        if data.empty:
            time.sleep(random.uniform(0.5, 1.0))
            data = stock.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False
            )
        
        return data
    except Exception as e:
        if "Too Many Requests" in str(e) or "Rate limited" in str(e):
            time.sleep(random.uniform(5, 10))
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True
            )
            return data
        raise e

# ============================================
# SESSION STATE
# ============================================

if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'forecast_running' not in st.session_state:
    st.session_state.forecast_running = False
if 'forecast_completed' not in st.session_state:
    st.session_state.forecast_completed = False
if 'forecast_error' not in st.session_state:
    st.session_state.forecast_error = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'rate_limit_retry_count' not in st.session_state:
    st.session_state.rate_limit_retry_count = 0
if 'market_type' not in st.session_state:
    st.session_state.market_type = "India"
if 'current_process' not in st.session_state:
    st.session_state.current_process = "Initializing..."
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    # Header
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 0.5rem;">
        <span style="font-size: 2rem;">📈</span>
        <span style="font-size: 1.1rem; font-weight: bold;">Configuration</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Market Selection Toggle
    st.subheader("🌍 Market Selection")
    
    market_col1, market_col2 = st.columns(2)
    with market_col1:
        if st.button("🇮🇳 India", use_container_width=True, type="primary" if st.session_state.market_type == "India" else "secondary"):
            st.session_state.market_type = "India"
            st.session_state.data = None
            st.session_state.predictions = {}
            st.session_state.results = None
            st.session_state.models_trained = False
            st.session_state.forecast_completed = False
            st.session_state.forecast_error = False
            st.rerun()
    with market_col2:
        if st.button("🌍 International", use_container_width=True, type="primary" if st.session_state.market_type == "International" else "secondary"):
            st.session_state.market_type = "International"
            st.session_state.data = None
            st.session_state.predictions = {}
            st.session_state.results = None
            st.session_state.models_trained = False
            st.session_state.forecast_completed = False
            st.session_state.forecast_error = False
            st.rerun()
    
    # Show current market badge
    if st.session_state.market_type == "India":
        st.markdown('<div style="text-align: center;"><span class="india-badge">🇮🇳 INDIAN STOCKS</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center;"><span class="us-badge">🌍 INTERNATIONAL STOCKS</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stock Selection
    st.subheader("📈 Stock Selection")
    
    # Get ticker options based on market type
    all_options = get_ticker_options(st.session_state.market_type)
    
    search_term = st.text_input("🔍 Search", placeholder="Type to search...", key="ticker_search")
    
    if search_term:
        filtered = [o for o in all_options if search_term.lower() in o["label"].lower()]
    else:
        filtered = all_options
    
    if not filtered:
        filtered = all_options
    
    ticker_options = [o["label"] for o in filtered]
    ticker_values = [o["value"] for o in filtered]
    
    # Set default based on market type
    default_ticker = "RELIANCE.NS" if st.session_state.market_type == "India" else "AAPL"
    default_index = 0
    for i, opt in enumerate(filtered):
        if opt["value"] == default_ticker:
            default_index = i
            break
    
    selected_label = st.selectbox(
        "Select Ticker",
        options=ticker_options,
        index=min(default_index, len(ticker_options) - 1),
        key="ticker_select"
    )
    
    selected_index = ticker_options.index(selected_label) if selected_label in ticker_options else 0
    ticker = ticker_values[selected_index] if ticker_values else default_ticker
    
    # Show category
    for opt in all_options:
        if opt["value"] == ticker:
            st.caption(f"📌 {opt['category']}")
            break
    
    exchange_info = get_exchange_info(ticker)
    st.caption(f"🏛️ {exchange_info}")
    
    # Date Range
    st.subheader("📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime(2020, 1, 1), key="start_date")
    with col2:
        end_date = st.date_input("End", datetime.now(), key="end_date")
    
    # Quick presets
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("1Y", key="preset_1y", use_container_width=True):
            start_date = datetime.now() - timedelta(days=365)
    with preset_col2:
        if st.button("2Y", key="preset_2y", use_container_width=True):
            start_date = datetime.now() - timedelta(days=730)
    with preset_col3:
        if st.button("5Y", key="preset_5y", use_container_width=True):
            start_date = datetime.now() - timedelta(days=1825)
    
    # Models
    st.subheader("🤖 Models")
    use_arima = st.checkbox("ARIMA", value=ARIMA_AVAILABLE, disabled=not ARIMA_AVAILABLE, key="use_arima")
    use_prophet = st.checkbox("Prophet", value=PROPHET_AVAILABLE, disabled=not PROPHET_AVAILABLE, key="use_prophet")
    use_lstm = st.checkbox("LSTM", value=LSTM_AVAILABLE, disabled=not LSTM_AVAILABLE, key="use_lstm")
    
    if not ARIMA_AVAILABLE:
        st.caption("⚠️ ARIMA not available")
    if not PROPHET_AVAILABLE:
        st.caption("⚠️ Prophet not available")
    if not LSTM_AVAILABLE:
        st.caption("⚠️ LSTM requires TensorFlow")
    
    # Parameters
    st.subheader("⚙️ Parameters")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="test_size")
    
    with st.expander("🧠 LSTM Parameters"):
        lookback = st.number_input("Lookback", min_value=10, max_value=120, value=30, key="lookback")  # Reduced default
        lstm_epochs = st.number_input("Epochs", min_value=5, max_value=50, value=10, key="lstm_epochs")  # Reduced default
    
    # Fast Mode Option
    st.subheader("⚡ Performance")
    fast_mode = st.checkbox("Fast Mode (faster ARIMA)", value=True, key="fast_mode")
    st.caption("Uses fixed ARIMA(2,1,2) parameters for speed")
    
    # Rate limit warning
    if st.session_state.rate_limit_retry_count > 2:
        st.warning("⚠️ Multiple rate limit errors. Please wait a moment.")
    
    # Run button
    st.markdown("---")
    run_button = st.button(
        "🚀 Run Forecast",
        use_container_width=True,
        disabled=st.session_state.forecast_running
    )

# ============================================
# MAIN CONTENT
# ============================================

# Dynamic header based on market type
header_text = "🇮🇳 Indian Stock Market Price Forecasting" if st.session_state.market_type == "India" else "🌍 International Stock Market Price Forecasting"
st.markdown(f'<h1 class="main-header">{header_text}</h1>', unsafe_allow_html=True)
st.markdown("*Interactive dashboard for ARIMA, Prophet, and LSTM models*")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Overview",
    "📈 Model Results",
    "📉 Predictions",
    "📋 Model Comparison",
    "📝 Live Logs",
    "ℹ️ About"
])

# ============================================
# TAB 1: Data Overview
# ============================================

with tab1:
    render_status_notification()
    
    if st.session_state.data is not None:
        data = st.session_state.data
        currency = get_currency_symbol(ticker)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(data))
        with col2:
            st.metric("Start Date", data.index[0].strftime("%Y-%m-%d"))
        with col3:
            st.metric("End Date", data.index[-1].strftime("%Y-%m-%d"))
        with col4:
            st.metric("Price Range", f"{currency}{data['Close'].min():.2f} - {currency}{data['Close'].max():.2f}")
        
        st.subheader("📊 Price History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1, dash='dash')
            ))
        if 'SMA_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='green', width=1, dash='dash')
            ))
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View Raw Data"):
            st.dataframe(data.tail(20), use_container_width=True)
    else:
        st.info("👈 Configure settings and click 'Run Forecast'")

# ============================================
# TAB 2: Model Results
# ============================================

with tab2:
    render_status_notification()
    
    if st.session_state.models_trained and st.session_state.results is not None:
        results = st.session_state.results
        currency = get_currency_symbol(ticker)
        
        st.subheader("📊 Model Performance Comparison")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(results, use_container_width=True)
        
        with col2:
            best_model = results.loc[results['RMSE'].idxmin()]
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏆 Best Model</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{best_model['Model']}</p>
                <p>RMSE: {currency}{best_model['RMSE']:.2f}</p>
                <p>MAPE: {best_model['MAPE']:.2f}%</p>
                <p>R²: {best_model['R²']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(results['Model'], results['RMSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('RMSE Comparison')
            ax.set_ylabel(f'RMSE ({currency})')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, results['RMSE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{currency}{value:.2f}', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(results['Model'], results['MAPE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('MAPE Comparison')
            ax.set_ylabel('MAPE (%)')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, results['MAPE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                       f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
    else:
        if not st.session_state.forecast_running:
            st.info("👈 Run the forecast to see model results.")

# ============================================
# TAB 3: Predictions
# ============================================

with tab3:
    render_status_notification()
    
    if st.session_state.models_trained and st.session_state.predictions:
        data = st.session_state.data
        train_size = int(len(data) * (1 - test_size))
        test_data = data['Close'][train_size:]
        currency = get_currency_symbol(ticker)
        
        st.subheader("📈 Predictions vs Actual")
        
        selected_models = st.multiselect(
            "Select models to display",
            options=list(st.session_state.predictions.keys()),
            default=list(st.session_state.predictions.keys())
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (name, pred) in enumerate(st.session_state.predictions.items()):
            if name in selected_models and len(pred) > 0:
                fig.add_trace(go.Scatter(
                    x=test_data.index[:len(pred)],
                    y=pred[:len(test_data)],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f'Stock Price Forecast - {ticker}',
            xaxis_title='Date',
            yaxis_title=f'Price ({currency})',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔍 Zoomed View")
        zoom = st.slider("Days to zoom", 10, 100, 50)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=test_data.index[-zoom:],
            y=test_data.values[-zoom:],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        for i, (name, pred) in enumerate(st.session_state.predictions.items()):
            if name in selected_models and len(pred) >= zoom:
                fig2.add_trace(go.Scatter(
                    x=test_data.index[-zoom:],
                    y=pred[-zoom:],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig2.update_layout(
            title=f'Zoomed View (Last {zoom} Days)',
            xaxis_title='Date',
            yaxis_title=f'Price ({currency})',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        if not st.session_state.forecast_running:
            st.info("👈 Run the forecast to see predictions.")

# ============================================
# TAB 4: Model Comparison
# ============================================

with tab4:
    render_status_notification()
    
    if st.session_state.models_trained and st.session_state.results is not None:
        results = st.session_state.results
        st.subheader("📋 Detailed Model Comparison")
        st.dataframe(results, use_container_width=True)
        
        st.subheader("💡 Recommendations")
        best_rmse = results.loc[results['RMSE'].idxmin()]
        currency = get_currency_symbol(ticker)
        st.info(f"""
        **📌 Best Model**
        - Model: **{best_rmse['Model']}**
        - RMSE: {currency}{best_rmse['RMSE']:.2f}
        - MAPE: {best_rmse['MAPE']:.2f}%
        - R²: {best_rmse['R²']:.4f}
        """)
    else:
        if not st.session_state.forecast_running:
            st.info("👈 Run the forecast to see model comparison.")

# ============================================
# TAB 5: Live Logs
# ============================================

with tab5:
    st.subheader("📝 Live Terminal Logs")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.logs = []
            st.rerun()
    with col3:
        st.caption(f"📊 {len(st.session_state.logs)} log entries")
    
    if st.session_state.logs:
        log_html = ""
        for log in st.session_state.logs[-50:]:
            color = {
                "info": "#4fc3f7",
                "success": "#81c784",
                "warning": "#ffb74d",
                "error": "#e57373",
                "header": "#ce93d8"
            }.get(log.get("type", "info"), "#4fc3f7")
            
            log_html += f'<div><span style="color: #888;">[{log["timestamp"]}]</span> <span style="color: {color};">{log["message"]}</span></div>'
        
        st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.info("📝 No logs yet. Run the forecast to see terminal output.")
    
    if st.session_state.forecast_running:
        st.warning("⏳ Process is currently running...")
    else:
        st.success("✅ Process idle")

# ============================================
# TAB 6: About
# ============================================

with tab6:
    st.markdown("""
    ### 📈 Stock Market Price Forecasting
    
    This interactive dashboard compares three forecasting models for both **Indian and International stocks**:
    
    #### 🤖 Models Implemented
    
    1. **ARIMA** - Classical statistical model for linear trends
    2. **Prophet** - Facebook's model for seasonal data
    3. **LSTM** - Deep learning model for complex patterns
    
    #### 🌍 Supported Markets
    
    **🇮🇳 Indian Stocks:**
    - NIFTY 50, SENSEX, and other indices
    - 100+ Indian companies across sectors
    - NSE (.NS) and BSE (.BO) support
    
    **🌍 International Stocks:**
    - S&P 500, NASDAQ, Dow Jones
    - US Tech, Financial, Healthcare, Retail companies
    - ETFs and Funds
    
    #### 📝 How to Use
    
    1. Select market type: 🇮🇳 India or 🌍 International
    2. Search and select a ticker
    3. Choose date range
    4. Select models
    5. Click "Run Forecast"
    6. Explore results in tabs!
    """)

# ============================================
# BACKEND LOGIC
# ============================================

def add_log(message, log_type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    })

def update_progress(process_name, progress):
    """Update the current process and progress"""
    st.session_state.current_process = process_name
    st.session_state.progress = progress
    add_log(f"{process_name} - {int(progress)}% complete", "info")

def run_forecast():
    """Execute the forecast with rate limiting and progress tracking"""
    st.session_state.forecast_running = True
    st.session_state.forecast_completed = False
    st.session_state.forecast_error = False
    st.session_state.logs = []
    st.session_state.progress = 0
    st.session_state.current_process = "Initializing..."
    
    market_name = "Indian" if st.session_state.market_type == "India" else "International"
    add_log(f"🚀 Starting {market_name} forecast...", "header")
    add_log(f"📊 Ticker: {ticker}")
    update_progress("Initializing", 0)
    
    start_time = time.time()
    max_time = 300  # 5 minutes max
    
    try:
        # Load data with caching
        add_log("📊 Loading stock data...", "info")
        update_progress("Loading stock data", 5)
        
        data = fetch_stock_data_with_cache(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if data.empty:
            add_log("❌ No data found", "error")
            st.error(f"No data found for {ticker}")
            st.session_state.forecast_running = False
            st.session_state.forecast_error = True
            return
        
        # Add indicators
        add_log("📊 Adding technical indicators...", "info")
        update_progress("Adding technical indicators", 10)
        
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data = data.dropna()
        
        st.session_state.data = data
        add_log(f"✅ Data loaded: {len(data)} rows", "success")
        update_progress("Data preparation complete", 15)
        
        # Split data
        train_size = int(len(data) * (1 - test_size))
        train = data['Close'][:train_size]
        test = data['Close'][train_size:]
        
        add_log(f"📊 Train: {len(train)} days, Test: {len(test)} days", "info")
        update_progress("Data split complete", 20)
        
        st.session_state.predictions = {}
        results = []
        
        # Total steps for progress
        total_steps = sum([use_arima, use_prophet, use_lstm])
        if total_steps == 0:
            total_steps = 1
        current_step = 0
        
        # ============================================
        # ARIMA (FIXED - Removed 'disp' parameter)
        # ============================================
        if use_arima and ARIMA_AVAILABLE:
            current_step += 1
            progress_val = 20 + (current_step / total_steps * 60)
            add_log("🤖 Training ARIMA...", "header")
            update_progress("Training ARIMA model", progress_val)
            try:
                if fast_mode:
                    # Fast mode: fixed parameters
                    add_log("⚡ Fast mode - using default ARIMA(2,1,2)", "info")
                    p, d, q = 2, 1, 2
                    arima_model = ARIMA(train, order=(p, d, q))
                    arima_fitted = arima_model.fit(method_kwargs={'maxiter': 50})  # Removed disp parameter
                else:
                    # Regular mode: auto_arima with limits
                    auto_model = auto_arima(
                        train,
                        start_p=0, max_p=3,
                        start_q=0, max_q=3,
                        max_d=2,
                        seasonal=False,
                        stepwise=True,
                        trace=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        n_fits=10,
                        information_criterion='aic',
                        n_jobs=1,
                        method='lbfgs'
                    )
                    arima_model = ARIMA(train, order=auto_model.order)
                    arima_fitted = arima_model.fit(method_kwargs={'maxiter': 100})  # Removed disp parameter
                
                arima_pred = arima_fitted.forecast(steps=len(test))
                st.session_state.predictions['ARIMA'] = arima_pred.values if hasattr(arima_pred, 'values') else arima_pred
                
                rmse = np.sqrt(np.mean((test.values - arima_pred[:len(test)])**2))
                mae = mean_absolute_error(test.values, arima_pred[:len(test)])
                mape = np.mean(np.abs((test.values - arima_pred[:len(test)]) / test.values)) * 100
                ss_res = np.sum((test.values - arima_pred[:len(test)])**2)
                ss_tot = np.sum((test.values - np.mean(test.values))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                results.append({'Model': 'ARIMA', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                add_log(f"✅ ARIMA complete! RMSE: {get_currency_symbol(ticker)}{rmse:.2f}", "success")
                update_progress("ARIMA training complete", min(progress_val + 10, 80))
            except Exception as e:
                add_log(f"❌ ARIMA failed: {str(e)[:100]}", "error")
        
        # ============================================
        # PROPHET (Fixed Timezone)
        # ============================================
        if use_prophet and PROPHET_AVAILABLE:
            current_step += 1
            progress_val = 20 + (current_step / total_steps * 60)
            add_log("🤖 Training Prophet...", "header")
            update_progress("Training Prophet model", progress_val)
            try:
                # Set environment for Prophet
                os.environ['STAN_BACKEND'] = 'CMDSTANPY'
                
                prophet_data = data.reset_index()[['Date', 'Close']]
                prophet_data.columns = ['ds', 'y']
                
                # FIX: Remove timezone
                if pd.api.types.is_datetime64_any_dtype(prophet_data['ds']):
                    prophet_data['ds'] = pd.to_datetime(prophet_data['ds']).dt.tz_localize(None)
                
                train_prophet = prophet_data[:train_size]
                train_prophet['y'] = train_prophet['y'].astype(float)
                
                prophet_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    holidays_prior_scale=10.0
                )
                
                # Fit with quiet mode
                prophet_model.fit(train_prophet, quiet=True)
                
                future = prophet_model.make_future_dataframe(periods=len(test))
                forecast = prophet_model.predict(future)
                prophet_pred = forecast['yhat'].values[-len(test):]
                prophet_pred = np.nan_to_num(prophet_pred, nan=test.mean())
                
                st.session_state.predictions['Prophet'] = prophet_pred
                
                rmse = np.sqrt(np.mean((test.values - prophet_pred)**2))
                mae = mean_absolute_error(test.values, prophet_pred)
                mape = np.mean(np.abs((test.values - prophet_pred) / test.values)) * 100
                ss_res = np.sum((test.values - prophet_pred)**2)
                ss_tot = np.sum((test.values - np.mean(test.values))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                results.append({'Model': 'Prophet', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                add_log(f"✅ Prophet complete! RMSE: {get_currency_symbol(ticker)}{rmse:.2f}", "success")
                update_progress("Prophet training complete", min(progress_val + 10, 80))
            except Exception as e:
                add_log(f"❌ Prophet failed: {str(e)[:100]}", "error")
        
        # ============================================
        # LSTM (Fixed - Won't Get Stuck)
        # ============================================
        if use_lstm and LSTM_AVAILABLE:
            current_step += 1
            progress_val = 20 + (current_step / total_steps * 60)
            add_log("🤖 Training LSTM...", "header")
            update_progress("Training LSTM model", progress_val)
            try:
                actual_lookback = min(lookback, 30)  # Reduced max
                actual_epochs = min(lstm_epochs, 10)  # Reduced max
                
                add_log(f"📊 Using lookback: {actual_lookback}, epochs: {actual_epochs}", "info")
                
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data[['Close']])
                
                X, y = [], []
                for i in range(actual_lookback, len(scaled_data)):
                    X.append(scaled_data[i-actual_lookback:i])
                    y.append(scaled_data[i, 0])
                
                X = np.array(X)
                y = np.array(y)
                
                if len(X) == 0:
                    add_log("❌ Not enough data for LSTM", "error")
                    raise ValueError("Not enough data")
                
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Simpler model
                model = Sequential([
                    LSTM(32, return_sequences=True, input_shape=(actual_lookback, 1)),
                    Dropout(0.2),
                    LSTM(16, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                
                early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
                
                for epoch in range(actual_epochs):
                    # Check timeout
                    if time.time() - start_time > max_time:
                        add_log("⏰ Timeout - stopping LSTM training", "warning")
                        break
                    
                    history = model.fit(
                        X_train, y_train, 
                        epochs=1, 
                        batch_size=32, 
                        verbose=0,
                        callbacks=[early_stop]
                    )
                    epoch_progress = progress_val + ((epoch + 1) / actual_epochs * 30)
                    update_progress(f"Training LSTM - Epoch {epoch+1}/{actual_epochs}", min(epoch_progress, 85))
                    
                    if len(history.history['loss']) > 0 and history.history['loss'][0] < 0.001:
                        add_log(f"✅ Early stopping at epoch {epoch+1}", "success")
                        break
                
                lstm_pred_scaled = model.predict(X_test, verbose=0)
                lstm_pred = scaler.inverse_transform(
                    np.concatenate([lstm_pred_scaled, np.zeros((len(lstm_pred_scaled), 1))], axis=1)
                )[:, 0]
                
                if len(lstm_pred) < len(test):
                    lstm_pred = np.pad(lstm_pred, (0, len(test) - len(lstm_pred)), mode='edge')
                st.session_state.predictions['LSTM'] = lstm_pred
                
                rmse = np.sqrt(np.mean((test.values - lstm_pred[:len(test)])**2))
                mae = mean_absolute_error(test.values, lstm_pred[:len(test)])
                mape = np.mean(np.abs((test.values - lstm_pred[:len(test)]) / test.values)) * 100
                ss_res = np.sum((test.values - lstm_pred[:len(test)])**2)
                ss_tot = np.sum((test.values - np.mean(test.values))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                results.append({'Model': 'LSTM', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                add_log(f"✅ LSTM complete! RMSE: {get_currency_symbol(ticker)}{rmse:.2f}", "success")
                update_progress("LSTM training complete", 90)
            except Exception as e:
                add_log(f"❌ LSTM failed: {str(e)[:100]}", "error")
        
        # ============================================
        # Complete
        # ============================================
        update_progress("Finalizing results", 95)
        
        st.session_state.results = pd.DataFrame(results)
        st.session_state.models_trained = True
        st.session_state.forecast_completed = True
        
        add_log(f"✅ Forecast complete! {len(results)} models trained.", "success")
        update_progress("Forecast complete!", 100)
        
        if len(results) > 0:
            best = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()]
            add_log(f"🏆 Best: {best['Model']} (RMSE: {get_currency_symbol(ticker)}{best['RMSE']:.2f})", "success")
            st.success(f"✅ Forecast complete! Best: **{best['Model']}**")
            st.balloons()
        
    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate limited" in error_msg:
            add_log("⚠️ Rate limit exceeded. Please wait.", "warning")
            st.session_state.rate_limit_retry_count += 1
            st.warning("⏳ Rate limit exceeded. Please wait a few minutes.")
        else:
            add_log(f"❌ Error: {error_msg}", "error")
            st.error(f"Error: {error_msg}")
        st.session_state.forecast_error = True
        update_progress("Error occurred", 0)
    
    st.session_state.forecast_running = False

# ============================================
# TRIGGER FORECAST
# ============================================

if run_button and not st.session_state.forecast_running:
    run_forecast()
    st.rerun()

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Built with ❤️ using Streamlit | 🇮🇳 Indian & 🌍 International Stock Market Price Forecasting
</div>
""", unsafe_allow_html=True)