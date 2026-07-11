"""
Stock Market Price Forecasting - Single File Version for Streamlit Cloud
With Rate Limiting Protection
"""

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
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Stock Market Forecasting - Indian Stocks",
    page_icon="🇮🇳",
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
        height: 350px;
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
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# ============================================
# TICKER DATABASE - INDIAN STOCKS
# ============================================

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
    options = []
    for category, tickers in TICKER_CATEGORIES.items():
        for ticker, name in tickers.items():
            options.append({
                "label": f"{name} ({ticker})",
                "value": ticker,
                "category": category
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

# ============================================
# CACHING FOR RATE LIMITING
# ============================================

@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def fetch_stock_data_with_cache(ticker, start_date, end_date):
    """Fetch stock data with caching to prevent rate limiting"""
    try:
        # Add a small random delay to prevent hitting rate limits
        time.sleep(random.uniform(0.5, 1.5))
        
        stock = yf.Ticker(ticker)
        data = stock.history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True
        )
        
        if data.empty:
            # Try without auto_adjust
            time.sleep(random.uniform(0.5, 1.0))
            data = stock.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False
            )
        
        return data
    except Exception as e:
        # If rate limited, wait longer and retry
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
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'rate_limit_retry_count' not in st.session_state:
    st.session_state.rate_limit_retry_count = 0

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 0.5rem;">
        <span style="font-size: 2rem;">📈</span>
        <span style="font-size: 1.1rem; font-weight: bold;">Configuration</span>
        <span class="india-badge">🇮🇳 INDIA</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock Selection
    st.subheader("📈 Stock Selection")
    
    search_term = st.text_input("🔍 Search", placeholder="Type to search...", key="ticker_search")
    
    all_options = get_ticker_options()
    
    if search_term:
        filtered = [o for o in all_options if search_term.lower() in o["label"].lower()]
    else:
        filtered = all_options
    
    if not filtered:
        filtered = all_options
        st.warning("No matching tickers found")
    
    ticker_options = [o["label"] for o in filtered]
    ticker_values = [o["value"] for o in filtered]
    
    # Default to Reliance
    default_index = 0
    for i, opt in enumerate(filtered):
        if opt["value"] == "RELIANCE.NS":
            default_index = i
            break
    
    selected_label = st.selectbox(
        "Select Ticker",
        options=ticker_options,
        index=min(default_index, len(ticker_options) - 1),
        key="ticker_select"
    )
    
    selected_index = ticker_options.index(selected_label) if selected_label in ticker_options else 0
    ticker = ticker_values[selected_index] if ticker_values else "RELIANCE.NS"
    
    # Show category and exchange info
    for opt in all_options:
        if opt["value"] == ticker:
            st.caption(f"📌 {opt['category']}")
            break
    
    st.caption(f"🏛️ {get_exchange_info(ticker)}")
    
    with st.expander("➕ Custom Ticker"):
        st.caption("Use .NS for NSE, .BO for BSE")
        custom_ticker = st.text_input("Ticker", placeholder="e.g., TATAMOTORS.NS", key="custom_ticker")
        if custom_ticker:
            ticker = custom_ticker
            st.caption(f"Using: {ticker}")
    
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
        lookback = st.number_input("Lookback", min_value=10, max_value=120, value=60, key="lookback")
        lstm_epochs = st.number_input("Epochs", min_value=5, max_value=100, value=20, key="lstm_epochs")
    
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

st.markdown('<h1 class="main-header">🇮🇳 Indian Stock Market Price Forecasting</h1>', unsafe_allow_html=True)
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
    if st.session_state.data is not None:
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(data))
        with col2:
            st.metric("Start Date", data.index[0].strftime("%Y-%m-%d"))
        with col3:
            st.metric("End Date", data.index[-1].strftime("%Y-%m-%d"))
        with col4:
            st.metric("Price Range", f"₹{data['Close'].min():.2f} - ₹{data['Close'].max():.2f}")
        
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
    if st.session_state.models_trained and st.session_state.results is not None:
        results = st.session_state.results
        
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
                <p>RMSE: ₹{best_model['RMSE']:.2f}</p>
                <p>MAPE: {best_model['MAPE']:.2f}%</p>
                <p>R²: {best_model['R²']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(results['Model'], results['RMSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('RMSE Comparison')
            ax.set_ylabel('RMSE (₹)')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, results['RMSE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'₹{value:.2f}', ha='center', va='bottom', fontsize=10)
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
        st.info("👈 Run the forecast to see model results.")

# ============================================
# TAB 3: Predictions
# ============================================

with tab3:
    if st.session_state.models_trained and st.session_state.predictions:
        data = st.session_state.data
        train_size = int(len(data) * (1 - test_size))
        test_data = data['Close'][train_size:]
        
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
        
        fig.update_layout(height=500, hovermode='x unified')
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
        
        fig2.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👈 Run the forecast to see predictions.")

# ============================================
# TAB 4: Model Comparison
# ============================================

with tab4:
    if st.session_state.models_trained and st.session_state.results is not None:
        results = st.session_state.results
        st.subheader("📋 Detailed Model Comparison")
        st.dataframe(results, use_container_width=True)
        
        st.subheader("💡 Recommendations")
        best_rmse = results.loc[results['RMSE'].idxmin()]
        st.info(f"""
        **📌 Best Model**
        - Model: **{best_rmse['Model']}**
        - RMSE: ₹{best_rmse['RMSE']:.2f}
        - MAPE: {best_rmse['MAPE']:.2f}%
        - R²: {best_rmse['R²']:.4f}
        """)
    else:
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
    ### 🇮🇳 Indian Stock Market Price Forecasting
    
    This interactive dashboard compares three forecasting models for Indian stocks:
    
    #### 🤖 Models Implemented
    
    1. **ARIMA** - Classical statistical model for linear trends
    2. **Prophet** - Facebook's model for seasonal data
    3. **LSTM** - Deep learning model for complex patterns
    
    #### 🇮🇳 Supported Stocks
    
    - NIFTY 50, SENSEX, and other indices
    - 100+ Indian companies across sectors
    - NSE (.NS) and BSE (.BO) support
    
    #### 📝 How to Use
    
    1. Search and select a ticker
    2. Choose date range
    3. Select models
    4. Click "Run Forecast"
    5. Explore results in tabs!
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

def run_forecast():
    """Execute the forecast with rate limiting"""
    st.session_state.forecast_running = True
    st.session_state.logs = []
    
    add_log("🚀 Starting forecast...", "header")
    add_log(f"📊 Ticker: {ticker}")
    
    try:
        # Load data with caching
        add_log("📊 Loading stock data...", "info")
        
        data = fetch_stock_data_with_cache(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if data.empty:
            add_log("❌ No data found", "error")
            st.error(f"No data found for {ticker}")
            st.session_state.forecast_running = False
            return
        
        # Add indicators
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data = data.dropna()
        
        st.session_state.data = data
        add_log(f"✅ Data loaded: {len(data)} rows", "success")
        
        # Split data
        train_size = int(len(data) * (1 - test_size))
        train = data['Close'][:train_size]
        test = data['Close'][train_size:]
        
        add_log(f"📊 Train: {len(train)} days, Test: {len(test)} days", "info")
        
        st.session_state.predictions = {}
        results = []
        
        # ARIMA
        if use_arima and ARIMA_AVAILABLE:
            add_log("🤖 Training ARIMA...", "header")
            try:
                auto_model = auto_arima(train, seasonal=False, stepwise=True, trace=False)
                arima_model = ARIMA(train, order=auto_model.order)
                arima_fitted = arima_model.fit()
                arima_pred = arima_fitted.forecast(steps=len(test))
                st.session_state.predictions['ARIMA'] = arima_pred.values if hasattr(arima_pred, 'values') else arima_pred
                
                rmse = np.sqrt(np.mean((test.values - arima_pred[:len(test)])**2))
                mae = mean_absolute_error(test.values, arima_pred[:len(test)])
                mape = np.mean(np.abs((test.values - arima_pred[:len(test)]) / test.values)) * 100
                ss_res = np.sum((test.values - arima_pred[:len(test)])**2)
                ss_tot = np.sum((test.values - np.mean(test.values))**2)
                r2 = 1 - (ss_res / ss_tot)
                results.append({'Model': 'ARIMA', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                add_log(f"✅ ARIMA complete! RMSE: ₹{rmse:.2f}", "success")
            except Exception as e:
                add_log(f"❌ ARIMA failed: {str(e)[:100]}", "error")
        
        # Prophet
        if use_prophet and PROPHET_AVAILABLE:
            add_log("🤖 Training Prophet...", "header")
            try:
                prophet_data = data.reset_index()[['Date', 'Close']]
                prophet_data.columns = ['ds', 'y']
                train_prophet = prophet_data[:train_size]
                prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                prophet_model.fit(train_prophet)
                future = prophet_model.make_future_dataframe(periods=len(test))
                forecast = prophet_model.predict(future)
                prophet_pred = forecast['yhat'].values[-len(test):]
                st.session_state.predictions['Prophet'] = prophet_pred
                
                rmse = np.sqrt(np.mean((test.values - prophet_pred)**2))
                mae = mean_absolute_error(test.values, prophet_pred)
                mape = np.mean(np.abs((test.values - prophet_pred) / test.values)) * 100
                ss_res = np.sum((test.values - prophet_pred)**2)
                ss_tot = np.sum((test.values - np.mean(test.values))**2)
                r2 = 1 - (ss_res / ss_tot)
                results.append({'Model': 'Prophet', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                add_log(f"✅ Prophet complete! RMSE: ₹{rmse:.2f}", "success")
            except Exception as e:
                add_log(f"❌ Prophet failed: {str(e)[:100]}", "error")
        
        # LSTM
        if use_lstm and LSTM_AVAILABLE:
            add_log("🤖 Training LSTM...", "header")
            try:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data[['Close']])
                
                X, y = [], []
                for i in range(lookback, len(scaled_data)):
                    X.append(scaled_data[i-lookback:i])
                    y.append(scaled_data[i, 0])
                X = np.array(X)
                y = np.array(y)
                
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=lstm_epochs, batch_size=32, verbose=0)
                
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
                r2 = 1 - (ss_res / ss_tot)
                results.append({'Model': 'LSTM', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                add_log(f"✅ LSTM complete! RMSE: ₹{rmse:.2f}", "success")
            except Exception as e:
                add_log(f"❌ LSTM failed: {str(e)[:100]}", "error")
        
        # Complete
        st.session_state.results = pd.DataFrame(results)
        st.session_state.models_trained = True
        
        add_log(f"✅ Forecast complete! {len(results)} models trained.", "success")
        
        if len(results) > 0:
            best = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()]
            add_log(f"🏆 Best: {best['Model']} (RMSE: ₹{best['RMSE']:.2f})", "success")
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
    Built with ❤️ using Streamlit | 🇮🇳 Indian Stock Market Price Forecasting
</div>
""", unsafe_allow_html=True)