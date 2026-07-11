"""
Stock Market Price Forecasting - Production Ready UI
Optimized for Performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import warnings
import traceback
import os
import threading
import gc
import io
import base64
from pathlib import Path

# Used to run LSTM training with a hard timeout in a background thread while still
# letting that thread safely call st.* / write to session_state. Wrapped in try/except
# since this is a semi-internal Streamlit API that has moved between versions.
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    add_script_run_ctx = None
    get_script_run_ctx = None

# Suppress TensorFlow logs and configure for Streamlit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="StockForge - Advanced Market Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/stock-forecasting',
        'Report a bug': 'https://github.com/yourusername/stock-forecasting/issues',
        'About': '# 📊 StockForge v2.0\n\nAdvanced Stock Market Forecasting Platform'
    }
)

# ============================================
# CUSTOM CSS - DARK/LIGHT THEME COMPATIBLE
# ============================================

def load_css():
    """Load production-grade CSS with dark/light theme support"""
    st.markdown("""
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        .main { padding: 0rem 1rem; }
        
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-title { color: white !important; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
        .header-subtitle { color: rgba(255,255,255,0.8) !important; font-size: 0.85rem; margin-top: 0.2rem; }
        .header-badge {
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 0.4rem 1.2rem;
            border-radius: 100px;
            color: white !important;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .status-box {
            padding: 0.8rem 1.2rem;
            border-radius: 12px;
            margin: 0.5rem 0 1rem 0;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 12px;
            border: 1px solid transparent;
            transition: all 0.3s ease;
        }
        .status-box .status-icon { font-size: 1.4rem; }
        .status-box.status-idle {
            background: var(--background-secondary, #f8f9fa);
            border-color: var(--border-color, #e9ecef);
            color: var(--text-color, #495057);
        }
        .status-box.status-processing {
            background: #fff3e0;
            border-color: #ffcc80;
            color: #e65100;
            animation: pulse-border 1.5s ease-in-out infinite;
        }
        .status-box.status-complete {
            background: #e8f5e9;
            border-color: #81c784;
            color: #1b5e20;
        }
        .status-box.status-error {
            background: #ffebee;
            border-color: #ef9a9a;
            color: #b71c1c;
        }
        
        @keyframes pulse-border {
            0%, 100% { border-color: #ffcc80; }
            50% { border-color: #ff6b35; }
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 107, 53, 0.2);
            border-radius: 50%;
            border-top-color: #ff6b35;
            animation: spin 0.8s ease-in-out infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .progress-container {
            width: 100%;
            background: #f0f0f0;
            border-radius: 100px;
            margin: 0.5rem 0;
            overflow: hidden;
            height: 8px;
            position: relative;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 100px;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            width: 0%;
        }
        
        .metric-card {
            background: white;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 1px solid #f0f0f0;
            transition: all 0.3s ease;
        }
        .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.08); }
        .metric-label { font-size: 0.75rem; color: #888; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; margin-top: 0.2rem; }
        .metric-change { font-size: 0.8rem; margin-top: 0.2rem; }
        .metric-change.positive { color: #4caf50; }
        .metric-change.negative { color: #f44336; }
        
        section[data-testid="stSidebar"] {
            background: var(--sidebar-background, #fafafa) !important;
            border-right: 1px solid var(--sidebar-border, #e9ecef) !important;
            padding-top: 0.5rem !important;
        }
        [data-theme="dark"] section[data-testid="stSidebar"] {
            background: #1a1a2e !important;
            border-right: 1px solid #2d2d44 !important;
        }
        section[data-testid="stSidebar"] .stMarkdown { color: var(--text-color, #1a1a2e) !important; }
        section[data-testid="stSidebar"] label { color: var(--text-color, #1a1a2e) !important; }
        [data-theme="dark"] section[data-testid="stSidebar"] .stMarkdown { color: #e0e0e8 !important; }
        [data-theme="dark"] section[data-testid="stSidebar"] label { color: #e0e0e8 !important; }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: var(--background-secondary, #f8f9fa);
            border-radius: 12px;
            padding: 0.3rem;
        }
        [data-theme="dark"] .stTabs [data-baseweb="tab-list"] { background: #2d2d44; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-weight: 500;
            color: var(--text-secondary, #666);
            transition: all 0.3s ease;
        }
        [data-theme="dark"] .stTabs [data-baseweb="tab"] { color: #9999b0; }
        .stTabs [aria-selected="true"] {
            background: white !important;
            color: #667eea !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        [data-theme="dark"] .stTabs [aria-selected="true"] {
            background: #2d2d44 !important;
            color: #667eea !important;
        }
        
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1rem;
            transition: all 0.3s ease;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        .stButton > button:disabled { opacity: 0.6; transform: none; box-shadow: none; }
        
        .log-container {
            background: #1a1a2e;
            color: #e0e0e0;
            padding: 1rem;
            border-radius: 12px;
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 0.8rem;
            height: 250px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid #2d2d44;
        }
        .log-container::-webkit-scrollbar { width: 6px; }
        .log-container::-webkit-scrollbar-track { background: #1a1a2e; }
        .log-container::-webkit-scrollbar-thumb { background: #4a4a6a; border-radius: 3px; }
        .log-info { color: #4fc3f7; }
        .log-success { color: #81c784; }
        .log-warning { color: #ffb74d; }
        .log-error { color: #e57373; }
        .log-header { color: #ce93d8; font-weight: bold; }
        
        .india-badge, .us-badge {
            display: inline-block;
            padding: 0.2rem 0.8rem;
            border-radius: 100px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .india-badge { background: #ff6b35; color: white; }
        .us-badge { background: #1f77b4; color: white; }
        
        hr { margin: 0.5rem 0 !important; }
        
        @media (max-width: 768px) {
            .header-container { flex-direction: column; text-align: center; padding: 1rem; }
            .header-title { font-size: 1.5rem; }
            .metric-value { font-size: 1.4rem; }
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================
# TRY IMPORT MODELS
# ============================================

try:
    from statsmodels.tsa.arima.model import ARIMA
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
# TICKER DATABASE
# ============================================

INDIAN_TICKERS = {
    "📊 Indices": {"^NSEI": "NIFTY 50", "^BSESN": "SENSEX", "^NSEBANK": "NIFTY BANK"},
    "🏢 IT": {"TCS.NS": "TCS", "INFY.NS": "Infosys", "WIPRO.NS": "Wipro", "HCLTECH.NS": "HCL Tech"},
    "🏦 Banking": {"HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "SBIN.NS": "SBI"},
    "🛢️ Energy": {"RELIANCE.NS": "Reliance", "ONGC.NS": "ONGC"},
    "🏭 Auto": {"TATAMOTORS.NS": "Tata Motors", "MARUTI.NS": "Maruti Suzuki"},
    "🏥 Pharma": {"SUNPHARMA.NS": "Sun Pharma", "DRREDDY.NS": "Dr Reddy's"},
    "🛒 FMCG": {"ITC.NS": "ITC", "HINDUNILVR.NS": "HUL"}
}

INTERNATIONAL_TICKERS = {
    "📊 US Indices": {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"},
    "🏢 Tech": {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon", "NVDA": "NVIDIA", "TSLA": "Tesla"},
    "🏦 Finance": {"JPM": "JPMorgan", "BAC": "Bank of America", "V": "Visa"},
    "🛒 Retail": {"WMT": "Walmart", "TGT": "Target", "COST": "Costco"},
    "💊 Healthcare": {"JNJ": "Johnson & Johnson", "PFE": "Pfizer"},
    "📈 ETFs": {"SPY": "S&P 500 ETF", "QQQ": "NASDAQ ETF"}
}

def get_ticker_options(market_type="India"):
    options = []
    categories = INDIAN_TICKERS if market_type == "India" else INTERNATIONAL_TICKERS
    for category, tickers in categories.items():
        for ticker, name in tickers.items():
            options.append({"label": f"{name} ({ticker})", "value": ticker, "category": category})
    return sorted(options, key=lambda x: x["label"])

def get_currency_symbol(ticker):
    if ticker.endswith('.NS') or ticker.endswith('.BO') or ticker.startswith('^NSE') or ticker.startswith('^BSESN'):
        return "₹"
    return "$"

# ============================================
# REPORT GENERATION (HTML, saved locally)
# ============================================

def _save_and_encode_fig(fig, path):
    """Save a matplotlib figure to disk AND return a base64 PNG string for embedding."""
    fig.savefig(path, bbox_inches="tight", dpi=110)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def generate_report(ticker, market_type, start_date, end_date, data, results, predictions, test_size):
    """
    Builds a self-contained HTML report (charts embedded as base64 PNGs, so it opens
    correctly with no internet connection or extra files) and saves it, along with the
    individual chart PNGs and the underlying data as CSVs, to:
        reports/<TICKER>_<timestamp>/
    Returns a dict with paths and the raw HTML bytes (for the in-app download button).
    """
    currency = get_currency_symbol(ticker)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace('^', 'idx_').replace('.', '_')
    report_dir = REPORTS_DIR / f"{safe_ticker}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    images = {}

    # --- Price history ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Close'], color='#667eea', linewidth=1.5)
    ax.fill_between(data.index, data['Close'], data['Close'].min(), color='#667eea', alpha=0.08)
    ax.set_title(f"{ticker} - Price History")
    ax.set_ylabel(f"Price ({currency})")
    ax.grid(True, alpha=0.3)
    images['price_history'] = _save_and_encode_fig(fig, report_dir / "price_history.png")

    # --- Model comparison (RMSE / MAPE) ---
    if results is not None and not results.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors = ['#667eea', '#764ba2', '#f093fb'][:len(results)]
        axes[0].bar(results['Model'], results['RMSE'], color=colors)
        axes[0].set_title('RMSE Comparison')
        axes[0].set_ylabel(f'RMSE ({currency})')
        axes[0].grid(True, alpha=0.3)
        axes[1].bar(results['Model'], results['MAPE'], color=colors)
        axes[1].set_title('MAPE Comparison')
        axes[1].set_ylabel('MAPE (%)')
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        images['model_comparison'] = _save_and_encode_fig(fig, report_dir / "model_comparison.png")

    # --- Predictions vs actual ---
    predictions_csv_df = None
    if predictions:
        train_size = int(len(data) * (1 - test_size))
        test_data = data['Close'][train_size:]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(test_data.index, test_data.values, color='#1a1a2e', linewidth=2, label='Actual')
        colors = ['#667eea', '#764ba2', '#f093fb']
        predictions_csv_df = pd.DataFrame({'Date': test_data.index, 'Actual': test_data.values})
        for i, (name, pred) in enumerate(predictions.items()):
            if len(pred) > 0:
                pred = np.asarray(pred)[:len(test_data)]
                ax.plot(test_data.index[:len(pred)], pred, linestyle='--', linewidth=1.5,
                        color=colors[i % len(colors)], label=name)
                col = pd.Series(pred, index=test_data.index[:len(pred)])
                predictions_csv_df[name] = col.reindex(predictions_csv_df['Date']).values
        ax.set_title(f'{ticker} - Forecast vs Actual')
        ax.set_ylabel(f'Price ({currency})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        images['predictions'] = _save_and_encode_fig(fig, report_dir / "predictions_vs_actual.png")

        predictions_csv_df.to_csv(report_dir / "predictions.csv", index=False)

    # --- Save underlying data as CSVs ("other files") ---
    data.to_csv(report_dir / "price_history.csv")
    if results is not None and not results.empty:
        results.to_csv(report_dir / "model_results.csv", index=False)

    # --- Build HTML ---
    best_model_html = ""
    if results is not None and not results.empty:
        best = results.loc[results['RMSE'].idxmin()]
        best_model_html = f"""
        <div class="best-model">
            <div class="label">🏆 Best Performing Model</div>
            <div class="name">{best['Model']}</div>
            <div class="stats">RMSE: {currency}{best['RMSE']:.2f} &nbsp;·&nbsp; MAPE: {best['MAPE']:.2f}% &nbsp;·&nbsp; R²: {best['R²']:.4f}</div>
        </div>
        """

    results_table_html = results.to_html(index=False, classes="data-table", border=0) if results is not None and not results.empty else "<p>No model results available.</p>"

    def img_block(key, title):
        if key not in images:
            return ""
        return f"""
        <div class="chart-block">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{images[key]}" />
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>StockForge Report - {ticker}</title>
<style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: #f8f9fa; color: #1a1a2e; margin: 0; padding: 2rem; }}
    .container {{ max-width: 960px; margin: 0 auto; background: white; border-radius: 12px; padding: 2rem 2.5rem; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; }}
    .header h1 {{ margin: 0 0 0.3rem 0; font-size: 1.6rem; }}
    .header .meta {{ opacity: 0.85; font-size: 0.9rem; }}
    .best-model {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.2rem 1.5rem; border-radius: 10px; margin: 1.5rem 0; }}
    .best-model .label {{ font-size: 0.8rem; opacity: 0.85; }}
    .best-model .name {{ font-size: 1.5rem; font-weight: 700; }}
    .chart-block {{ margin: 2rem 0; }}
    .chart-block h3 {{ font-size: 1.05rem; color: #333; margin-bottom: 0.6rem; }}
    .chart-block img {{ width: 100%; border-radius: 8px; border: 1px solid #eee; }}
    table.data-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
    table.data-table th, table.data-table td {{ padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
    table.data-table th {{ background: #f8f9fa; font-weight: 600; }}
    .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; color: #999; font-size: 0.75rem; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 StockForge Report — {ticker}</h1>
        <div class="meta">{market_type} Market &nbsp;·&nbsp; {start_date} to {end_date} &nbsp;·&nbsp; Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    {best_model_html}

    <h2>Model Performance</h2>
    {results_table_html}

    {img_block('model_comparison', 'RMSE / MAPE Comparison')}
    {img_block('price_history', 'Price History')}
    {img_block('predictions', 'Predictions vs Actual')}

    <div class="footer">
        Generated by StockForge v2.0. Companion files (chart PNGs, price history CSV, model results CSV, predictions CSV) saved alongside this report in the same folder.
    </div>
</div>
</body>
</html>"""

    html_path = report_dir / "report.html"
    html_bytes = html.encode("utf-8")
    html_path.write_bytes(html_bytes)

    return {
        "report_dir": report_dir,
        "html_path": html_path,
        "html_bytes": html_bytes,
    }

# FIX: Yahoo aggressively rate-limits/blocks non-browser traffic, which surfaces as
# "No timezone found, symbol may be delisted" / "Expecting value: line 1 column 1"
# for EVERY ticker (not just delisted ones). Using a curl_cffi session that
# impersonates a real browser's TLS fingerprint is the current workaround, along
# with a small backoff between retries.
def _get_yf_session():
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except ImportError:
        # curl_cffi not installed - fall back to yfinance's default session.
        # `pip install curl_cffi` is strongly recommended if you keep seeing
        # "No timezone found" errors for every ticker.
        return None

# ============================================
# DATA FETCHING
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data with caching, browser-impersonation session, and retry/backoff."""
    session = _get_yf_session()
    last_error = None

    for attempt in range(3):
        try:
            kwargs = {"session": session} if session is not None else {}

            stock = yf.Ticker(ticker, **kwargs)
            data = stock.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

            if not data.empty:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in data.columns for col in required_cols):
                    data = data.dropna(how='all')
                    if not data.empty:
                        return data

            # Fall back to yf.download as a second code path
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                session=session,
            )

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not data.empty:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in data.columns for col in required_cols):
                    data = data.dropna(how='all')
                    if not data.empty:
                        return data

        except Exception as e:
            last_error = e

        # Exponential backoff between retries - Yahoo's rate limiting is often
        # short-lived (a few seconds), so this alone can resolve transient blocks.
        if attempt < 2:
            time.sleep(2 ** attempt)

    if last_error:
        print(f"Error fetching {ticker} after 3 attempts: {last_error}")

    return pd.DataFrame()

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
if 'market_type' not in st.session_state:
    st.session_state.market_type = "India"
if 'current_process' not in st.session_state:
    st.session_state.current_process = "Ready"
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'trigger_forecast' not in st.session_state:
    st.session_state.trigger_forecast = False
# FIX: persistent, session-backed date range (preset buttons write here)
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime(2021, 1, 1)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now()
if 'last_report' not in st.session_state:
    st.session_state.last_report = None

REPORTS_DIR = Path.home() / "Documents" / "StockForgeReports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 0.5rem 0;">
        <div style="font-size: 2.2rem;">📊</div>
        <div style="font-size: 1rem; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">StockForge</div>
        <div style="font-size: 0.6rem; color: #888;">v2.0 · Advanced Forecasting</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Market Selection
    st.markdown("### 🌍 Market")
    market_col1, market_col2 = st.columns(2)
    with market_col1:
        if st.button("🇮🇳 India", use_container_width=True,
                    type="primary" if st.session_state.market_type == "India" else "secondary"):
            st.session_state.market_type = "India"
            st.session_state.data = None
            st.session_state.predictions = {}
            st.session_state.results = None
            st.session_state.models_trained = False
            st.session_state.forecast_completed = False
            # st.rerun()
    with market_col2:
        if st.button("🌍 Global", use_container_width=True,
                    type="primary" if st.session_state.market_type == "International" else "secondary"):
            st.session_state.market_type = "International"
            st.session_state.data = None
            st.session_state.predictions = {}
            st.session_state.results = None
            st.session_state.models_trained = False
            st.session_state.forecast_completed = False
            # st.rerun()
    
    # Show current market badge
    if st.session_state.market_type == "India":
        st.markdown('<div style="text-align: center; margin: 0.2rem 0;"><span class="india-badge">🇮🇳 INDIAN STOCKS</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center; margin: 0.2rem 0;"><span class="us-badge">🌍 INTERNATIONAL STOCKS</span></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Stock Selection
    st.markdown("### 📈 Stock")
    all_options = get_ticker_options(st.session_state.market_type)
    
    search = st.text_input("🔍 Search", placeholder="Type to search...", label_visibility="collapsed")
    
    if search:
        filtered = [o for o in all_options if search.lower() in o["label"].lower()]
    else:
        filtered = all_options
    
    if not filtered:
        filtered = all_options
    
    selected = st.selectbox("Select Ticker", [o["label"] for o in filtered], label_visibility="collapsed")
    ticker = [o["value"] for o in filtered if o["label"] == selected][0] if filtered else "AAPL"
    
    category = [o['category'] for o in filtered if o['value'] == ticker][0] if filtered else ""
    st.caption(f"📌 {category}")
    
    st.divider()
    
    # Date Range
    st.markdown("### 📅 Date Range")

    # FIX: quick presets now placed above the pickers and write to session_state
    # *before* the date_input widgets are instantiated, so the change actually sticks.
    cols = st.columns(4)
    presets = [("1Y", 365), ("2Y", 730), ("3Y", 1095), ("5Y", 1825)]
    for col, (label, days) in zip(cols, presets):
        with col:
            if st.button(label, use_container_width=True, key=f"preset_{label}"):
                st.session_state.start_date = datetime.now() - timedelta(days=days)
                st.session_state.end_date = datetime.now()
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=st.session_state.start_date, key="start_date_input")
        st.session_state.start_date = start_date
    with col2:
        end_date = st.date_input("End", value=st.session_state.end_date, key="end_date_input")
        st.session_state.end_date = end_date
    
    st.divider()
    
    # Model Selection
    st.markdown("### 🤖 Models")
    use_arima = st.checkbox("ARIMA", value=ARIMA_AVAILABLE, disabled=not ARIMA_AVAILABLE)
    use_prophet = st.checkbox("Prophet", value=PROPHET_AVAILABLE, disabled=not PROPHET_AVAILABLE)
    use_lstm = st.checkbox("LSTM", value=LSTM_AVAILABLE, disabled=not LSTM_AVAILABLE)
    
    if not any([use_arima, use_prophet, use_lstm]):
        st.warning("⚠️ Select at least one model")
    
    st.divider()
    
    # Parameters
    st.markdown("### ⚙️ Settings")
    test_size = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
    
    with st.expander("🧠 LSTM Settings"):
        lookback = st.slider("Lookback Days", 10, 60, 20)
        lstm_epochs = st.slider("Epochs", 3, 10, 5)
    
    st.divider()
    
    # Run Button
    run_button = st.button(
        "🚀 Run Forecast", 
        use_container_width=True,
        disabled=st.session_state.forecast_running,
        type="primary"
    )
    
    st.caption("📌 v2.0 | ARIMA · Prophet · LSTM")

# ============================================
# MAIN CONTENT
# ============================================

# Header
st.markdown(f"""
<div class="header-container">
    <div>
        <div class="header-title">📊 {'' if st.session_state.market_type == 'India' else '🌍 '}Market Forecasting</div>
        <div class="header-subtitle">Advanced ARIMA · Prophet · LSTM Models</div>
    </div>
    <div class="header-badge">
        {st.session_state.market_type} Market • {ticker}
    </div>
</div>
""", unsafe_allow_html=True)

# Status Bar
# FIX: this placeholder is what we update *live* from inside run_forecast(),
# instead of relying on session_state + a full rerun (which never happened mid-run before).
status_placeholder = st.empty()

def render_status_html(state, message="", progress=0):
    if state == "processing":
        status_placeholder.markdown(f"""
        <div class="status-box status-processing">
            <span class="status-icon">⏳</span>
            <div style="flex: 1;">
                <div>Processing: {message}</div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {progress}%;"></div>
                </div>
            </div>
            <div class="spinner-container">
                <div class="spinner"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif state == "complete":
        status_placeholder.markdown("""
        <div class="status-box status-complete">
            <span class="status-icon">✅</span>
            <span><strong>Forecast Complete!</strong> Results are ready below.</span>
        </div>
        """, unsafe_allow_html=True)
    elif state == "error":
        status_placeholder.markdown("""
        <div class="status-box status-error">
            <span class="status-icon">❌</span>
            <span><strong>Forecast Failed</strong> Check logs for details.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        status_placeholder.markdown("""
        <div class="status-box status-idle">
            <span class="status-icon">💡</span>
            <span>Configure settings and click <strong>Run Forecast</strong> to start analysis.</span>
        </div>
        """, unsafe_allow_html=True)

# Render initial (idle/complete/error) status based on last known state
if st.session_state.forecast_completed:
    render_status_html("complete")
elif st.session_state.forecast_error:
    render_status_html("error")
else:
    render_status_html("idle")

# ============================================
# SAVE REPORT (HTML, written to local ./reports/ folder on demand)
# ============================================

if (st.session_state.forecast_completed and st.session_state.data is not None
        and st.session_state.results is not None):
    save_col1, save_col2 = st.columns([1, 3])
    with save_col1:
        if st.button("💾 Save Report", use_container_width=True):
            with st.spinner("Building report..."):
                st.session_state.last_report = generate_report(
                    ticker, st.session_state.market_type, start_date, end_date,
                    st.session_state.data, st.session_state.results,
                    st.session_state.predictions, test_size
                )
    with save_col2:
        info = st.session_state.last_report
        if info is not None:
            st.success(f"✅ Saved to `{info['report_dir']}` (report.html + chart PNGs + CSVs)")
            st.download_button(
                "⬇️ Download report.html",
                data=info["html_bytes"],
                file_name=info["html_path"].name,
                mime="text/html",
                use_container_width=False,
            )

# ============================================
# TABS
# ============================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "📈 Model Results", 
    "📉 Predictions",
    "📋 Comparison",
    "📝 Logs"
])

# ============================================
# TAB 1: Dashboard
# ============================================

with tab1:
    if st.session_state.data is not None:
        data = st.session_state.data
        currency = get_currency_symbol(ticker)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">{currency}{data['Close'].iloc[-1]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
            pct = (change / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
            color = "positive" if change >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Change</div>
                <div class="metric-value {color}">{'▲' if change >= 0 else '▼'} {currency}{abs(change):.2f}</div>
                <div class="metric-change {color}">{pct:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">52-Week Range</div>
                <div class="metric-value">{currency}{data['Close'].min():.2f}</div>
                <div class="metric-change">to {currency}{data['Close'].max():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volatility (30d)</div>
                <div class="metric-value">{data['Close'].pct_change().std() * 100:.2f}%</div>
                <div class="metric-change">Annualized: {(data['Close'].pct_change().std() * np.sqrt(252) * 100):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Price History")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            mode='lines', name='Close Price',
            line=dict(color='#667eea', width=2),
            fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=data.index, y=data['Volume'],
            name='Volume', marker_color='rgba(102, 126, 234, 0.3)'
        ), row=2, col=1)
        
        fig.update_layout(height=500, showlegend=False, hovermode='x unified')
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Raw Data"):
            st.dataframe(data.tail(20), use_container_width=True)
    else:
        st.info("👈 Configure settings and click **Run Forecast** to get started.")

# ============================================
# TAB 2: Model Results
# ============================================

with tab2:
    if st.session_state.models_trained and st.session_state.results is not None and not st.session_state.results.empty:
        results = st.session_state.results
        currency = get_currency_symbol(ticker)
        
        st.markdown("### 📊 Model Performance")
        
        st.dataframe(
            results.style.background_gradient(subset=['RMSE', 'MAE', 'MAPE'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        if not results.empty:
            best = results.loc[results['RMSE'].idxmin()]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; margin-top: 1rem;">
                <div style="font-size: 0.8rem; opacity: 0.8;">🏆 Best Performing Model</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{best['Model']}</div>
                <div style="display: flex; gap: 2rem; margin-top: 0.5rem; flex-wrap: wrap;">
                    <div><span style="opacity: 0.7;">RMSE:</span> {currency}{best['RMSE']:.2f}</div>
                    <div><span style="opacity: 0.7;">MAPE:</span> {best['MAPE']:.2f}%</div>
                    <div><span style="opacity: 0.7;">R²:</span> {best['R²']:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(results['Model'], results['RMSE'], 
                         color=['#667eea', '#764ba2', '#f093fb'][:len(results)])
            ax.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'RMSE ({currency})')
            ax.grid(True, alpha=0.3)
            for bar, val in zip(bars, results['RMSE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(results['Model'], results['MAPE'],
                         color=['#667eea', '#764ba2', '#f093fb'][:len(results)])
            ax.set_title('MAPE Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('MAPE (%)')
            ax.grid(True, alpha=0.3)
            for bar, val in zip(bars, results['MAPE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
    else:
        st.info("👈 Run forecast to see model results.")

# ============================================
# TAB 3: Predictions
# ============================================

with tab3:
    if st.session_state.models_trained and st.session_state.predictions:
        data = st.session_state.data
        train_size = int(len(data) * (1 - test_size))
        test_data = data['Close'][train_size:]
        currency = get_currency_symbol(ticker)
        
        st.markdown("### 📈 Predictions vs Actual")
        
        selected_models = st.multiselect(
            "Select models to display",
            options=list(st.session_state.predictions.keys()),
            default=list(st.session_state.predictions.keys())
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_data.index, y=test_data.values,
            mode='lines', name='Actual',
            line=dict(color='#1a1a2e', width=3)
        ))
        
        colors = ['#667eea', '#764ba2', '#f093fb']
        for i, (name, pred) in enumerate(st.session_state.predictions.items()):
            if name in selected_models and len(pred) > 0:
                fig.add_trace(go.Scatter(
                    x=test_data.index[:len(pred)], y=pred[:len(test_data)],
                    mode='lines', name=name,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            title=f'{ticker} - Forecast vs Actual',
            xaxis_title='Date',
            yaxis_title=f'Price ({currency})'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Error Analysis"):
            for name, pred in st.session_state.predictions.items():
                if len(pred) > 0:
                    errors = test_data.values[:len(pred)] - pred[:len(test_data)]
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax.plot(range(len(errors)), errors, label=name, alpha=0.7)
                    ax.axhline(y=np.mean(errors), color='red', linestyle='--', alpha=0.5, label='Mean Error')
                    ax.fill_between(range(len(errors)), -np.std(errors), np.std(errors), alpha=0.2, color='gray')
                    ax.set_title(f'{name} - Prediction Errors')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Error')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
    else:
        st.info("👈 Run forecast to see predictions.")

# ============================================
# TAB 4: Comparison
# ============================================

with tab4:
    if st.session_state.models_trained and st.session_state.results is not None and not st.session_state.results.empty:
        results = st.session_state.results
        
        st.markdown("### 📋 Model Comparison")
        
        st.dataframe(
            results.style.background_gradient(subset=['RMSE', 'MAE', 'MAPE'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        st.markdown("### 💡 Recommendations")
        
        if not results.empty:
            best_rmse = results.loc[results['RMSE'].idxmin()]
            best_mape = results.loc[results['MAPE'].idxmin()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="background: #f0f4ff; padding: 1rem; border-radius: 12px; border-left: 4px solid #667eea;">
                    <div style="font-size: 0.8rem; color: #666;">🎯 Best by Accuracy (RMSE)</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #667eea;">{best_rmse['Model']}</div>
                    <div style="font-size: 0.9rem; color: #888;">RMSE: {best_rmse['RMSE']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background: #f5f0ff; padding: 1rem; border-radius: 12px; border-left: 4px solid #764ba2;">
                    <div style="font-size: 0.8rem; color: #666;">🎯 Best by Percentage (MAPE)</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #764ba2;">{best_mape['Model']}</div>
                    <div style="font-size: 0.9rem; color: #888;">MAPE: {best_mape['MAPE']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 📊 When to Use Each Model
            
            | Model | Best For | Key Strength |
            |-------|----------|--------------|
            | **ARIMA** | Short-term (1-10 days) | Fast, interpretable |
            | **Prophet** | Seasonal patterns | Handles seasonality well |
            | **LSTM** | Long-term (30+ days) | Captures complex patterns |
            """)
    else:
        st.info("👈 Run forecast to see comparison.")

# ============================================
# TAB 5: Logs
# ============================================

with tab5:
    st.markdown("### 📝 Live Terminal Logs")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.logs = []
            st.rerun()
    with col3:
        st.caption(f"📊 {len(st.session_state.logs)} log entries")
    
    if st.session_state.logs:
        log_html = ""
        for log in st.session_state.logs[-100:]:
            color_map = {"info": "log-info", "success": "log-success", 
                        "warning": "log-warning", "error": "log-error", "header": "log-header"}
            color = color_map.get(log.get("type", "info"), "log-info")
            log_html += f'<div><span style="color: #888;">[{log["timestamp"]}]</span> <span class="{color}">{log["message"]}</span></div>'
        st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.info("📝 No logs yet. Run the forecast to see terminal output.")

# ============================================
# BACKEND LOGIC
# ============================================

def add_log(message, log_type="info"):
    st.session_state.logs.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "type": log_type
    })

def safe_mape(actual, pred):
    """FIX: guard against division-by-zero / near-zero prices corrupting MAPE."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.abs(actual) > 1e-8
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

def train_arima_optimized(train_data, test_len):
    """Optimized ARIMA training with better error handling"""
    try:
        train_clean = train_data.dropna()
        if len(train_clean) < 50:
            return None, None
        
        for order in [(1,1,1), (1,1,0), (0,1,1), (2,1,1)]:
            try:
                model = ARIMA(train_clean, order=order)
                fitted = model.fit()
                pred = fitted.forecast(steps=test_len)
                if not np.isnan(pred).any() and len(pred) == test_len:
                    return pred, fitted
            except Exception:
                continue
        
        # Fallback to auto_arima
        try:
            from pmdarima import auto_arima
            auto_model = auto_arima(
                train_clean,
                start_p=0, max_p=2,
                start_q=0, max_q=2,
                max_d=1,
                seasonal=False,
                stepwise=True,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                n_fits=5,
                n_jobs=1
            )
            model = ARIMA(train_clean, order=auto_model.order)
            fitted = model.fit()
            pred = fitted.forecast(steps=test_len)
            if not np.isnan(pred).any():
                return pred, fitted
        except Exception:
            pass
        
        return None, None
    except Exception:
        return None, None

def run_forecast(ticker, start_date, end_date, use_arima, use_prophet, use_lstm,
                  test_size, lookback, lstm_epochs):
    st.session_state.forecast_running = True
    st.session_state.forecast_completed = False
    st.session_state.forecast_error = False
    st.session_state.logs = []
    
    # FIX: live-updating progress. render_status_html() writes directly into the
    # status_placeholder created earlier in this script run, so the browser actually
    # sees each stage instead of everything happening invisibly between reruns.
    def set_progress(pct, stage):
        st.session_state.progress = pct
        st.session_state.current_process = stage
        render_status_html("processing", message=stage, progress=pct)

    add_log("🚀 Launching StockForge Analysis", "header")
    add_log(f"📊 Ticker: {ticker}", "info")
    add_log(f"📅 Range: {start_date} to {end_date}", "info")
    add_log(f"🤖 Models: {'ARIMA ' if use_arima else ''}{'Prophet ' if use_prophet else ''}{'LSTM' if use_lstm else ''}", "info")
    
    try:
        # STEP 1: Load Data
        add_log("📊 Loading market data...", "info")
        set_progress(5, "Loading Data")
        
        data = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if data.empty:
            add_log("❌ No data found for ticker", "error")
            add_log("   Possible reasons:", "error")
            add_log("   • Ticker symbol is incorrect or delisted", "error")
            add_log("   • Yahoo Finance API is temporarily unavailable", "error")
            add_log("   • Network connectivity issues", "error")
            add_log("   Please try a different ticker or check your connection.", "error")
            
            st.session_state.forecast_running = False
            st.session_state.forecast_error = True
            render_status_html("error")
            return
        
        add_log(f"✅ Loaded {len(data)} rows", "success")
        st.session_state.data = data
        set_progress(15, "Loaded Data")
        
        # STEP 2: Add Indicators
        add_log("📊 Computing technical indicators...", "info")
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data = data.dropna()
        add_log("✅ Technical indicators added", "success")
        set_progress(25, "Indicators Ready")
        
        if len(data) < 60:
            add_log("❌ Not enough data after indicator warm-up period. Choose a longer date range.", "error")
            st.session_state.forecast_running = False
            st.session_state.forecast_error = True
            render_status_html("error")
            return
        
        # STEP 3: Split Data
        # FIX: single, shared split point (by index position) used by every model,
        # so ARIMA/Prophet/LSTM are all evaluated against the exact same test dates.
        train_size = int(len(data) * (1 - test_size))
        train = data['Close'][:train_size]
        test = data['Close'][train_size:]
        add_log(f"📊 Train: {len(train)} days, Test: {len(test)} days", "info")
        set_progress(35, "Data Split")
        
        st.session_state.predictions = {}
        results = []
        
        total_models = sum([use_arima, use_prophet, use_lstm]) or 1
        model_idx = 0
        
        # ============================================
        # ARIMA
        # ============================================
        if use_arima and ARIMA_AVAILABLE:
            model_idx += 1
            pct = 35 + (model_idx / total_models * 50)
            add_log("🤖 Training ARIMA...", "header")
            set_progress(pct, "Training ARIMA")
            
            try:
                add_log(f"   Train data: {len(train)} samples", "info")
                pred, fitted = train_arima_optimized(train, len(test))
                
                if pred is not None and not np.isnan(pred).any():
                    pred = np.asarray(pred)[:len(test)]
                    st.session_state.predictions['ARIMA'] = pred
                    
                    rmse = float(np.sqrt(np.mean((test.values - pred) ** 2)))
                    mae = float(mean_absolute_error(test.values, pred))
                    mape = safe_mape(test.values, pred)
                    ss_res = np.sum((test.values - pred) ** 2)
                    ss_tot = np.sum((test.values - np.mean(test.values)) ** 2)
                    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                    results.append({'Model': 'ARIMA', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                    add_log(f"✅ ARIMA complete (RMSE: {rmse:.2f})", "success")
                else:
                    add_log("⚠️ ARIMA could not fit - predictions were None or NaN", "warning")
            except Exception as e:
                add_log(f"❌ ARIMA failed: {str(e)[:80]}", "error")
        
        # ============================================
        # Prophet
        # ============================================
        if use_prophet and PROPHET_AVAILABLE:
            model_idx += 1
            pct = 35 + (model_idx / total_models * 50)
            add_log("🤖 Training Prophet...", "header")
            set_progress(pct, "Training Prophet")
            
            try:
                prophet_data = data.reset_index()
                date_col = prophet_data.columns[0]  # FIX: don't hard-assume 'Date'
                prophet_data = prophet_data[[date_col, 'Close']]
                prophet_data.columns = ['ds', 'y']
                if pd.api.types.is_datetime64_any_dtype(prophet_data['ds']):
                    prophet_data['ds'] = pd.to_datetime(prophet_data['ds']).dt.tz_localize(None)
                
                train_p = prophet_data[:train_size]
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model.fit(train_p)
                future = model.make_future_dataframe(periods=len(test))
                forecast = model.predict(future)
                pred = forecast['yhat'].values[-len(test):]
                st.session_state.predictions['Prophet'] = pred
                
                rmse = float(np.sqrt(np.mean((test.values - pred) ** 2)))
                mae = float(mean_absolute_error(test.values, pred))
                mape = safe_mape(test.values, pred)
                ss_res = np.sum((test.values - pred) ** 2)
                ss_tot = np.sum((test.values - np.mean(test.values)) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                results.append({'Model': 'Prophet', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                add_log(f"✅ Prophet complete (RMSE: {rmse:.2f})", "success")
            except Exception as e:
                add_log(f"❌ Prophet failed: {str(e)[:80]}", "error")
        
        # ============================================
        # LSTM
        # ============================================
        if use_lstm and LSTM_AVAILABLE:
            model_idx += 1
            pct = 35 + (model_idx / total_models * 50)
            add_log("🤖 Training LSTM...", "header")
            set_progress(pct, "Training LSTM")
            
            # FIX: Streamlit keeps the same Python process alive across reruns, so
            # TensorFlow's graph/session state from a PREVIOUS forecast run is still
            # sitting in memory. Rebuilding a model on top of that stale state is what
            # causes runs after the first one to slow down or hang partway through
            # training. Clear it out before building anything new.
            try:
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception:
                pass

            try:
                actual_lookback = min(lookback, max(5, train_size - 1))
                add_log(f"   Lookback: {actual_lookback}, Epochs: {min(lstm_epochs, 5)}", "info")
                
                # FIX: scale using ONLY the training portion to avoid leaking test-set
                # information into the scaler, and build train/test sequences from the
                # *same* train/test split used by ARIMA/Prophet (aligned on train_size),
                # instead of re-splitting the windowed sequences by test_size again.
                scaler = MinMaxScaler()
                scaler.fit(data[['Close']].iloc[:train_size])
                scaled = scaler.transform(data[['Close']])
                
                X, y = [], []
                for i in range(actual_lookback, len(scaled)):
                    X.append(scaled[i - actual_lookback:i])
                    y.append(scaled[i, 0])
                X = np.array(X)
                y = np.array(y)

                # Index i in scaled corresponds to X/y row (i - actual_lookback).
                # Test rows are those whose target index i >= train_size.
                split = train_size - actual_lookback
                split = max(split, 0)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                add_log(f"   Train: {len(X_train)}, Test: {len(X_test)}", "info")
                
                if len(X_train) > 0 and len(X_test) > 0:
                    add_log("   Building model...", "info")
                    model = Sequential()
                    model.add(LSTM(32, return_sequences=True, input_shape=(actual_lookback, 1)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(16, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    
                    add_log("   Training started...", "info")
                    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

                    n_epochs = min(lstm_epochs, 5)
                    pct_base = pct
                    pct_span = (85 - pct)  # LSTM's slice of the overall progress bar

                    # Live per-epoch feedback so the UI doesn't sit silent for the
                    # whole training run.
                    class StreamlitEpochLogger(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            loss = (logs or {}).get('loss')
                            epoch_pct = pct_base + (epoch + 1) / n_epochs * pct_span
                            stage = f"Training LSTM — epoch {epoch + 1}/{n_epochs}"
                            if loss is not None:
                                stage += f" (loss: {loss:.4f})"
                            set_progress(epoch_pct, stage)
                            add_log(f"   Epoch {epoch + 1}/{n_epochs} - loss: {loss:.4f}" if loss is not None
                                    else f"   Epoch {epoch + 1}/{n_epochs}", "info")

                    # FIX: run .fit() in a background thread with a hard timeout so a
                    # hang (TF resource contention, a slow/loaded machine, etc.) can
                    # never freeze the whole app again. If it doesn't finish in time we
                    # abandon LSTM and still return ARIMA/Prophet results below.
                    LSTM_TIMEOUT_SECONDS = 120
                    fit_outcome = {}

                    def _fit_worker():
                        try:
                            fit_outcome['history'] = model.fit(
                                X_train, y_train,
                                epochs=n_epochs,
                                batch_size=32,
                                verbose=0,
                                callbacks=[early_stop, StreamlitEpochLogger()]
                            )
                        except Exception as fit_err:
                            fit_outcome['error'] = fit_err

                    fit_thread = threading.Thread(target=_fit_worker, daemon=True)
                    if add_script_run_ctx is not None:
                        # Lets the worker thread safely update session_state / the
                        # status placeholder via the callback above.
                        ctx = get_script_run_ctx()
                        if ctx is not None:
                            add_script_run_ctx(fit_thread, ctx)
                    fit_thread.start()
                    fit_thread.join(timeout=LSTM_TIMEOUT_SECONDS)

                    if fit_thread.is_alive():
                        add_log(f"⚠️ LSTM training exceeded {LSTM_TIMEOUT_SECONDS}s and was abandoned. "
                                f"Skipping LSTM for this run — ARIMA/Prophet results are unaffected.", "warning")
                        # Thread is daemonized so it won't block the app; it'll keep
                        # running in the background and get garbage-collected eventually.
                    elif 'error' in fit_outcome:
                        raise fit_outcome['error']
                    else:
                        add_log("   Generating predictions...", "info")
                        pred_scaled = model.predict(X_test, verbose=0)
                        pred = scaler.inverse_transform(
                            np.concatenate([pred_scaled, np.zeros((len(pred_scaled), 1))], axis=1)
                        )[:, 0]
                        
                        # Align length to `test` exactly (test set now matches by
                        # construction, but guard for the lookback edge-case where a
                        # few leading rows are lost).
                        if len(pred) < len(test):
                            pad_n = len(test) - len(pred)
                            pred = np.concatenate([np.full(pad_n, pred[0] if len(pred) else np.nan), pred])
                        pred = pred[-len(test):]
                        
                        st.session_state.predictions['LSTM'] = pred
                        
                        rmse = float(np.sqrt(np.mean((test.values - pred) ** 2)))
                        mae = float(mean_absolute_error(test.values, pred))
                        mape = safe_mape(test.values, pred)
                        ss_res = np.sum((test.values - pred) ** 2)
                        ss_tot = np.sum((test.values - np.mean(test.values)) ** 2)
                        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                        results.append({'Model': 'LSTM', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                        add_log(f"✅ LSTM complete (RMSE: {rmse:.2f})", "success")
                    
                    tf.keras.backend.clear_session()
                    gc.collect()
                else:
                    add_log("⚠️ Not enough data for LSTM given this lookback/test split", "warning")
            except Exception as e:
                add_log(f"❌ LSTM failed: {str(e)[:80]}", "error")
                try:
                    tf.keras.backend.clear_session()
                    gc.collect()
                except Exception:
                    pass
        
        # ============================================
        # Complete
        # ============================================
        set_progress(95, "Finalizing")
        
        st.session_state.results = pd.DataFrame(results)
        st.session_state.models_trained = True
        st.session_state.forecast_completed = True
        st.session_state.progress = 100
        st.session_state.current_process = "Complete"
        
        add_log(f"✅ Analysis complete! {len(results)} models trained", "success")
        
        if len(results) > 0:
            best = results[int(np.argmin([r['RMSE'] for r in results]))]
            add_log(f"🏆 Best: {best['Model']} (RMSE: {best['RMSE']:.2f})", "success")
        else:
            add_log("⚠️ No model produced valid results", "warning")
            st.session_state.forecast_error = True
        
        render_status_html("complete" if results else "error")
        
    except Exception as e:
        add_log(f"❌ Error: {str(e)}", "error")
        add_log(f"Traceback: {traceback.format_exc()}", "error")
        st.session_state.forecast_error = True
        st.session_state.progress = 0
        st.session_state.current_process = "Error"
        render_status_html("error")
    
    st.session_state.forecast_running = False

# ============================================
# TRIGGER FORECAST (FIXED)
# ============================================

# Initialize state
if "forecast_running" not in st.session_state:
    st.session_state.forecast_running = False

if "forecast_completed" not in st.session_state:
    st.session_state.forecast_completed = False

if run_button and not st.session_state.forecast_running:

    # Reset previous run
    st.session_state.forecast_completed = False
    st.session_state.forecast_error = False
    st.session_state.models_trained = False
    st.session_state.predictions = {}
    st.session_state.results = None

    # Run only once
    run_forecast(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        use_arima=use_arima,
        use_prophet=use_prophet,
        use_lstm=use_lstm,
        test_size=test_size,
        lookback=lookback,
        lstm_epochs=lstm_epochs,
    )

    # DO NOT CALL st.rerun() HERE

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; color: #888; font-size: 0.75rem; padding: 0.5rem 0;">
    <span>📊 StockForge v2.0</span>
    <span>Built with ❤️ using Streamlit</span>
    <span>📈 Real-time Market Data</span>
</div>
""", unsafe_allow_html=True)