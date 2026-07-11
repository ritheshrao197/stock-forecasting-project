"""
Stock Market Price Forecasting - Interactive Web UI
Main Entry Point - Fixed for Infinite Loop & Rate Limiting
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
from config.settings import APP_TITLE, APP_ICON, APP_LAYOUT

# UI Components
from ui.sidebar import render_sidebar
from ui.tabs.data_overview import render_data_overview
from ui.tabs.model_results import render_model_results
from ui.tabs.predictions import render_predictions
from ui.tabs.model_comparison import render_model_comparison
from ui.tabs.live_logs import render_live_logs
from ui.tabs.about import render_about
from ui.utils.logging_utils import (
    add_log, clear_logs, log_info, log_success, 
    log_warning, log_error, log_header
)
from ui.utils.output_capture import OutputCapture

# Models
from src.data_loader import StockDataLoader
from src.data_preprocessing import DataPreprocessor
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel

# Try to import LSTM - if not available, use a dummy
try:
    from src.models.lstm_model import LSTMModel, is_lstm_available
    LSTM_AVAILABLE = is_lstm_available()
except ImportError:
    LSTM_AVAILABLE = False

if not LSTM_AVAILABLE:
    # Create a dummy LSTM class that does nothing
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            pass
        def prepare_data(self, *args, **kwargs):
            return None, None, None, None
        def build_model(self, *args, **kwargs):
            pass
        def train(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            return np.array([])
        def evaluate(self, *args, **kwargs):
            return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        def save_model(self, *args, **kwargs):
            pass
        def plot_training_history(self):
            pass

# ============================================
# CACHING FOR RATE LIMITING
# ============================================

@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def fetch_stock_data_with_cache(ticker, start_date, end_date):
    """Fetch stock data with caching to prevent rate limiting"""
    try:
        # Add a small random delay to prevent hitting rate limits
        time.sleep(random.uniform(0.5, 1.5))
        
        loader = StockDataLoader(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        data = loader.get_ready_data(add_indicators=True)
        return data
    except Exception as e:
        # If rate limited, wait longer and retry
        if "Too Many Requests" in str(e) or "Rate limited" in str(e):
            time.sleep(random.uniform(5, 10))
            loader = StockDataLoader(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            data = loader.get_ready_data(add_indicators=True)
            return data
        raise e

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Fix sidebar scrolling */
    section[data-testid="stSidebar"] {
        height: 100vh !important;
        overflow-y: auto !important;
    }
    
    section[data-testid="stSidebar"] > div {
        height: 100% !important;
        overflow-y: auto !important;
        padding-bottom: 3rem !important;
    }
    
    /* Ensure the sidebar content is fully visible */
    .css-1d391kg {
        height: 100vh !important;
        overflow-y: auto !important;
    }
    
    /* Make scrollbar visible */
    section[data-testid="stSidebar"]::-webkit-scrollbar {
        width: 8px !important;
    }
    
    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: #666 !important;
        border-radius: 4px !important;
    }
    
    /* Hide footer */
    footer {
        display: none !important;
    }
    
    /* Compact sidebar elements */
    .stSubheader {
        margin-top: 0.5rem !important;
        margin-bottom: 0.2rem !important;
        font-size: 0.9rem !important;
    }
    
    .stMarkdown {
        margin-bottom: 0.2rem !important;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE - Initialize Only Once
# ============================================

if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'forecast_completed' not in st.session_state:
    st.session_state.forecast_completed = False
if 'rate_limit_retry_count' not in st.session_state:
    st.session_state.rate_limit_retry_count = 0

# ============================================
# SIDEBAR
# ============================================

config = render_sidebar()

# ============================================
# MAIN CONTENT
# ============================================

st.markdown(f'<h1 class="main-header">{APP_ICON} {APP_TITLE}</h1>', unsafe_allow_html=True)
st.markdown("*Interactive dashboard for ARIMA, Prophet, and LSTM models*")

# Create tabs
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
    render_data_overview(st.session_state.data, config['ticker'])

# ============================================
# TAB 2: Model Results
# ============================================

with tab2:
    render_model_results(
        st.session_state.models_trained,
        st.session_state.results,
        st.session_state.data,
        st.session_state.predictions,
        config['test_size']
    )

# ============================================
# TAB 3: Predictions
# ============================================

with tab3:
    render_predictions(
        st.session_state.models_trained,
        st.session_state.predictions,
        st.session_state.data,
        config['ticker'],
        config['test_size']
    )

# ============================================
# TAB 4: Model Comparison
# ============================================

with tab4:
    render_model_comparison(
        st.session_state.models_trained,
        st.session_state.results
    )

# ============================================
# TAB 5: Live Logs
# ============================================

with tab5:
    render_live_logs()

# ============================================
# TAB 6: About
# ============================================

with tab6:
    render_about()

# ============================================
# BACKEND LOGIC - Run Forecast (ONLY ON BUTTON CLICK)
# ============================================

def run_forecast():
    """Execute the forecast with rate limiting handling"""
    
    # Clear previous logs
    clear_logs()
    st.session_state.is_running = True
    st.session_state.forecast_completed = False
    
    # Add header
    log_header("="*60)
    log_header(f"🚀 STARTING FORECAST - {config['ticker']}")
    log_info(f"📅 Date Range: {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}")
    
    models_enabled = []
    if config['use_arima']: models_enabled.append("ARIMA")
    if config['use_prophet']: models_enabled.append("Prophet")
    if config['use_lstm'] and LSTM_AVAILABLE: models_enabled.append("LSTM")
    log_info(f"🤖 Models: {', '.join(models_enabled)}")
    if not LSTM_AVAILABLE:
        log_warning("⚠️ LSTM disabled - TensorFlow not installed")
    log_header("="*60)
    
    with st.spinner("🔄 Fetching data and training models..."):
        try:
            # ============================================
            # LOAD DATA WITH CACHING
            # ============================================
            log_info("📊 Step 1: Loading stock data...")
            log_info(f"   Ticker: {config['ticker']}")
            
            # Use cached data loader
            data = fetch_stock_data_with_cache(
                config['ticker'],
                config['start_date'].strftime("%Y-%m-%d"),
                config['end_date'].strftime("%Y-%m-%d")
            )
            
            if data is None or data.empty:
                log_error("❌ No data found. Please check the ticker symbol.")
                st.error("❌ No data found. Please check the ticker symbol.")
                st.session_state.is_running = False
                return
            
            st.session_state.data = data
            log_success(f"✅ Data loaded: {len(data)} rows")
            log_info(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            # ============================================
            # SPLIT DATA
            # ============================================
            log_info("📊 Step 2: Splitting data...")
            preprocessor = DataPreprocessor(data)
            train, val, test = preprocessor.split_data(test_size=config['test_size'], validation_size=0.1)
            log_info(f"   Training: {len(train)} days")
            log_info(f"   Validation: {len(val)} days")
            log_info(f"   Test: {len(test)} days")
            
            st.session_state.predictions = {}
            results = []
            
            # ============================================
            # ARIMA
            # ============================================
            if config['use_arima']:
                log_header("="*50)
                log_header("🤖 Step 3: Training ARIMA Model")
                log_header("="*50)
                
                try:
                    arima = ARIMAModel(train)
                    log_info("   Finding optimal ARIMA parameters...")
                    
                    with OutputCapture() as capture:
                        arima.find_best_params()
                        arima.fit()
                        arima_pred, _ = arima.predict(steps=len(test))
                    
                    st.session_state.predictions['ARIMA'] = arima_pred.values if hasattr(arima_pred, 'values') else arima_pred
                    
                    rmse = np.sqrt(np.mean((test['Close'].values - st.session_state.predictions['ARIMA'][:len(test)])**2))
                    mae = np.mean(np.abs(test['Close'].values - st.session_state.predictions['ARIMA'][:len(test)]))
                    mape = np.mean(np.abs((test['Close'].values - st.session_state.predictions['ARIMA'][:len(test)]) / test['Close'].values)) * 100
                    r2 = 1 - (np.sum((test['Close'].values - st.session_state.predictions['ARIMA'][:len(test)])**2) / 
                             np.sum((test['Close'].values - np.mean(test['Close'].values))**2))
                    
                    results.append({'Model': 'ARIMA', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                    
                    log_success(f"✅ ARIMA training complete!")
                    log_info(f"   RMSE: {rmse:.4f}")
                    log_info(f"   MAPE: {mape:.2f}%")
                    log_info(f"   R²: {r2:.4f}")
                    
                except Exception as e:
                    log_error(f"❌ ARIMA model failed: {str(e)}")
            
            # ============================================
            # PROPHET
            # ============================================
            if config['use_prophet']:
                log_header("="*50)
                log_header("🤖 Step 4: Training Prophet Model")
                log_header("="*50)
                
                try:
                    prophet_data = data.copy()
                    prophet_data.index = prophet_data.index.tz_localize(None)
                    
                    prophet = ProphetModel(prophet_data)
                    prophet.split_data(test_size=config['test_size'])
                    log_info("   Fitting Prophet model...")
                    
                    with OutputCapture() as capture:
                        prophet.fit(yearly_seasonality=True, weekly_seasonality=True)
                        prophet.predict(periods=len(test))
                    
                    prophet_pred = prophet.forecast['yhat'].values[-len(test):]
                    st.session_state.predictions['Prophet'] = prophet_pred
                    
                    rmse = np.sqrt(np.mean((test['Close'].values - prophet_pred)**2))
                    mae = np.mean(np.abs(test['Close'].values - prophet_pred))
                    mape = np.mean(np.abs((test['Close'].values - prophet_pred) / test['Close'].values)) * 100
                    r2 = 1 - (np.sum((test['Close'].values - prophet_pred)**2) / 
                             np.sum((test['Close'].values - np.mean(test['Close'].values))**2))
                    
                    results.append({'Model': 'Prophet', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                    
                    log_success(f"✅ Prophet training complete!")
                    log_info(f"   RMSE: {rmse:.4f}")
                    log_info(f"   MAPE: {mape:.2f}%")
                    log_info(f"   R²: {r2:.4f}")
                    
                except Exception as e:
                    log_error(f"❌ Prophet model failed: {str(e)}")
            
            # ============================================
            # LSTM
            # ============================================
            if config['use_lstm'] and LSTM_AVAILABLE:
                log_header("="*50)
                log_header("🤖 Step 5: Training LSTM Model")
                log_header("="*50)
                
                try:
                    feature_cols = ['Close', 'SMA_20', 'RSI', 'Volume', 'Volatility']
                    
                    max_lookback = min(config['lookback'], len(data) // 5)
                    if config['lookback'] > max_lookback:
                        log_warning(f"   Reducing lookback from {config['lookback']} to {max_lookback}")
                        lookback_actual = max_lookback
                    else:
                        lookback_actual = config['lookback']
                    
                    units_list = eval(config['lstm_units'])
                    log_info(f"   Architecture: {units_list} units")
                    log_info(f"   Lookback: {lookback_actual} days")
                    log_info(f"   Epochs: {config['lstm_epochs']}")
                    
                    lstm = LSTMModel(lookback=lookback_actual, n_features=len(feature_cols))
                    log_info("   Preparing data for LSTM...")
                    
                    with OutputCapture() as capture:
                        lstm.prepare_data(data, target_column='Close', 
                                        feature_columns=feature_cols, test_size=config['test_size'])
                        lstm.build_model(lstm_units=units_list, dropout_rate=0.2)
                        log_info("   Training LSTM neural network...")
                        lstm.train(epochs=config['lstm_epochs'], batch_size=16, validation_split=0.1)
                    
                    lstm_pred = lstm.predict(inverse_transform=True)
                    
                    if len(lstm_pred) > 0:
                        if len(lstm_pred) < len(test):
                            lstm_pred = np.pad(lstm_pred, (0, len(test) - len(lstm_pred)), mode='edge')
                        elif len(lstm_pred) > len(test):
                            lstm_pred = lstm_pred[:len(test)]
                        
                        st.session_state.predictions['LSTM'] = lstm_pred
                        
                        rmse = np.sqrt(np.mean((test['Close'].values - lstm_pred)**2))
                        mae = np.mean(np.abs(test['Close'].values - lstm_pred))
                        mape = np.mean(np.abs((test['Close'].values - lstm_pred) / test['Close'].values)) * 100
                        r2 = 1 - (np.sum((test['Close'].values - lstm_pred)**2) / 
                                 np.sum((test['Close'].values - np.mean(test['Close'].values))**2))
                        
                        results.append({'Model': 'LSTM', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                        
                        log_success(f"✅ LSTM training complete!")
                        log_info(f"   RMSE: {rmse:.4f}")
                        log_info(f"   MAPE: {mape:.2f}%")
                        log_info(f"   R²: {r2:.4f}")
                    else:
                        log_warning("   LSTM predictions not available")
                        
                except Exception as e:
                    log_error(f"❌ LSTM model failed: {str(e)}")
            elif config['use_lstm'] and not LSTM_AVAILABLE:
                log_warning("⚠️ LSTM skipped - TensorFlow not installed")
            
            # ============================================
            # COMPLETE
            # ============================================
            log_header("="*50)
            log_header("✅ FORECAST COMPLETE!")
            log_header("="*50)
            log_success(f"🎯 {len(results)} models trained successfully")
            
            results_df = pd.DataFrame(results)
            st.session_state.results = results_df
            st.session_state.models_trained = True
            st.session_state.forecast_completed = True
            
            st.success(f"✅ Forecast complete! {len(results)} models trained successfully.")
            st.balloons()
            
            if len(results) > 0:
                best = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()]
                log_success(f"🏆 Best Model: {best['Model']} (RMSE: {best['RMSE']:.4f})")
                st.info(f"🏆 Best Model: **{best['Model']}** (RMSE: {best['RMSE']:.4f})")
            
            st.session_state.is_running = False
            
        except Exception as e:
            error_msg = str(e)
            log_error(f"❌ Error: {error_msg}")
            
            # Check if it's a rate limiting error
            if "Too Many Requests" in error_msg or "Rate limited" in error_msg:
                st.error("⏳ Rate limit exceeded. Please wait a few minutes and try again.")
                st.info("💡 Tip: Try selecting a different ticker or reduce the date range.")
                st.session_state.rate_limit_retry_count += 1
            else:
                st.error(f"❌ Error: {error_msg}")
                st.exception(e)
            
            st.session_state.is_running = False
            st.session_state.forecast_completed = False

# ============================================
# TRIGGER FORECAST
# ============================================

if config['run_button'] and not st.session_state.is_running and not st.session_state.forecast_completed:
    run_forecast()
    st.rerun()

# ============================================
# STATUS MESSAGES
# ============================================

if st.session_state.forecast_completed and st.session_state.models_trained:
    st.success("✅ Forecast completed! Results are available in the tabs above.")

if st.session_state.get('rate_limit_retry_count', 0) > 2:
    st.warning("⚠️ Multiple rate limit errors detected. Please wait a few minutes before trying again.")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Built with ❤️ using Streamlit | Stock Market Price Forecasting
</div>
""", unsafe_allow_html=True)