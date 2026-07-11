"""
Stock Market Price Forecasting - Interactive Web UI
Main Entry Point
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
from src.models.lstm_model import LSTMModel

# Page configuration
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
.sub-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
SESSION_STATE_KEYS = {
    'data': None,
    'predictions': {},
    'results': None,
    'models_trained': False,
    'logs': [],
    'is_running': False,
    'forecast_count': 0
}

for key, default_value in SESSION_STATE_KEYS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

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
# BACKEND LOGIC - Run Forecast
# ============================================

if config['run_button']:
    # Clear previous logs
    clear_logs()
    st.session_state.is_running = True
    
    # Add header
    log_header("="*60)
    log_header(f"🚀 STARTING FORECAST - {config['ticker']}")
    log_info(f"📅 Date Range: {config['start_date'].strftime('%Y-%m-%d')} to {config['end_date'].strftime('%Y-%m-%d')}")
    
    models_enabled = []
    if config['use_arima']: models_enabled.append("ARIMA")
    if config['use_prophet']: models_enabled.append("Prophet")
    if config['use_lstm']: models_enabled.append("LSTM")
    log_info(f"🤖 Models: {', '.join(models_enabled)}")
    log_header("="*60)
    
    with st.spinner("🔄 Fetching data and training models..."):
        try:
            # Load data
            log_info("📊 Step 1: Loading stock data...")
            log_info(f"   Ticker: {config['ticker']}")
            
            loader = StockDataLoader(
                ticker=config['ticker'],
                start_date=config['start_date'].strftime("%Y-%m-%d"),
                end_date=config['end_date'].strftime("%Y-%m-%d")
            )
            
            with OutputCapture() as capture:
                data = loader.get_ready_data(add_indicators=True)
            
            if data is None or data.empty:
                log_error("❌ No data found. Please check the ticker symbol.")
                st.error("❌ No data found. Please check the ticker symbol.")
                st.session_state.is_running = False
                st.stop()
            
            st.session_state.data = data
            log_success(f"✅ Data loaded: {len(data)} rows")
            log_info(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            # Split data
            log_info("📊 Step 2: Splitting data into train/validation/test...")
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
                    
                    # Evaluate
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
            # Prophet
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
            if config['use_lstm']:
                log_header("="*50)
                log_header("🤖 Step 5: Training LSTM Model")
                log_header("="*50)
                
                try:
                    feature_cols = ['Close', 'SMA_20', 'RSI', 'Volume', 'Volatility']
                    
                    max_lookback = min(config['lookback'], len(data) // 5)
                    if config['lookback'] > max_lookback:
                        log_warning(f"   Reducing lookback from {config['lookback']} to {max_lookback} due to data size")
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
                        log_warning("   LSTM predictions not available - skipping evaluation")
                        
                except Exception as e:
                    log_error(f"❌ LSTM model failed: {str(e)}")
            
            # ============================================
            # Complete
            # ============================================
            log_header("="*50)
            log_header("✅ FORECAST COMPLETE!")
            log_header("="*50)
            log_success(f"🎯 {len(results)} models trained successfully")
            
            results_df = pd.DataFrame(results)
            st.session_state.results = results_df
            st.session_state.models_trained = True
            
            st.success(f"✅ Forecast complete! {len(results)} models trained successfully.")
            st.balloons()
            
            if len(results) > 0:
                best = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()]
                log_success(f"🏆 Best Model: {best['Model']} (RMSE: {best['RMSE']:.4f})")
                st.info(f"🏆 Best Model: **{best['Model']}** (RMSE: {best['RMSE']:.4f})")
            
            st.session_state.is_running = False
            st.rerun()
            
        except Exception as e:
            log_error(f"❌ Error: {str(e)}")
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            st.session_state.is_running = False

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Built with ❤️ using Streamlit | Stock Market Price Forecasting
</div>
""", unsafe_allow_html=True)