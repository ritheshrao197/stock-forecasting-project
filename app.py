"""
Stock Market Price Forecasting - Interactive Web UI
Indian Stocks Version 🇮🇳
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Market Forecasting - Indian Stocks",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
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
    </style>
""", unsafe_allow_html=True)

# Import models
from src.data_loader import StockDataLoader
from src.data_preprocessing import DataPreprocessor
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.model_comparison import ModelComparison

# ============================================
# HELPER FUNCTIONS
# ============================================

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
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc."
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
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'forecast_completed' not in st.session_state:
    st.session_state.forecast_completed = False

# ============================================
# SIDEBAR - Configuration
# ============================================

with st.sidebar:
    # Header with Indian badge
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 0.5rem;">
        <span style="font-size: 2rem;">📈</span>
        <span style="font-size: 1.1rem; font-weight: bold;">Configuration</span>
        <span class="india-badge">🇮🇳 INDIA</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock Selection
    st.subheader("📈 Stock Selection")
    
    search_term = st.text_input("🔍 Search Ticker", placeholder="Type company name or symbol...", key="ticker_search")
    
    all_options = get_ticker_options()
    
    if search_term:
        filtered_options = [
            opt for opt in all_options 
            if search_term.lower() in opt["label"].lower() or 
               search_term.lower() in opt["value"].lower() or
               search_term.lower() in opt["category"].lower()
        ]
    else:
        filtered_options = all_options
    
    if not filtered_options:
        filtered_options = all_options
        st.warning("No matching tickers found.")
    
    ticker_options = [opt["label"] for opt in filtered_options]
    ticker_values = [opt["value"] for opt in filtered_options]
    
    # Default to Reliance (Indian stock)
    default_index = 0
    for i, opt in enumerate(filtered_options):
        if opt["value"] == "RELIANCE.NS":
            default_index = i
            break
    
    selected_label = st.selectbox(
        "Select Ticker Symbol",
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
    
    # Show exchange info
    st.caption(f"🏛️ {get_exchange_info(ticker)}")
    
    # Custom ticker
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
        start_date = st.date_input(
            "Start",
            value=datetime(2020, 1, 1),
            max_value=datetime.now(),
            key="start_date"
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=datetime.now(),
            max_value=datetime.now(),
            key="end_date"
        )
    
    # Quick presets
    st.subheader("⚡ Quick Presets")
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
    
    # Model Selection
    st.subheader("🤖 Models")
    use_arima = st.checkbox("ARIMA", value=True, key="use_arima")
    use_prophet = st.checkbox("Prophet", value=True, key="use_prophet")
    use_lstm = st.checkbox("LSTM", value=True, key="use_lstm")
    
    # Parameters
    st.subheader("⚙️ Parameters")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="test_size")
    
    # LSTM Parameters
    with st.expander("🧠 LSTM Parameters"):
        lookback = st.number_input("Lookback Period", min_value=10, max_value=120, value=60, key="lookback")
        lstm_epochs = st.number_input("Epochs", min_value=5, max_value=100, value=20, key="lstm_epochs")
        lstm_units = st.selectbox("Hidden Units", ["[100, 50]", "[50, 25]", "[100, 50, 25]"], index=1, key="lstm_units")
    
    # Action Buttons
    st.markdown("---")
    
    is_running = st.session_state.get('is_running', False)
    forecast_completed = st.session_state.get('forecast_completed', False)
    
    run_button = st.button(
        "🚀 Run Forecast", 
        use_container_width=True,
        disabled=is_running,
        type="primary" if not is_running else "secondary"
    )
    
    if is_running:
        st.info("⏳ Running... Please wait.")
    elif forecast_completed:
        st.success("✅ Completed! Rerun to update.")
    
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.data = None
        st.session_state.predictions = {}
        st.session_state.results = None
        st.session_state.models_trained = False
        st.session_state.logs = []
        st.session_state.is_running = False
        st.session_state.forecast_completed = False
        st.rerun()
    
    # Version info
    st.caption("📌 v1.0 | 🇮🇳 Indian Stocks | ARIMA · Prophet · LSTM")

# ============================================
# MAIN CONTENT
# ============================================

st.markdown('<h1 class="main-header">🇮🇳 Indian Stock Market Price Forecasting</h1>', unsafe_allow_html=True)
st.markdown("*Interactive dashboard for ARIMA, Prophet, and LSTM models - Now with 100+ Indian stocks*")

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
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(data))
        with col2:
            st.metric("Start Date", data.index[0].strftime("%Y-%m-%d"))
        with col3:
            st.metric("End Date", data.index[-1].strftime("%Y-%m-%d"))
        with col4:
            st.metric("Price Range", f"₹{data['Close'].min():.2f} - ₹{data['Close'].max():.2f}")
        
        # Price chart
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
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume and RSI
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Trading Volume")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            fig2.update_layout(height=250, xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            if 'RSI' in data.columns:
                st.subheader("📊 RSI Indicator")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig3.update_layout(height=250, xaxis_title="Date", yaxis_title="RSI")
                st.plotly_chart(fig3, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(data.tail(20), use_container_width=True)
    
    else:
        st.info("👈 Configure settings in the sidebar and click 'Run Forecast' to get started.")
        
        # Sample preview - Show Indian stock sample
        st.subheader("📊 Sample Data Preview")
        st.caption("Select an Indian stock and click 'Run Forecast' to load data.")
        
        # Show sample chart for NIFTY 50
        sample_data = yf.download("^NSEI", period="1y", progress=False)
        if not sample_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_data.index,
                y=sample_data['Close'],
                mode='lines',
                name='NIFTY 50 (Sample)',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(height=300, title="Sample: NIFTY 50 - Last Year")
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2: Model Results
# ============================================

with tab2:
    if st.session_state.models_trained and st.session_state.results is not None:
        results = st.session_state.results
        
        st.subheader("📊 Model Performance Comparison")
        
        # Metrics table
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                results.style.background_gradient(subset=['RMSE', 'MAPE'], cmap='RdYlGn_r'),
                use_container_width=True
            )
        
        with col2:
            # Best model
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
        
        # Comparison charts
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(results['Model'], results['RMSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('RMSE Comparison', fontsize=14)
            ax.set_ylabel('RMSE (₹)')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, results['RMSE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'₹{value:.2f}', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(results['Model'], results['MAPE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('MAPE Comparison', fontsize=14)
            ax.set_ylabel('MAPE (%)')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, results['MAPE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                       f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
        
        # Short-term vs Long-term
        st.subheader("📊 Short-term vs Long-term Performance")
        
        # Calculate short and long term metrics
        data = st.session_state.data
        train_size = int(len(data) * (1 - test_size))
        test_data = data['Close'][train_size:]
        
        short_metrics = {}
        long_metrics = {}
        
        for name, pred in st.session_state.predictions.items():
            if len(pred) > 0:
                short_len = min(10, len(pred))
                long_len = min(30, len(pred))
                
                short_rmse = np.sqrt(np.mean((test_data.values[:short_len] - pred[:short_len])**2))
                long_rmse = np.sqrt(np.mean((test_data.values[:long_len] - pred[:long_len])**2))
                
                short_metrics[name] = short_rmse
                long_metrics[name] = long_rmse
        
        if short_metrics:
            df_compare = pd.DataFrame({
                'Model': list(short_metrics.keys()),
                'Short-term (10 days)': list(short_metrics.values()),
                'Long-term (30 days)': list(long_metrics.values())
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Short-term (10 days)', x=df_compare['Model'], y=df_compare['Short-term (10 days)'], marker_color='#1f77b4'))
            fig.add_trace(go.Bar(name='Long-term (30 days)', x=df_compare['Model'], y=df_compare['Long-term (30 days)'], marker_color='#ff7f0e'))
            fig.update_layout(
                title='RMSE: Short-term vs Long-term',
                xaxis_title='Model',
                yaxis_title='RMSE (₹)',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            best_short = min(short_metrics.items(), key=lambda x: x[1])
            best_long = min(long_metrics.items(), key=lambda x: x[1])
            
            st.success(f"""
            **📌 Key Insights:**
            - Best for **Short-term** forecasting: **{best_short[0]}** (RMSE: ₹{best_short[1]:.2f})
            - Best for **Long-term** forecasting: **{best_long[0]}** (RMSE: ₹{best_long[1]:.2f})
            """)
    
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
        
        # Interactive plot with model selection
        selected_models = st.multiselect(
            "Select models to display",
            options=list(st.session_state.predictions.keys()),
            default=list(st.session_state.predictions.keys())
        )
        
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        # Add predictions
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
            yaxis_title='Price (₹)',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Zoomed view
        st.subheader("🔍 Zoomed View (Last 50 Days)")
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
            yaxis_title='Price (₹)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Error analysis
        with st.expander("📊 Error Analysis"):
            for name, pred in st.session_state.predictions.items():
                if len(pred) > 0:
                    errors = test_data.values[:len(pred)] - pred[:len(test_data)]
                    fig3, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(range(len(errors)), errors, label=f'{name} Errors')
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax.axhline(y=np.mean(errors), color='red', linestyle='--', label='Mean Error')
                    ax.set_title(f'{name} - Prediction Errors')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Error (₹)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig3)
    
    else:
        st.info("👈 Run the forecast to see predictions.")

# ============================================
# TAB 4: Model Comparison
# ============================================

with tab4:
    if st.session_state.models_trained and st.session_state.results is not None:
        st.subheader("📋 Detailed Model Comparison")
        
        # Show all metrics in a nice table
        results = st.session_state.results
        st.dataframe(
            results.style.background_gradient(subset=['RMSE', 'MAE', 'MAPE'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        # Radar chart
        st.subheader("📊 Model Performance Radar")
        
        # Normalize metrics for radar
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
        df_radar = results[['Model'] + metrics_to_plot].copy()
        
        # Normalize
        for metric in metrics_to_plot:
            max_val = df_radar[metric].max()
            if max_val > 0:
                df_radar[f'{metric}_norm'] = 1 - (df_radar[metric] / max_val)
        
        # Create radar chart
        fig = go.Figure()
        
        for model in df_radar['Model']:
            model_data = df_radar[df_radar['Model'] == model]
            fig.add_trace(go.Scatterpolar(
                r=[model_data[f'{m}_norm'].values[0] for m in metrics_to_plot],
                theta=metrics_to_plot,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='Model Performance Radar (Higher is Better)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        best_rmse = results.loc[results['RMSE'].idxmin()]
        best_mape = results.loc[results['MAPE'].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **📌 Best Model (RMSE)**
            - Model: **{best_rmse['Model']}**
            - RMSE: ₹{best_rmse['RMSE']:.2f}
            - MAPE: {best_rmse['MAPE']:.2f}%
            - R²: {best_rmse['R²']:.4f}
            """)
        with col2:
            st.success(f"""
            **🎯 Best Model (MAPE)**
            - Model: **{best_mape['Model']}**
            - RMSE: ₹{best_mape['RMSE']:.2f}
            - MAPE: {best_mape['MAPE']:.2f}%
            - R²: {best_mape['R²']:.4f}
            """)
        
        # Usage recommendations
        st.markdown("""
        ### 📚 When to Use Each Model
        
        | Model | Best For | Strengths | Limitations |
        |-------|----------|-----------|-------------|
        | **ARIMA** | Short-term forecasting | Simple, interpretable, fast | Struggles with non-linear patterns |
        | **Prophet** | Seasonal data | Handles seasonality, missing data | Less accurate for chaotic markets |
        | **LSTM** | Long-term, complex patterns | Captures non-linear relationships | Requires more data, computationally heavy |
        """)
    
    else:
        st.info("👈 Run the forecast to see model comparison.")

# ============================================
# TAB 5: Live Logs
# ============================================

with tab5:
    st.subheader("📝 Live Terminal Logs")
    
    # Log controls
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🔄 Refresh Logs"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Logs"):
            st.session_state.logs = []
            st.rerun()
    with col3:
        st.caption(f"📊 {len(st.session_state.logs)} log entries")
    
    # Auto-refresh checkbox
    auto_refresh = st.checkbox("🔄 Auto-refresh logs", value=True)
    if auto_refresh:
        st.caption("Logs will auto-refresh every 3 seconds")
        st.empty()
    
    # Render logs
    logs = st.session_state.logs
    if logs:
        log_html = ""
        for log in logs:
            log_html += f'<div><span style="color: #888;">[{log["timestamp"]}]</span> <span style="color: #4fc3f7;">{log["message"]}</span></div>'
        
        st.markdown(f"""
        <div style="
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
        ">
            {log_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📝 No logs yet. Run the forecast to see terminal output.")
    
    # Status indicator
    if st.session_state.is_running:
        st.warning("⏳ Process is currently running...")
    else:
        st.success("✅ Process idle")

# ============================================
# TAB 6: About
# ============================================

with tab6:
    st.markdown("""
    <h2 style="font-size: 1.5rem; color: #2c3e50; margin-top: 1rem;">🇮🇳 Indian Stock Market Forecasting</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 About This Project
    
    This interactive dashboard provides a comprehensive comparison of three popular time series forecasting models for **Indian stocks**.
    
    #### 🤖 Models Implemented
    
    1. **ARIMA** - Classical statistical model for linear trends       - Best for: Short-term forecasting
       - Strengths: Simple, interpretable, fast
    
    2. **Prophet** - Facebook's model for seasonal data
       - Best for: Seasonal patterns and trend changes
       - Strengths: Handles missing data, robust to outliers
    
    3. **LSTM** - Deep learning model for complex patterns
       - Best for: Long-term, complex patterns
       - Strengths: Captures non-linear relationships
    
    #### 🇮🇳 Supported Indian Stocks
    
    **Indices:**
    - NIFTY 50 (^NSEI)
    - SENSEX (^BSESN)
    - NIFTY BANK (^NSEBANK)
    - NIFTY IT (^CNXIT)
    - NIFTY PHARMA (^CNXPHARMA)
    - NIFTY FMCG (^CNXFMCG)
    - NIFTY AUTO (^CNXAUTO)
    
    **Top Companies:**
    - Reliance Industries (RELIANCE.NS)
    - TCS (TCS.NS)
    - Infosys (INFY.NS)
    - HDFC Bank (HDFCBANK.NS)
    - ICICI Bank (ICICIBANK.NS)
    - ITC (ITC.NS)
    - Hindustan Unilever (HINDUNILVR.NS)
    - State Bank of India (SBIN.NS)
    - Bharti Airtel (BHARTIARTL.NS)
    - Tata Motors (TATAMOTORS.NS)
    
    #### 🎯 Key Features
    
    - 📈 Real-time data from Yahoo Finance
    - 🔄 Interactive parameter tuning
    - 📊 Comprehensive visualizations
    - 📋 Model comparison with metrics
    - 📉 Short-term vs long-term analysis
    - 🔍 Searchable ticker dropdown with 100+ Indian stocks
    - 📝 Live terminal logs during processing
    
    #### 📝 How to Use
    
    1. Search and select a ticker symbol from the dropdown
    2. Select date range (or use quick presets)
    3. Choose models to run
    4. Adjust parameters
    5. Click "Run Forecast"
    6. Watch live logs in the "Live Logs" tab
    7. Explore results across other tabs!
    
    #### 📚 Technology Stack
    
    - **Frontend**: Streamlit
    - **Data**: yfinance, pandas
    - **Models**: statsmodels, prophet, tensorflow
    - **Visualization**: plotly, matplotlib
    
    #### 📖 Learn More
    
    - [NSE India](https://www.nseindia.com/)
    - [BSE India](https://www.bseindia.com/)
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [Prophet Documentation](https://facebook.github.io/prophet/)
    - [TensorFlow Documentation](https://www.tensorflow.org/)
    """)
    
    # Version info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Python", "3.10+")
    with col2:
        st.metric("Models", "3")
    with col3:
        st.metric("Indian Stocks", "100+")

# ============================================
# BACKEND LOGIC - Run Forecast
# ============================================

if run_button and not st.session_state.is_running and not st.session_state.forecast_completed:
    st.session_state.is_running = True
    st.session_state.forecast_completed = False
    
    # Clear previous logs
    st.session_state.logs = []
    
    # Add initial log
    from datetime import datetime
    st.session_state.logs.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "message": f"🚀 Starting forecast for {ticker}"
    })
    st.session_state.logs.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "message": f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    })
    
    with st.spinner("🔄 Fetching data and training models..."):
        try:
            # Load data
            st.session_state.logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": "📊 Step 1: Loading stock data..."
            })
            
            loader = StockDataLoader(
                ticker=ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            data = loader.get_ready_data(add_indicators=True)
            
            if data is None or data.empty:
                st.error(f"❌ No data found for {ticker}")
                st.session_state.is_running = False
                st.stop()
            
            st.session_state.data = data
            st.session_state.logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": f"✅ Data loaded: {len(data)} rows"
            })
            
            # Split data
            st.session_state.logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": "📊 Step 2: Splitting data..."
            })
            
            preprocessor = DataPreprocessor(data)
            train, val, test = preprocessor.split_data(test_size=test_size, validation_size=0.1)
            
            st.session_state.logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": f"   Training: {len(train)} days"
            })
            st.session_state.logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": f"   Validation: {len(val)} days"
            })
            st.session_state.logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": f"   Test: {len(test)} days"
            })
            
            st.session_state.predictions = {}
            results = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            
            # ============================================
            # ARIMA
            # ============================================
            if use_arima:
                st.session_state.logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": "🤖 Step 3: Training ARIMA Model"
                })
                progress_bar.progress(20)
                
                try:
                    arima = ARIMAModel(train)
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": "   Finding optimal ARIMA parameters..."
                    })
                    
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
                    
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": f"✅ ARIMA complete! RMSE: ₹{rmse:.2f}, MAPE: {mape:.2f}%"
                    })
                except Exception as e:
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": f"❌ ARIMA failed: {str(e)[:100]}"
                    })
            
            # ============================================
            # Prophet
            # ============================================
            if use_prophet:
                st.session_state.logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": "🤖 Step 4: Training Prophet Model"
                })
                progress_bar.progress(50)
                
                try:
                    prophet_data = data.copy()
                    prophet_data.index = prophet_data.index.tz_localize(None)
                    
                    prophet = ProphetModel(prophet_data)
                    prophet.split_data(test_size=test_size)
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": "   Fitting Prophet model..."
                    })
                    
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
                    
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": f"✅ Prophet complete! RMSE: ₹{rmse:.2f}, MAPE: {mape:.2f}%"
                    })
                except Exception as e:
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": f"❌ Prophet failed: {str(e)[:100]}"
                    })
            
            # ============================================
            # LSTM
            # ============================================
            if use_lstm:
                st.session_state.logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": "🤖 Step 5: Training LSTM Model"
                })
                progress_bar.progress(75)
                
                try:
                    feature_cols = ['Close', 'SMA_20', 'RSI', 'Volume', 'Volatility']
                    
                    max_lookback = min(lookback, len(data) // 5)
                    if lookback > max_lookback:
                        st.session_state.logs.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "message": f"   Reducing lookback from {lookback} to {max_lookback}"
                        })
                        lookback_actual = max_lookback
                    else:
                        lookback_actual = lookback
                    
                    units_list = eval(lstm_units)
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": f"   Architecture: {units_list} units, Lookback: {lookback_actual} days"
                    })
                    
                    lstm = LSTMModel(lookback=lookback_actual, n_features=len(feature_cols))
                    lstm.prepare_data(data, target_column='Close', 
                                    feature_columns=feature_cols, test_size=test_size)
                    lstm.build_model(lstm_units=units_list, dropout_rate=0.2)
                    
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": "   Training LSTM neural network..."
                    })
                    
                    lstm.train(epochs=lstm_epochs, batch_size=16, validation_split=0.1)
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
                        
                        st.session_state.logs.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "message": f"✅ LSTM complete! RMSE: ₹{rmse:.2f}, MAPE: {mape:.2f}%"
                        })
                    else:
                        st.session_state.logs.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "message": "⚠️ LSTM predictions not available"
                        })
                except Exception as e:
                    st.session_state.logs.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": f"❌ LSTM failed: {str(e)[:100]}"
                    })
            
            # ============================================
            # Complete
            # ============================================
            progress_bar.progress(100)
            
            st.session_state.logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": f"✅ Forecast complete! {len(results)} models trained successfully."
            })
            
            results_df = pd.DataFrame(results)
            st.session_state.results = results_df
            st.session_state.models_trained = True
            st.session_state.forecast_completed = True
            
            st.success(f"✅ Forecast complete! {len(results)} models trained successfully.")
            st.balloons()
            
            if len(results) > 0:
                best = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()]
                st.session_state.logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": f"🏆 Best Model: {best['Model']} (RMSE: ₹{best['RMSE']:.2f})"
                })
                st.info(f"🏆 Best Model: **{best['Model']}** (RMSE: ₹{best['RMSE']:.2f})")
            
            st.session_state.is_running = False
            st.rerun()
            
        except Exception as e:
            st.session_state.logs.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": f"❌ Error: {str(e)}"
            })
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            st.session_state.is_running = False
            st.session_state.forecast_completed = False

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Built with ❤️ using Streamlit | 🇮🇳 Indian Stock Market Price Forecasting
</div>
""", unsafe_allow_html=True)