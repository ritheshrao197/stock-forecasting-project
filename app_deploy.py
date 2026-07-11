"""
Stock Market Price Forecasting - Single File Version for Streamlit Cloud
"""

import streamlit as st
# Enable caching for data loading
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker, start_date, end_date):
    """Load stock data with caching"""
    from src.data_loader import StockDataLoader
    loader = StockDataLoader(ticker=ticker, start_date=start_date, end_date=end_date)
    data = loader.get_ready_data(add_indicators=True)
    return data

# Use this in your backend logic instead of creating a new loader each time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import models
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
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# ============================================
# TICKER DATABASE
# ============================================

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

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### 📈 Configuration")
    
    # Stock Selection
    st.subheader("Stock Selection")
    all_options = get_ticker_options()
    
    search_term = st.text_input("🔍 Search", placeholder="Type to search...")
    
    if search_term:
        filtered = [o for o in all_options if search_term.lower() in o["label"].lower()]
    else:
        filtered = all_options
    
    if filtered:
        selected_label = st.selectbox(
            "Select Ticker",
            [o["label"] for o in filtered]
        )
        ticker = [o["value"] for o in filtered if o["label"] == selected_label][0]
    else:
        ticker = "AAPL"
        st.warning("No matching tickers found")
    
    # Date Range
    st.subheader("📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End", datetime.now())
    
    # Models
    st.subheader("🤖 Models")
    use_arima = st.checkbox("ARIMA", value=ARIMA_AVAILABLE, disabled=not ARIMA_AVAILABLE)
    use_prophet = st.checkbox("Prophet", value=PROPHET_AVAILABLE, disabled=not PROPHET_AVAILABLE)
    use_lstm = st.checkbox("LSTM", value=LSTM_AVAILABLE, disabled=not LSTM_AVAILABLE)
    
    if not ARIMA_AVAILABLE:
        st.caption("⚠️ ARIMA not available")
    if not PROPHET_AVAILABLE:
        st.caption("⚠️ Prophet not available")
    if not LSTM_AVAILABLE:
        st.caption("⚠️ LSTM requires TensorFlow")
    
    # Parameters
    st.subheader("⚙️ Parameters")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    
    # LSTM Parameters
    with st.expander("LSTM Parameters"):
        lookback = st.number_input("Lookback", min_value=10, max_value=120, value=60)
        lstm_epochs = st.number_input("Epochs", min_value=5, max_value=100, value=20)
    
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

st.markdown('<h1 class="main-header">📈 Stock Market Price Forecasting</h1>', unsafe_allow_html=True)
st.markdown("*Interactive dashboard for ARIMA, Prophet, and LSTM models*")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview",
    "📈 Model Results",
    "📉 Predictions",
    "📋 Model Comparison",
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
            st.metric("Price Range", f"${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        # Price Chart
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
        
        # Data Table
        with st.expander("📋 View Raw Data"):
            st.dataframe(data.tail(20), use_container_width=True)
    else:
        st.info("👈 Configure settings in the sidebar and click 'Run Forecast'")

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
                <p>RMSE: {best_model['RMSE']:.4f}</p>
                <p>MAPE: {best_model['MAPE']:.2f}%</p>
                <p>R²: {best_model['R²']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Comparison Charts
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(results['Model'], results['RMSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('RMSE Comparison')
            ax.set_ylabel('RMSE')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, results['RMSE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
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
        
        # Zoomed View
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
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        best_rmse = results.loc[results['RMSE'].idxmin()]
        
        st.info(f"""
        **📌 Best Model**
        - Model: **{best_rmse['Model']}**
        - RMSE: {best_rmse['RMSE']:.4f}
        - MAPE: {best_rmse['MAPE']:.2f}%
        - R²: {best_rmse['R²']:.4f}
        """)
        
        st.markdown("""
        ### 📚 When to Use Each Model
        
        | Model | Best For | Strengths |
        |-------|----------|-----------|
        | **ARIMA** | Short-term forecasting | Simple, interpretable, fast |
        | **Prophet** | Seasonal data | Handles seasonality, trend changes |
        | **LSTM** | Long-term, complex patterns | Captures non-linear relationships |
        """)
    else:
        st.info("👈 Run the forecast to see model comparison.")

# ============================================
# TAB 5: About
# ============================================

with tab5:
    st.markdown("""
    ### 📊 Stock Market Price Forecasting
    
    This interactive dashboard provides a comprehensive comparison of three popular time series forecasting models:
    
    #### 🤖 Models Implemented
    
    1. **ARIMA** - Classical statistical model for linear trends
    2. **Prophet** - Facebook's model for seasonal data
    3. **LSTM** - Deep learning model for complex patterns
    
    #### 🎯 Key Features
    
    - 📈 Real-time data from Yahoo Finance
    - 🔄 Interactive parameter tuning
    - 📊 Comprehensive visualizations
    - 📋 Model comparison with metrics
    - 📉 Short-term vs long-term analysis
    
    #### 📝 How to Use
    
    1. Select a ticker symbol from the dropdown
    2. Choose date range
    3. Select models to run
    4. Adjust parameters
    5. Click "Run Forecast"
    6. Explore results across tabs!
    """)

# ============================================
# BACKEND LOGIC
# ============================================

if run_button:
    st.session_state.forecast_running = True
    
    with st.spinner("🔄 Fetching data and training models..."):
        try:
            # Download data
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d")
            )
            
            if data.empty:
                st.error(f"No data found for {ticker}")
                st.session_state.forecast_running = False
                st.stop()
            
            # Add indicators
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data = data.dropna()
            
            st.session_state.data = data
            
            # Split data
            train_size = int(len(data) * (1 - test_size))
            train = data['Close'][:train_size]
            test = data['Close'][train_size:]
            
            st.session_state.predictions = {}
            results = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            status = st.empty()
            
            # ============================================
            # ARIMA
            # ============================================
            if use_arima and ARIMA_AVAILABLE:
                status.text("Training ARIMA model...")
                progress_bar.progress(20)
                
                try:
                    auto_model = auto_arima(
                        train,
                        seasonal=False,
                        stepwise=True,
                        trace=False
                    )
                    arima_model = ARIMA(train, order=auto_model.order)
                    arima_fitted = arima_model.fit()
                    arima_pred = arima_fitted.forecast(steps=len(test))
                    
                    st.session_state.predictions['ARIMA'] = (
                        arima_pred.values if hasattr(arima_pred, 'values') else arima_pred
                    )
                    
                    # Evaluate
                    rmse = np.sqrt(np.mean((test.values - arima_pred[:len(test)])**2))
                    mae = mean_absolute_error(test.values, arima_pred[:len(test)])
                    mape = np.mean(np.abs((test.values - arima_pred[:len(test)]) / test.values)) * 100
                    ss_res = np.sum((test.values - arima_pred[:len(test)])**2)
                    ss_tot = np.sum((test.values - np.mean(test.values))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    
                    results.append({
                        'Model': 'ARIMA',
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape,
                        'R²': r2
                    })
                except Exception as e:
                    st.warning(f"ARIMA failed: {str(e)[:100]}")
            
            # ============================================
            # Prophet
            # ============================================
            if use_prophet and PROPHET_AVAILABLE:
                status.text("Training Prophet model...")
                progress_bar.progress(40)
                
                try:
                    prophet_data = data.reset_index()[['Date', 'Close']]
                    prophet_data.columns = ['ds', 'y']
                    train_prophet = prophet_data[:train_size]
                    
                    prophet_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True
                    )
                    prophet_model.fit(train_prophet)
                    
                    future = prophet_model.make_future_dataframe(periods=len(test))
                    forecast = prophet_model.predict(future)
                    prophet_pred = forecast['yhat'].values[-len(test):]
                    
                    st.session_state.predictions['Prophet'] = prophet_pred
                    
                    # Evaluate
                    rmse = np.sqrt(np.mean((test.values - prophet_pred)**2))
                    mae = mean_absolute_error(test.values, prophet_pred)
                    mape = np.mean(np.abs((test.values - prophet_pred) / test.values)) * 100
                    ss_res = np.sum((test.values - prophet_pred)**2)
                    ss_tot = np.sum((test.values - np.mean(test.values))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    
                    results.append({
                        'Model': 'Prophet',
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape,
                        'R²': r2
                    })
                except Exception as e:
                    st.warning(f"Prophet failed: {str(e)[:100]}")
            
            # ============================================
            # LSTM
            # ============================================
            if use_lstm and LSTM_AVAILABLE:
                status.text("Training LSTM model...")
                progress_bar.progress(60)
                
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
                    
                    # Evaluate
                    rmse = np.sqrt(np.mean((test.values - lstm_pred[:len(test)])**2))
                    mae = mean_absolute_error(test.values, lstm_pred[:len(test)])
                    mape = np.mean(np.abs((test.values - lstm_pred[:len(test)]) / test.values)) * 100
                    ss_res = np.sum((test.values - lstm_pred[:len(test)])**2)
                    ss_tot = np.sum((test.values - np.mean(test.values))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    
                    results.append({
                        'Model': 'LSTM',
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape,
                        'R²': r2
                    })
                except Exception as e:
                    st.warning(f"LSTM failed: {str(e)[:100]}")
            
            # ============================================
            # Complete
            # ============================================
            progress_bar.progress(100)
            status.text("✅ Complete!")
            
            st.session_state.results = pd.DataFrame(results)
            st.session_state.models_trained = True
            
            st.success(f"✅ Forecast complete! {len(results)} models trained successfully.")
            st.balloons()
            
            if len(results) > 0:
                best = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()]
                st.info(f"🏆 Best Model: **{best['Model']}** (RMSE: {best['RMSE']:.4f})")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        st.session_state.forecast_running = False