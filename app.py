"""
Stock Market Price Forecasting - Interactive Web UI
Built with Streamlit
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
    page_title="Stock Market Forecasting",
    page_icon="📈",
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
        margin-bottom: 1rem;
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
    </style>
""", unsafe_allow_html=True)

# Import models
from src.data_loader import StockDataLoader
from src.data_preprocessing import DataPreprocessor
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.model_comparison import ModelComparison

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# ============================================
# SIDEBAR - Configuration
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stock.png", width=80)
    st.title("📊 Configuration")
    
    # Stock selection
    st.subheader("📈 Stock Selection")
    ticker = st.text_input("Ticker Symbol", value="AAPL", help="e.g., AAPL, MSFT, ^GSPC")
    
    # Date range
    st.subheader("📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Model selection
    st.subheader("🤖 Models")
    use_arima = st.checkbox("ARIMA", value=True)
    use_prophet = st.checkbox("Prophet", value=True)
    use_lstm = st.checkbox("LSTM", value=True)
    
    # Parameters
    st.subheader("⚙️ Parameters")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    
    # LSTM parameters
    with st.expander("LSTM Parameters"):
        lookback = st.number_input("Lookback Period", min_value=10, max_value=120, value=60)
        lstm_epochs = st.number_input("Epochs", min_value=5, max_value=100, value=20)
        lstm_units = st.selectbox("Hidden Units", ["[100, 50]", "[50, 25]", "[100, 50, 25]"], index=1)
    
    # Run button
    st.markdown("---")
    run_button = st.button("🚀 Run Forecast", use_container_width=True)
    
    # Clear button
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.data = None
        st.session_state.predictions = {}
        st.session_state.results = None
        st.session_state.models_trained = False
        st.rerun()

# ============================================
# MAIN CONTENT
# ============================================

st.markdown('<h1 class="main-header">📈 Stock Market Price Forecasting</h1>', unsafe_allow_html=True)
st.markdown("*Interactive dashboard for ARIMA, Prophet, and LSTM models*")

# Create tabs
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
        
        # Data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(data))
        with col2:
            st.metric("Start Date", data.index[0].strftime("%Y-%m-%d"))
        with col3:
            st.metric("End Date", data.index[-1].strftime("%Y-%m-%d"))
        with col4:
            st.metric("Price Range", f"${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
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
            yaxis_title="Price ($)",
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
        
        # Sample preview
        st.subheader("📊 Sample Data Preview")
        st.caption("Enter a ticker symbol and click 'Run Forecast' to load data.")
        
        # Show sample chart
        sample_data = yf.download("AAPL", period="1y", progress=False)
        if not sample_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_data.index,
                y=sample_data['Close'],
                mode='lines',
                name='AAPL (Sample)',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(height=300, title="Sample: Apple Inc. (AAPL) - Last Year")
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
                <p>RMSE: {best_model['RMSE']:.4f}</p>
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
            ax.set_ylabel('RMSE')
            ax.grid(True, alpha=0.3)
            # Add values on bars
            for bar, value in zip(bars, results['RMSE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
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
                yaxis_title='RMSE',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            best_short = min(short_metrics.items(), key=lambda x: x[1])
            best_long = min(long_metrics.items(), key=lambda x: x[1])
            
            st.success(f"""
            **📌 Key Insights:**
            - Best for **Short-term** forecasting: **{best_short[0]}** (RMSE: {best_short[1]:.4f})
            - Best for **Long-term** forecasting: **{best_long[0]}** (RMSE: {best_long[1]:.4f})
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
            yaxis_title='Price ($)',
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
            yaxis_title='Price ($)',
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
                    ax.set_ylabel('Error')
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
                df_radar[f'{metric}_norm'] = 1 - (df_radar[metric] / max_val)  # Higher is better
        
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
            - RMSE: {best_rmse['RMSE']:.4f}
            - MAPE: {best_rmse['MAPE']:.2f}%
            - R²: {best_rmse['R²']:.4f}
            """)
        with col2:
            st.success(f"""
            **🎯 Best Model (MAPE)**
            - Model: **{best_mape['Model']}**
            - RMSE: {best_mape['RMSE']:.4f}
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
# TAB 5: About
# ============================================

with tab5:
    st.markdown("""
    <h2 class="sub-header">ℹ️ About This Project</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Stock Market Price Forecasting
    
    This interactive dashboard provides a comprehensive comparison of three popular time series forecasting models:
    
    #### 🤖 Models Implemented
    
    1. **ARIMA** (AutoRegressive Integrated Moving Average)
       - Classical statistical model
       - Best for linear trends and short-term forecasts
       - Fast and interpretable
    
    2. **Prophet** (by Meta/Facebook)
       - Designed for business time series
       - Handles seasonality and trend changes
       - Robust to missing data
    
    3. **LSTM** (Long Short-Term Memory)
       - Deep learning neural network
       - Captures complex, non-linear patterns
       - Best for long-term forecasts
    
    #### 🎯 Key Features
    
    - 📈 Real-time data from Yahoo Finance
    - 🔄 Interactive parameter tuning
    - 📊 Comprehensive visualizations
    - 📋 Model comparison with metrics
    - 📉 Short-term vs long-term analysis
    
    #### 📚 Technology Stack
    
    - **Frontend**: Streamlit
    - **Data**: yfinance, pandas
    - **Models**: statsmodels, prophet, tensorflow
    - **Visualization**: plotly, matplotlib
    
    #### 📝 How to Use
    
    1. Enter a stock ticker (e.g., AAPL, MSFT, ^GSPC)
    2. Select date range
    3. Choose models to run
    4. Adjust parameters
    5. Click "Run Forecast"
    6. Explore results across tabs!
    
    #### 📖 Learn More
    
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [Prophet Documentation](https://facebook.github.io/prophet/)
    - [TensorFlow Documentation](https://www.tensorflow.org/)
    - [Statsmodels Documentation](https://www.statsmodels.org/)
    """)
    
    # Version info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Python", "3.10+")
    with col2:
        st.metric("Models", "3")
    with col3:
        st.metric("Data Sources", "Yahoo Finance")

# ============================================
# BACKEND LOGIC - Run Forecast
# ============================================

if run_button:
    with st.spinner("🔄 Fetching data and training models..."):
        try:
            # Load data
            loader = StockDataLoader(
                ticker=ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            data = loader.get_ready_data(add_indicators=True)
            
            if data is None or data.empty:
                st.error("❌ No data found. Please check the ticker symbol.")
                st.stop()
            
            st.session_state.data = data
            
            # Split data
            preprocessor = DataPreprocessor(data)
            train, val, test = preprocessor.split_data(test_size=test_size, validation_size=0.1)
            
            st.session_state.predictions = {}
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ============================================
            # ARIMA
            # ============================================
            if use_arima:
                status_text.text("Training ARIMA model...")
                progress_bar.progress(20)
                
                arima = ARIMAModel(train)
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
            
            # ============================================
            # Prophet
            # ============================================
            if use_prophet:
                status_text.text("Training Prophet model...")
                progress_bar.progress(40)
                
                # Fix timezone issue
                prophet_data = data.copy()
                prophet_data.index = prophet_data.index.tz_localize(None)
                
                prophet = ProphetModel(prophet_data)
                prophet.split_data(test_size=test_size)
                prophet.fit(yearly_seasonality=True, weekly_seasonality=True)
                prophet.predict(periods=len(test))
                
                # Get predictions
                prophet_pred = prophet.forecast['yhat'].values[-len(test):]
                st.session_state.predictions['Prophet'] = prophet_pred
                
                # Evaluate
                rmse = np.sqrt(np.mean((test['Close'].values - prophet_pred)**2))
                mae = np.mean(np.abs(test['Close'].values - prophet_pred))
                mape = np.mean(np.abs((test['Close'].values - prophet_pred) / test['Close'].values)) * 100
                r2 = 1 - (np.sum((test['Close'].values - prophet_pred)**2) / 
                         np.sum((test['Close'].values - np.mean(test['Close'].values))**2))
                
                results.append({'Model': 'Prophet', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
            
            # ============================================
            # LSTM
            # ============================================
            if use_lstm:
                status_text.text("Training LSTM model...")
                progress_bar.progress(60)
                
                try:
                    feature_cols = ['Close', 'SMA_20', 'RSI', 'Volume', 'Volatility']
                    
                    # Adjust lookback if needed
                    max_lookback = min(lookback, len(data) // 5)
                    if lookback > max_lookback:
                        st.warning(f"Reducing lookback from {lookback} to {max_lookback} due to data size")
                        lookback_actual = max_lookback
                    else:
                        lookback_actual = lookback
                    
                    # Parse LSTM units
                    units_list = eval(lstm_units)
                    
                    lstm = LSTMModel(lookback=lookback_actual, n_features=len(feature_cols))
                    lstm.prepare_data(data, target_column='Close', 
                                    feature_columns=feature_cols, test_size=test_size)
                    lstm.build_model(lstm_units=units_list, dropout_rate=0.2)
                    lstm.train(epochs=lstm_epochs, batch_size=16, validation_split=0.1)
                    
                    # Get predictions
                    lstm_pred = lstm.predict(inverse_transform=True)
                    
                    # Align predictions with test data
                    if len(lstm_pred) > 0:
                        # Pad or trim to match test size
                        if len(lstm_pred) < len(test):
                            lstm_pred = np.pad(lstm_pred, (0, len(test) - len(lstm_pred)), mode='edge')
                        elif len(lstm_pred) > len(test):
                            lstm_pred = lstm_pred[:len(test)]
                        
                        st.session_state.predictions['LSTM'] = lstm_pred
                        
                        # Evaluate
                        rmse = np.sqrt(np.mean((test['Close'].values - lstm_pred)**2))
                        mae = np.mean(np.abs(test['Close'].values - lstm_pred))
                        mape = np.mean(np.abs((test['Close'].values - lstm_pred) / test['Close'].values)) * 100
                        r2 = 1 - (np.sum((test['Close'].values - lstm_pred)**2) / 
                                 np.sum((test['Close'].values - np.mean(test['Close'].values))**2))
                        
                        results.append({'Model': 'LSTM', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
                    else:
                        st.warning("LSTM predictions not available - skipping evaluation")
                        
                except Exception as e:
                    st.warning(f"LSTM model failed: {str(e)}")
            
            # ============================================
            # Complete
            # ============================================
            progress_bar.progress(100)
            status_text.text("✅ Complete!")

            # Store results
            results_df = pd.DataFrame(results)  # Create DataFrame first
            st.session_state.results = results_df
            st.session_state.models_trained = True

            # Update tab display
            st.success(f"✅ Forecast complete! {len(results)} models trained successfully.")

            # Show quick summary
            st.balloons()

            # Display best model
            if len(results) > 0:  # Check list length
                best = st.session_state.results.loc[st.session_state.results['RMSE'].idxmin()]
                st.info(f"🏆 Best Model: **{best['Model']}** (RMSE: {best['RMSE']:.4f})")

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Built with ❤️ using Streamlit | Stock Market Price Forecasting
</div>
""", unsafe_allow_html=True)