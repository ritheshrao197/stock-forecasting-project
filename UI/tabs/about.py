"""
About Tab
"""

import streamlit as st

def render_about():
    """Render the about tab"""
    st.markdown("""
    <h2 style="font-size: 1.5rem; color: #2c3e50; margin-top: 1rem;">ℹ️ About This Project</h2>
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
    - 🔍 Searchable ticker dropdown with 100+ symbols
    - 📝 Live terminal logs during processing
    
    #### 📚 Technology Stack
    
    - **Frontend**: Streamlit
    - **Data**: yfinance, pandas
    - **Models**: statsmodels, prophet, tensorflow
    - **Visualization**: plotly, matplotlib
    
    #### 📝 How to Use
    
    1. Search and select a ticker symbol from the dropdown
    2. Select date range (or use quick presets)
    3. Choose models to run
    4. Adjust parameters
    5. Click "Run Forecast"
    6. Watch live logs in the "Live Logs" tab
    7. Explore results across other tabs!
    
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