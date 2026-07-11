"""
About Tab - Indian Stocks Version
"""

import streamlit as st

def render_about():
    """Render the about tab with Indian stocks focus"""
    st.markdown("""
    <h2 style="font-size: 1.5rem; color: #2c3e50; margin-top: 1rem;">🇮🇳 Indian Stock Market Forecasting</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Stock Market Price Forecasting
    
    This interactive dashboard provides a comprehensive comparison of three popular time series forecasting models for **Indian stocks**.
    
    #### 🤖 Models Implemented
    
    1. **ARIMA** - Classical statistical model for linear trends
       - Best for: Short-term forecasting
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
    
    #### 📝 How to Use
    
    1. Search and select a ticker symbol from the dropdown
    2. Select date range (or use quick presets)
    3. Choose models to run
    4. Adjust parameters
    5. Click "Run Forecast"
    6. Explore results across tabs!
    
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