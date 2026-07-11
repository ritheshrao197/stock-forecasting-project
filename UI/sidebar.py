"""
Sidebar configuration for the Streamlit app
"""

import streamlit as st
from datetime import datetime, timedelta
from config.settings import get_ticker_options

def render_sidebar():
    """Render the sidebar with all configuration options"""
    with st.sidebar:
        # Logo/Header
        st.markdown('<div style="text-align: center; font-size: 3rem;">📈</div>', unsafe_allow_html=True)
        st.title("📊 Configuration")
        
        # Stock Selection
        st.subheader("📈 Stock Selection")
        ticker = render_ticker_selector()
        
        # Date Range
        st.subheader("📅 Date Range")
        start_date, end_date = render_date_selector()
        
        # Model Selection
        st.subheader("🤖 Models")
        use_arima, use_prophet, use_lstm = render_model_selector()
        
        # Parameters
        st.subheader("⚙️ Parameters")
        test_size = render_parameter_controls()
        
        # LSTM Parameters
        with st.expander("LSTM Parameters"):
            lookback, lstm_epochs, lstm_units = render_lstm_parameters()
        
        # Action Buttons
        st.markdown("---")
        run_button = st.button("🚀 Run Forecast", use_container_width=True)
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            clear_all_results()
            st.rerun()
        
        return {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'use_arima': use_arima,
            'use_prophet': use_prophet,
            'use_lstm': use_lstm,
            'test_size': test_size,
            'lookback': lookback,
            'lstm_epochs': lstm_epochs,
            'lstm_units': lstm_units,
            'run_button': run_button
        }

def render_ticker_selector():
    """Render the ticker selection dropdown"""
    search_term = st.text_input("🔍 Search Ticker", placeholder="Type to search...", key="ticker_search")
    
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
        st.warning("No matching tickers found. Showing all options.")
    
    ticker_options = [opt["label"] for opt in filtered_options]
    ticker_values = [opt["value"] for opt in filtered_options]
    
    default_index = 0
    for i, opt in enumerate(filtered_options):
        if opt["value"] == "AAPL":
            default_index = i
            break
    
    selected_label = st.selectbox(
        "Select Ticker Symbol",
        options=ticker_options,
        index=min(default_index, len(ticker_options) - 1),
        help="Search above to filter tickers"
    )
    
    selected_index = ticker_options.index(selected_label) if selected_label in ticker_options else 0
    ticker = ticker_values[selected_index] if ticker_values else "AAPL"
    
    # Show category
    for opt in all_options:
        if opt["value"] == ticker:
            st.caption(f"📌 {opt['category']}")
            break
    
    # Custom ticker
    with st.expander("➕ Enter Custom Ticker"):
        custom_ticker = st.text_input("Custom Ticker", placeholder="e.g., BTC-USD, EURUSD=X")
        if custom_ticker:
            ticker = custom_ticker
            st.info(f"Using custom ticker: {ticker}")
    
    return ticker

def render_date_selector():
    """Render date range selector"""
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
    
    # Quick presets
    st.subheader("⚡ Quick Date Presets")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("📅 1Y"):
            start_date = datetime.now() - timedelta(days=365)
    with preset_col2:
        if st.button("📅 2Y"):
            start_date = datetime.now() - timedelta(days=730)
    with preset_col3:
        if st.button("📅 5Y"):
            start_date = datetime.now() - timedelta(days=1825)
    
    return start_date, end_date

def render_model_selector():
    """Render model selection checkboxes"""
    use_arima = st.checkbox("ARIMA", value=True)
    use_prophet = st.checkbox("Prophet", value=True)
    use_lstm = st.checkbox("LSTM", value=True)
    return use_arima, use_prophet, use_lstm

def render_parameter_controls():
    """Render parameter controls"""
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    return test_size

def render_lstm_parameters():
    """Render LSTM specific parameters"""
    lookback = st.number_input("Lookback Period", min_value=10, max_value=120, value=60)
    lstm_epochs = st.number_input("Epochs", min_value=5, max_value=100, value=20)
    lstm_units = st.selectbox("Hidden Units", ["[100, 50]", "[50, 25]", "[100, 50, 25]"], index=1)
    return lookback, lstm_epochs, lstm_units

def clear_all_results():
    """Clear all results from session state"""
    if 'data' in st.session_state:
        st.session_state.data = None
    if 'predictions' in st.session_state:
        st.session_state.predictions = {}
    if 'results' in st.session_state:
        st.session_state.results = None
    if 'models_trained' in st.session_state:
        st.session_state.models_trained = False
    if 'logs' in st.session_state:
        st.session_state.logs = []
    if 'is_running' in st.session_state:
        st.session_state.is_running = False