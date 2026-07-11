"""
Sidebar configuration for the Streamlit app - Indian Stocks Version
"""

import streamlit as st
from datetime import datetime, timedelta
from config.settings import get_ticker_options

# Check if LSTM is available
try:
    from src.models.lstm_model import is_lstm_available
    LSTM_AVAILABLE = is_lstm_available()
except:
    LSTM_AVAILABLE = False

def render_sidebar():
    """Render the sidebar with all configuration options - Indian Stocks"""
    
    # Apply custom CSS
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            height: 100vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
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
        
        /* Indian stocks badge */
        .india-badge {
            background-color: #ff6b35;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.6rem;
            font-weight: bold;
            display: inline-block;
            margin-left: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Header with Indian badge
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 0.5rem;">
            <span style="font-size: 2rem;">📈</span>
            <span style="font-size: 1.2rem; font-weight: bold;">Configuration</span>
            <span class="india-badge">🇮🇳 INDIA</span>
        </div>
        """, unsafe_allow_html=True)
        
        # ============================================
        # Stock Selection
        # ============================================
        st.subheader("📈 Stock Selection")
        
        search_term = st.text_input(
            "🔍 Search Ticker",
            placeholder="Type company name or symbol...",
            key="ticker_search",
            label_visibility="collapsed"
        )
        
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
        
        ticker_options = [opt["label"] for opt in filtered_options]
        ticker_values = [opt["value"] for opt in filtered_options]
        
        # Set default to Reliance (Indian stock)
        default_index = 0
        for i, opt in enumerate(filtered_options):
            if opt["value"] == "RELIANCE.NS":
                default_index = i
                break
        
        selected_label = st.selectbox(
            "Select Ticker Symbol",
            options=ticker_options,
            index=min(default_index, len(ticker_options) - 1),
            key="ticker_select",
            label_visibility="collapsed"
        )
        
        selected_index = ticker_options.index(selected_label) if selected_label in ticker_options else 0
        ticker = ticker_values[selected_index] if ticker_values else "RELIANCE.NS"
        
        # Show category
        for opt in all_options:
            if opt["value"] == ticker:
                st.markdown(f'<div class="category-label">📌 {opt["category"]}</div>', unsafe_allow_html=True)
                break
        
        # Custom ticker
        with st.expander("➕ Custom Ticker"):
            st.caption("Use .NS for NSE or .BO for BSE")
            custom_ticker = st.text_input(
                "Ticker", 
                placeholder="e.g., TATAMOTORS.NS, RELIANCE.BO",
                key="custom_ticker"
            )
            if custom_ticker:
                ticker = custom_ticker
                st.caption(f"Using: {ticker}")
        
        # ============================================
        # Date Range
        # ============================================
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
        
        # ============================================
        # Model Selection
        # ============================================
        st.subheader("🤖 Models")
        use_arima = st.checkbox("ARIMA", value=True, key="use_arima")
        use_prophet = st.checkbox("Prophet", value=True, key="use_prophet")
        
        if LSTM_AVAILABLE:
            use_lstm = st.checkbox("LSTM", value=True, key="use_lstm")
        else:
            st.caption("⚠️ LSTM disabled")
            use_lstm = False
        
        # ============================================
        # Parameters
        # ============================================
        st.subheader("⚙️ Parameters")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="test_size")
        
        # LSTM Parameters
        with st.expander("🧠 LSTM Params"):
            lookback = st.number_input("Lookback", min_value=10, max_value=120, value=60, key="lookback")
            lstm_epochs = st.number_input("Epochs", min_value=5, max_value=100, value=20, key="lstm_epochs")
            lstm_units = st.selectbox("Units", ["[100, 50]", "[50, 25]", "[100, 50, 25]"], index=1, key="lstm_units")
        
        # ============================================
        # Action Buttons
        # ============================================
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
            st.caption("⏳ Running...")
        elif forecast_completed:
            st.caption("✅ Done! Rerun to update.")
        
        if st.button("🗑️ Clear", use_container_width=True):
            clear_all_results()
            st.rerun()
        
        # Version info
        st.caption("v1.0 | 🇮🇳 Indian Stocks | ARIMA · Prophet · LSTM")
        
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

def clear_all_results():
    """Clear all results from session state"""
    keys_to_clear = ['data', 'predictions', 'results', 'models_trained', 'logs', 'is_running', 'forecast_completed']
    for key in keys_to_clear:
        if key in st.session_state:
            if key == 'predictions':
                st.session_state[key] = {}
            elif key in ['data', 'results', 'models_trained']:
                st.session_state[key] = None
            elif key == 'logs':
                st.session_state[key] = []
            elif key == 'forecast_completed':
                st.session_state[key] = False
            elif key == 'is_running':
                st.session_state[key] = False