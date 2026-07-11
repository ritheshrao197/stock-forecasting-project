"""
Live Logs Tab
Displays real-time terminal output during forecasting
"""

import streamlit as st
from ui.utils.logging_utils import render_logs, clear_logs, get_logs

def render_live_logs():
    """Render the live logs tab"""
    st.subheader("📝 Live Terminal Logs")
    
    # Log controls
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🔄 Refresh Logs"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Logs"):
            clear_logs()
            st.rerun()
    with col3:
        logs = get_logs()
        st.caption(f"📊 {len(logs)} log entries")
    
    # Auto-refresh checkbox
    auto_refresh = st.checkbox("🔄 Auto-refresh logs", value=True)
    if auto_refresh:
        st.caption("Logs will auto-refresh every 3 seconds")
        st.empty()
    
    # Render logs
    render_logs()
    
    # Status indicator
    if st.session_state.get('is_running', False):
        st.warning("⏳ Process is currently running...")
    else:
        st.success("✅ Process idle")