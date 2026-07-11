"""
Logging utilities for the Streamlit app
"""

import streamlit as st
from datetime import datetime

def add_log(message, log_type="info"):
    """Add a log message with timestamp and type"""
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    }
    st.session_state.logs.append(log_entry)
    
    # Keep only last 200 logs
    if len(st.session_state.logs) > 200:
        st.session_state.logs = st.session_state.logs[-200:]

def clear_logs():
    """Clear all logs"""
    if 'logs' in st.session_state:
        st.session_state.logs = []

def get_logs():
    """Get all logs"""
    return st.session_state.get('logs', [])

def log_info(message):
    """Add info log"""
    add_log(message, "info")

def log_success(message):
    """Add success log"""
    add_log(message, "success")

def log_warning(message):
    """Add warning log"""
    add_log(message, "warning")

def log_error(message):
    """Add error log"""
    add_log(message, "error")

def log_header(message):
    """Add header log"""
    add_log(message, "header")

def render_logs():
    """Render logs in the UI"""
    logs = get_logs()
    
    if not logs:
        st.info("📝 No logs yet. Run the forecast to see terminal output.")
        return
    
    color_map = {
        "info": "log-info",
        "success": "log-success",
        "warning": "log-warning",
        "error": "log-error",
        "header": "log-header"
    }
    
    log_html = ""
    for log in logs:
        color_class = color_map.get(log["type"], "log-info")
        log_html += f'<div><span style="color: #888;">[{log["timestamp"]}]</span> <span class="{color_class}">{log["message"]}</span></div>'
    
    st.markdown(f"""
    <style>
    .log-container {{
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
    }}
    .log-container::-webkit-scrollbar {{
        width: 8px;
    }}
    .log-container::-webkit-scrollbar-track {{
        background: #1e1e1e;
    }}
    .log-container::-webkit-scrollbar-thumb {{
        background: #555;
        border-radius: 4px;
    }}
    .log-container::-webkit-scrollbar-thumb:hover {{
        background: #777;
    }}
    .log-info {{ color: #4fc3f7; }}
    .log-success {{ color: #81c784; }}
    .log-warning {{ color: #ffb74d; }}
    .log-error {{ color: #e57373; }}
    .log-header {{ color: #ce93d8; font-weight: bold; }}
    </style>
    <div class="log-container">
        {log_html}
    </div>
    """, unsafe_allow_html=True)