"""
Logging utilities for the Streamlit app
"""

import streamlit as st
from datetime import datetime

def add_log(message, log_type="info"):
    """
    Add a log message with timestamp and type
    
    Parameters:
    -----------
    message : str
        The log message
    log_type : str
        Type of log: info, success, warning, error, header
    """
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    }
    st.session_state.logs.append(log_entry)
    
    # Keep only last N logs
    max_logs = st.session_state.get('max_log_entries', 200)
    if len(st.session_state.logs) > max_logs:
        st.session_state.logs = st.session_state.logs[-max_logs:]

def clear_logs():
    """Clear all logs"""
    if 'logs' in st.session_state:
        st.session_state.logs = []

def get_logs():
    """Get all logs"""
    return st.session_state.get('logs', [])

def get_log_count():
    """Get the number of log entries"""
    return len(st.session_state.get('logs', []))

def export_logs(filepath='reports/logs_export.txt'):
    """Export logs to a text file
    
    Parameters:
    -----------
    filepath : str
        Path to save the log file
    """
    import os
    os.makedirs('reports', exist_ok=True)
    
    logs = get_logs()
    with open(filepath, 'w') as f:
        for log in logs:
            f.write(f"[{log['timestamp']}] [{log['type'].upper()}] {log['message']}\n")
    return filepath

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
    
    # Color mapping for log types
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