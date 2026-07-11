"""
Reusable UI components for the Streamlit app
"""

import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def render_metric_card(title, value, description=None, icon=None):
    """Render a metric card with icon"""
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''
    st.markdown(f"""
    <div style="
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    ">
        {icon_html}
        <span style="font-size: 0.9rem; color: #666;">{title}</span>
        <p style="font-size: 1.5rem; font-weight: bold; margin: 0.2rem 0;">{value}</p>
        {f'<p style="font-size: 0.8rem; color: #888; margin: 0;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def render_plotly_chart(fig, height=400, use_container_width=True):
    """Render a Plotly chart with consistent styling"""
    fig.update_layout(
        height=height,
        hovermode='x unified',
        template='plotly_white',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=14, color='#2c3e50')
    )
    st.plotly_chart(fig, use_container_width=use_container_width)

def render_matplotlib_figure(fig, title=None):
    """Render a matplotlib figure"""
    if title:
        st.subheader(title)
    st.pyplot(fig)
    plt.close('all')

def render_dataframe(df, title=None, use_container_width=True):
    """Render a pandas DataFrame with styling"""
    if title:
        st.subheader(title)
    
    # Apply styling if columns exist
    style_cols = []
    for col in ['RMSE', 'MAE', 'MAPE']:
        if col in df.columns:
            style_cols.append(col)
    
    if style_cols:
        styled_df = df.style.background_gradient(subset=style_cols, cmap='RdYlGn_r')
    else:
        styled_df = df
    
    st.dataframe(styled_df, use_container_width=use_container_width)

def render_footer():
    """Render a consistent footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Built with ❤️ using Streamlit | Stock Market Price Forecasting
    </div>
    """, unsafe_allow_html=True)

def render_info_box(title, message, icon="ℹ️"):
    """Render a styled info box
    
    Parameters:
    -----------
    title : str
        Box title
    message : str
        Box message content
    icon : str
        Emoji icon to display
    """
    st.markdown(f"""
    <div style="
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    ">
        <strong>{icon} {title}</strong>
        <p style="margin: 0.5rem 0 0 0;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def render_loading_spinner(message="Loading..."):
    """Render a loading spinner with message
    
    Parameters:
    -----------
    message : str
        Loading message to display
    """
    with st.spinner(message):
        pass

def render_success_banner(message):
    """Render a success banner
    
    Parameters:
    -----------
    message : str
        Success message to display
    """
    st.markdown(f"""
    <div style="
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        color: #155724;
    ">
        ✅ <strong>{message}</strong>
    </div>
    """, unsafe_allow_html=True)