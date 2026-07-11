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
    
    # Apply styling
    styled_df = df.style.background_gradient(
        subset=['RMSE', 'MAE', 'MAPE'] if all(col in df.columns for col in ['RMSE', 'MAE', 'MAPE']) else None,
        cmap='RdYlGn_r'
    )
    st.dataframe(styled_df, use_container_width=use_container_width)