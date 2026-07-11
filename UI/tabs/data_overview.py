"""
Data Overview Tab
"""

import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

def render_data_overview(data, ticker):
    """Render the data overview tab"""
    if data is not None:
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