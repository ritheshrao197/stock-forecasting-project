"""
Predictions Tab
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def render_predictions(models_trained, predictions, data, ticker, test_size):
    """Render the predictions tab"""
    if models_trained and predictions:
        train_size = int(len(data) * (1 - test_size))
        test_data = data['Close'][train_size:]
        
        st.subheader("📈 Predictions vs Actual")
        
        # Interactive plot with model selection
        selected_models = st.multiselect(
            "Select models to display",
            options=list(predictions.keys()),
            default=list(predictions.keys())
        )
        
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        # Add predictions
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (name, pred) in enumerate(predictions.items()):
            if name in selected_models and len(pred) > 0:
                fig.add_trace(go.Scatter(
                    x=test_data.index[:len(pred)],
                    y=pred[:len(test_data)],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f'Stock Price Forecast - {ticker}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Zoomed view
        st.subheader("🔍 Zoomed View")
        zoom = st.slider("Days to zoom", 10, 100, 50)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=test_data.index[-zoom:],
            y=test_data.values[-zoom:],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        for i, (name, pred) in enumerate(predictions.items()):
            if name in selected_models and len(pred) >= zoom:
                fig2.add_trace(go.Scatter(
                    x=test_data.index[-zoom:],
                    y=pred[-zoom:],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig2.update_layout(
            title=f'Zoomed View (Last {zoom} Days)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Error analysis
        with st.expander("📊 Error Analysis"):
            for name, pred in predictions.items():
                if len(pred) > 0:
                    errors = test_data.values[:len(pred)] - pred[:len(test_data)]
                    fig3, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(range(len(errors)), errors, label=f'{name} Errors')
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax.axhline(y=np.mean(errors), color='red', linestyle='--', label='Mean Error')
                    ax.set_title(f'{name} - Prediction Errors')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Error')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig3)
    else:
        st.info("👈 Run the forecast to see predictions.")