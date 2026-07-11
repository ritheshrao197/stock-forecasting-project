"""
Model Results Tab
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def render_model_results(models_trained, results, data, predictions, test_size):
    """Render the model results tab"""
    if models_trained and results is not None:
        st.subheader("📊 Model Performance Comparison")
        
        # Metrics table
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                results.style.background_gradient(subset=['RMSE', 'MAPE'], cmap='RdYlGn_r'),
                use_container_width=True
            )
        
        with col2:
            # Best model
            best_model = results.loc[results['RMSE'].idxmin()]
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #1f77b4;
                margin: 0.5rem 0;
            ">
                <h4>🏆 Best Model</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{best_model['Model']}</p>
                <p>RMSE: {best_model['RMSE']:.4f}</p>
                <p>MAPE: {best_model['MAPE']:.2f}%</p>
                <p>R²: {best_model['R²']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(results['Model'], results['RMSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('RMSE Comparison', fontsize=14)
            ax.set_ylabel('RMSE')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, results['RMSE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(results['Model'], results['MAPE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('MAPE Comparison', fontsize=14)
            ax.set_ylabel('MAPE (%)')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, results['MAPE']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                       f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
        
        # Short-term vs Long-term
        st.subheader("📊 Short-term vs Long-term Performance")
        
        if data is not None and predictions:
            train_size = int(len(data) * (1 - test_size))
            test_data = data['Close'][train_size:]
            
            short_metrics = {}
            long_metrics = {}
            
            for name, pred in predictions.items():
                if len(pred) > 0:
                    short_len = min(10, len(pred))
                    long_len = min(30, len(pred))
                    
                    short_rmse = np.sqrt(np.mean((test_data.values[:short_len] - pred[:short_len])**2))
                    long_rmse = np.sqrt(np.mean((test_data.values[:long_len] - pred[:long_len])**2))
                    
                    short_metrics[name] = short_rmse
                    long_metrics[name] = long_rmse
            
            if short_metrics:
                df_compare = pd.DataFrame({
                    'Model': list(short_metrics.keys()),
                    'Short-term (10 days)': list(short_metrics.values()),
                    'Long-term (30 days)': list(long_metrics.values())
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Short-term (10 days)', x=df_compare['Model'], 
                                    y=df_compare['Short-term (10 days)'], marker_color='#1f77b4'))
                fig.add_trace(go.Bar(name='Long-term (30 days)', x=df_compare['Model'], 
                                    y=df_compare['Long-term (30 days)'], marker_color='#ff7f0e'))
                fig.update_layout(
                    title='RMSE: Short-term vs Long-term',
                    xaxis_title='Model',
                    yaxis_title='RMSE',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                best_short = min(short_metrics.items(), key=lambda x: x[1])
                best_long = min(long_metrics.items(), key=lambda x: x[1])
                
                st.success(f"""
                **📌 Key Insights:**
                - Best for **Short-term** forecasting: **{best_short[0]}** (RMSE: {best_short[1]:.4f})
                - Best for **Long-term** forecasting: **{best_long[0]}** (RMSE: {best_long[1]:.4f})
                """)
    else:
        st.info("👈 Run the forecast to see model results.")