"""
Model Comparison Tab
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_model_comparison(models_trained, results):
    """Render the model comparison tab"""
    if models_trained and results is not None:
        st.subheader("📋 Detailed Model Comparison")
        
        # Show all metrics in a nice table
        st.dataframe(
            results.style.background_gradient(subset=['RMSE', 'MAE', 'MAPE'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        # Radar chart
        st.subheader("📊 Model Performance Radar")
        
        # Normalize metrics for radar
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
        df_radar = results[['Model'] + metrics_to_plot].copy()
        
        # Normalize
        for metric in metrics_to_plot:
            max_val = df_radar[metric].max()
            if max_val > 0:
                df_radar[f'{metric}_norm'] = 1 - (df_radar[metric] / max_val)
        
        # Create radar chart
        fig = go.Figure()
        
        for model in df_radar['Model']:
            model_data = df_radar[df_radar['Model'] == model]
            fig.add_trace(go.Scatterpolar(
                r=[model_data[f'{m}_norm'].values[0] for m in metrics_to_plot],
                theta=metrics_to_plot,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='Model Performance Radar (Higher is Better)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        best_rmse = results.loc[results['RMSE'].idxmin()]
        best_mape = results.loc[results['MAPE'].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **📌 Best Model (RMSE)**
            - Model: **{best_rmse['Model']}**
            - RMSE: {best_rmse['RMSE']:.4f}
            - MAPE: {best_rmse['MAPE']:.2f}%
            - R²: {best_rmse['R²']:.4f}
            """)
        with col2:
            st.success(f"""
            **🎯 Best Model (MAPE)**
            - Model: **{best_mape['Model']}**
            - RMSE: {best_mape['RMSE']:.4f}
            - MAPE: {best_mape['MAPE']:.2f}%
            - R²: {best_mape['R²']:.4f}
            """)
        
        # Usage recommendations
        st.markdown("""
        ### 📚 When to Use Each Model
        
        | Model | Best For | Strengths | Limitations |
        |-------|----------|-----------|-------------|
        | **ARIMA** | Short-term forecasting | Simple, interpretable, fast | Struggles with non-linear patterns |
        | **Prophet** | Seasonal data | Handles seasonality, missing data | Less accurate for chaotic markets |
        | **LSTM** | Long-term, complex patterns | Captures non-linear relationships | Requires more data, computationally heavy |
        """)
    else:
        st.info("👈 Run the forecast to see model comparison.")