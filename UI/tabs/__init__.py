"""
Tab modules for the Stock Market Forecasting app
"""

from ui.tabs.data_overview import render_data_overview
from ui.tabs.model_results import render_model_results
from ui.tabs.predictions import render_predictions
from ui.tabs.model_comparison import render_model_comparison
from ui.tabs.live_logs import render_live_logs
from ui.tabs.about import render_about

__all__ = [
    'render_data_overview',
    'render_model_results',
    'render_predictions',
    'render_model_comparison',
    'render_live_logs',
    'render_about'
]