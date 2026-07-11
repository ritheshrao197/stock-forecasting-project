# Stock Market Price Forecasting

A comprehensive project implementing ARIMA, Prophet, and LSTM models for stock market price forecasting with short-term vs long-term prediction analysis.

## 📋 Features

- **Data Acquisition**: Automatic data fetching from Yahoo Finance
- **Data Preprocessing**: Technical indicators, feature engineering, and data splitting
- **Three Models**: ARIMA, Prophet, and LSTM implementations
- **Model Comparison**: Comprehensive evaluation with RMSE, MAE, MAPE, R²
- **Short-term vs Long-term**: Analysis of model performance across different horizons
- **Visualization**: Interactive and static plots for analysis
- **Interactive UI**: Streamlit-based dashboard with live logs

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Or run the CLI pipeline
python main.py
```

## 📊 Models

| Model | Type | Best For |
|-------|------|----------|
| ARIMA | Statistical | Short-term, linear trends |
| Prophet | ML/Seasonal | Business time series |
| LSTM | Deep Learning | Complex, non-linear patterns |

## 📁 Project Structure

```
├── app.py              # Streamlit web application
├── main.py             # CLI pipeline entry point
├── config/             # Configuration settings
├── src/                # Source code
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_comparison.py
│   └── models/
├── UI/                 # UI components
│   ├── sidebar.py
│   ├── components.py
│   └── tabs/
└── reports/            # Generated reports
```

## 📈 Usage

1. Select a ticker symbol (or search from 100+ presets)
2. Choose date range
3. Select models to run
4. Adjust parameters
5. Click "Run Forecast"

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow, Prophet, Statsmodels
- Streamlit, Plotly, Matplotlib
- pandas, numpy, scikit-learn

## 📝 License

MIT License