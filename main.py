"""
Main Pipeline: Stock Market Price Forecasting
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import StockDataLoader
from src.data_preprocessing import DataPreprocessor
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.model_comparison import ModelComparison

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'reports']
    for d in directories:
        os.makedirs(d, exist_ok=True)
    print("Directories created")

def run_pipeline():
    """Main pipeline execution"""
    print("="*70)
    print("STOCK MARKET PRICE FORECASTING PIPELINE")
    print("="*70)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Load data
    print("\n" + "="*50)
    print("STEP 1: Data Loading")
    print("="*50)
    
    ticker = input("Enter stock ticker (e.g., AAPL, ^GSPC, MSFT): ") or "^GSPC"
    start_date = input("Enter start date (YYYY-MM-DD): ") or "2020-01-01"
    
    loader = StockDataLoader(ticker=ticker, start_date=start_date)
    data = loader.get_ready_data(add_indicators=True)
    
    if data is None or data.empty:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded {len(data)} rows of data")
    print(f"Columns: {data.columns.tolist()}")
    
    # Step 3: Explore data
    print("\n" + "="*50)
    print("STEP 2: Data Exploration")
    print("="*50)
    
    preprocessor = DataPreprocessor(data)
    preprocessor.explore_data()
    
    # Step 4: Split data
    print("\n" + "="*50)
    print("STEP 3: Data Splitting")
    print("="*50)
    
    test_size = float(input("Enter test size (0.1-0.3): ") or "0.2")
    train, val, test = preprocessor.split_data(test_size=test_size, validation_size=0.1)
    
    # Save data splits
    train.to_csv('data/train.csv')
    val.to_csv('data/validation.csv')
    test.to_csv('data/test.csv')
    print("Data splits saved to data/")
    
    # Step 5: Train ARIMA
    print("\n" + "="*50)
    print("STEP 4: Training ARIMA Model")
    print("="*50)
    
    arima = ARIMAModel(train)
    arima.check_stationarity()
    
    use_auto = input("Use auto_arima for parameter selection? (y/n): ").lower() == 'y'
    if use_auto:
        arima.find_best_params()
    else:
        p = int(input("Enter p (AR order): ") or "1")
        d = int(input("Enter d (differencing): ") or "1")
        q = int(input("Enter q (MA order): ") or "1")
        arima.fit(order=(p, d, q))
    
    arima.save_model()
    arima_results = arima.evaluate(test['Close'])
    
    # Step 6: Train Prophet
    print("\n" + "="*50)
    print("STEP 5: Training Prophet Model")
    print("="*50)
    
    prophet = ProphetModel(data)
    prophet.split_data(test_size=test_size)
    prophet.fit(yearly_seasonality=True, weekly_seasonality=True)
    prophet.predict(periods=len(test))
    prophet.save_model()
    prophet_results = prophet.evaluate()
    
    try:
        prophet.plot_components()
    except:
        print("Prophet components plot not available")
    
    # Step 7: Train LSTM
    print("\n" + "="*50)
    print("STEP 6: Training LSTM Model")
    print("="*50)
    
    feature_cols = ['Close', 'SMA_20', 'RSI', 'Volume', 'Volatility']
    lookback = int(input("Enter lookback period (e.g., 60): ") or "60")
    
    lstm = LSTMModel(lookback=lookback, n_features=len(feature_cols))
    lstm.prepare_data(data, target_column='Close', 
                      feature_columns=feature_cols, test_size=test_size)
    
    lstm.build_model(lstm_units=[100, 50], dropout_rate=0.2)
    epochs = int(input("Enter number of epochs (e.g., 20): ") or "20")
    lstm.train(epochs=epochs, batch_size=32)
    lstm.save_model()
    lstm_results = lstm.evaluate()
    lstm.plot_training_history()
    
    # Step 8: Compare models
    print("\n" + "="*50)
    print("STEP 7: Model Comparison")
    print("="*50)
    
    models_dict = {
        'ARIMA': arima,
        'Prophet': prophet,
        'LSTM': lstm
    }
    
    comparator = ModelComparison(models_dict)
    comparison_results = comparator.evaluate_all(test['Close'])
    
    # Generate comparison visualizations
    comparator.plot_comparison()
    comparator.generate_report()
    
    # Step 9: Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults Summary:")
    print(comparison_results.to_string(index=False))
    
    print("\nBest performing model:")
    best_model = comparison_results.loc[comparison_results['RMSE'].idxmin()]
    print(f"  {best_model['Model']} - RMSE: {best_model['RMSE']:.4f}")
    
    print("\nOutputs saved to:")
    print("  - models/: Saved model files")
    print("  - reports/: Evaluation plots and reports")
    print("  - data/: Data splits")
    
    return comparison_results

if __name__ == "__main__":
    run_pipeline()