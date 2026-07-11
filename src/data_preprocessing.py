"""
Data Preprocessing Module
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing class for time series forecasting"""
    
    def __init__(self, data, target_column='Close'):
        self.data = data
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        
    def explore_data(self):
        """Generate exploratory data analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Closing price over time
        axes[0, 0].plot(self.data.index, self.data[self.target_column])
        axes[0, 0].set_title(f'{self.target_column} Price Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True)
        
        # Plot 2: Distribution of returns
        returns = self.data['Close'].pct_change().dropna()
        axes[0, 1].hist(returns, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution of Daily Returns')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Moving averages
        axes[1, 0].plot(self.data.index, self.data['Close'], label='Close', alpha=0.5)
        if 'SMA_20' in self.data.columns:
            axes[1, 0].plot(self.data.index, self.data['SMA_20'], label='SMA 20', linewidth=2)
        if 'SMA_50' in self.data.columns:
            axes[1, 0].plot(self.data.index, self.data['SMA_50'], label='SMA 50', linewidth=2)
        axes[1, 0].set_title('Moving Averages')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Volume over time
        axes[1, 1].bar(self.data.index, self.data['Volume'], alpha=0.5)
        axes[1, 1].set_title('Trading Volume')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Volume')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Create reports folder if it doesn't exist
        import os
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/eda_plots.png', dpi=300)
        plt.show()
        
        # Print statistics
        print("=== Data Statistics ===")
        print(f"Total rows: {len(self.data)}")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"\nTarget column: {self.target_column}")
        print(f"Mean: {self.data[self.target_column].mean():.2f}")
        print(f"Std: {self.data[self.target_column].std():.2f}")
        print(f"Min: {self.data[self.target_column].min():.2f}")
        print(f"Max: {self.data[self.target_column].max():.2f}")
        print(f"\nMissing values: {self.data.isnull().sum().sum()}")
        
    def split_data(self, test_size=0.2, validation_size=0.1):
        """Split data into train, validation, and test sets"""
        n = len(self.data)
        test_start_idx = int(n * (1 - test_size))
        val_start_idx = int(n * (1 - test_size - validation_size))
        
        train_data = self.data.iloc[:val_start_idx]
        val_data = self.data.iloc[val_start_idx:test_start_idx]
        test_data = self.data.iloc[test_start_idx:]
        
        print(f"Train size: {len(train_data)} ({len(train_data)/n*100:.1f}%)")
        print(f"Validation size: {len(val_data)} ({len(val_data)/n*100:.1f}%)")
        print(f"Test size: {len(test_data)} ({len(test_data)/n*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def prepare_for_arima(self, data):
        """Prepare data for ARIMA model"""
        return data[self.target_column]
    
    def prepare_for_prophet(self, data):
        """Prepare data for Prophet model"""
        df = data.reset_index()
        df = df.rename(columns={'Date': 'ds', self.target_column: 'y'})
        return df[['ds', 'y']]


if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import StockDataLoader
    
    loader = StockDataLoader(ticker="AAPL", start_date="2020-01-01")
    data = loader.get_ready_data(add_indicators=True)
    
    preprocessor = DataPreprocessor(data, target_column='Close')
    preprocessor.explore_data()
    
    train, val, test = preprocessor.split_data(test_size=0.2, validation_size=0.1)
    print(f"\nTrain shape: {train.shape}")
    print(f"Validation shape: {val.shape}")
    print(f"Test shape: {test.shape}")