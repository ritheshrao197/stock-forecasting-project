"""
Prophet Model Implementation
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProphetModel:
    """Prophet model for time series forecasting"""
    
    def __init__(self, data, target_column='Close'):
        self.target_column = target_column
        self.model = None
        self.forecast = None
        self.train_data = None
        self.test_data = None
        self.data = None
        
        if isinstance(data, pd.DataFrame):
            self.prepare_data(data)
    
    def prepare_data(self, data):
        """Prepare data for Prophet format"""
        self.raw_data = data.copy()
        
        if data.index.name == 'Date' or isinstance(data.index, pd.DatetimeIndex):
            df = data.reset_index()
            df = df.rename(columns={'Date': 'ds', self.target_column: 'y'})
        else:
            df = data.copy()
            df = df.rename(columns={self.target_column: 'y'})
        
        # CRITICAL FIX: Remove timezone from dates
        if pd.api.types.is_datetime64_any_dtype(df['ds']):
            # Convert to naive datetime (remove timezone)
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        
        self.data = df[['ds', 'y']]
        print(f"Data prepared for Prophet: {len(self.data)} rows")
        print(f"Date range: {self.data['ds'].min()} to {self.data['ds'].max()}")
        print(f"Data types: ds={self.data['ds'].dtype}, y={self.data['y'].dtype}")
    
    def split_data(self, test_size=0.2):
        """Split data into train and test sets"""
        n = len(self.data)
        split_idx = int(n * (1 - test_size))
        
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]
        
        print(f"Train size: {len(self.train_data)}")
        print(f"Test size: {len(self.test_data)}")
        return self.train_data, self.test_data
    
    def fit(self, **kwargs):
        """Fit Prophet model"""
        print("Fitting Prophet model...")
        
        default_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0
        }
        default_params.update(kwargs)
        
        self.model = Prophet(**default_params)
        
        # Use train_data or full data
        data_to_fit = self.train_data if hasattr(self, 'train_data') and self.train_data is not None else self.data
        
        # Ensure no timezone in training data
        if 'ds' in data_to_fit.columns:
            data_to_fit = data_to_fit.copy()
            data_to_fit['ds'] = pd.to_datetime(data_to_fit['ds']).dt.tz_localize(None)
        
        self.model.fit(data_to_fit)
        print("Model fitted successfully")
        return self.model
    
    def predict(self, periods=30, include_history=True):
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before predicting")
        
        print(f"Generating predictions for {periods} periods...")
        
        future = self.model.make_future_dataframe(
            periods=periods, 
            include_history=include_history
        )
        self.forecast = self.model.predict(future)
        
        print(f"Predictions generated for {len(self.forecast)} periods")
        return self.forecast
    
    def evaluate(self, test_data=None, plot=True):
        """Evaluate model on test data"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        if test_data is None:
            test_data = self.test_data
        
        if test_data is None:
            raise ValueError("No test data provided")
        
        # Ensure test data has no timezone
        test_data = test_data.copy()
        if 'ds' in test_data.columns:
            test_data['ds'] = pd.to_datetime(test_data['ds']).dt.tz_localize(None)
        
        if self.forecast is None:
            self.predict(periods=len(test_data))
        
        pred_df = self.forecast[['ds', 'yhat']]
        merged = pd.merge(test_data, pred_df, on='ds', how='inner')
        
        if len(merged) == 0:
            raise ValueError("No overlapping dates")
        
        actual = merged['y']
        predicted = merged['yhat']
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Calculate R² score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
        
        print("\n=== Prophet Model Evaluation ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        if plot:
            self.plot_predictions(actual, predicted)
        
        return metrics
    
    def plot_predictions(self, actual=None, predicted=None, title="Prophet Model Predictions"):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(14, 6))
        
        if self.forecast is not None:
            plt.plot(self.forecast['ds'], self.forecast['yhat'], 
                    label='Predicted', linewidth=2, color='blue')
            plt.fill_between(
                self.forecast['ds'],
                self.forecast['yhat_lower'],
                self.forecast['yhat_upper'],
                alpha=0.2,
                color='blue',
                label='Confidence Interval'
            )
        
        if actual is not None and predicted is not None:
            plt.scatter(range(len(actual)), actual, 
                       label='Actual', color='green', alpha=0.5)
            plt.scatter(range(len(predicted)), predicted, 
                       label='Predicted', color='red', alpha=0.5)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        import os
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/prophet_predictions.png', dpi=300)
        plt.show()
    
    def plot_components(self):
        """Plot trend and seasonality components"""
        if self.forecast is None:
            raise ValueError("No forecast available")
        
        fig = self.model.plot_components(self.forecast)
        import os
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/prophet_components.png', dpi=300)
        plt.show()
    
    def save_model(self, filename="prophet_model.pkl"):
        """Save the fitted model"""
        import joblib
        import os
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, f"models/{filename}")
        print(f"Model saved to models/{filename}")


if __name__ == "__main__":
    from data_loader import StockDataLoader
    
    loader = StockDataLoader(ticker="AAPL", start_date="2020-01-01")
    data = loader.get_ready_data(add_indicators=True)
    
    prophet = ProphetModel(data)
    prophet.split_data(test_size=0.2)
    prophet.fit(yearly_seasonality=True, weekly_seasonality=True)
    prophet.predict(periods=30)
    prophet.evaluate()
    prophet.plot_components()