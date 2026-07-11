"""
ARIMA Model Implementation
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ARIMAModel:
    """
    ARIMA model for time series forecasting
    """
    
    def __init__(self, data, target_column='Close'):
        """
        Initialize ARIMA model
        
        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Time series data
        target_column : str
            Column to forecast
        """
        if isinstance(data, pd.DataFrame):
            self.data = data[target_column]
        else:
            self.data = data
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.results = None
    
    def check_stationarity(self):
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Returns:
        --------
        bool: True if data is stationary
        """
        result = adfuller(self.data.dropna())
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        
        is_stationary = result[1] < 0.05
        print(f'\nData is {"stationary" if is_stationary else "non-stationary"}')
        return is_stationary
    
    def plot_acf_pacf(self, lags=40):
        """
        Plot ACF and PACF for model order selection
        
        Parameters:
        -----------
        lags : int
            Number of lags to display
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(self.data.dropna(), ax=ax1, lags=lags)
        plot_pacf(self.data.dropna(), ax=ax2, lags=lags)
        plt.tight_layout()
        plt.savefig('reports/arima_acf_pacf.png', dpi=300)
        plt.show()
    
    def find_best_params(self, start_p=0, max_p=5, start_q=0, max_q=5, seasonal=False):
        """
        Use auto_arima to find optimal parameters
        
        Parameters:
        -----------
        start_p, max_p : int
            Range for AR order
        start_q, max_q : int
            Range for MA order
        seasonal : bool
            Whether to use seasonal ARIMA
            
        Returns:
        --------
        tuple: (p, d, q) optimal parameters
        """
        print("Finding optimal ARIMA parameters...")
        
        # Check if differencing is needed
        d = 0
        if not self.check_stationarity():
            d = 1
            print(f"Will use d={d} (differencing)")
        
        # Use auto_arima for parameter selection
        auto_model = auto_arima(
            self.data,
            start_p=start_p,
            max_p=max_p,
            start_q=start_q,
            max_q=max_q,
            d=d,
            seasonal=seasonal,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        self.best_params = auto_model.order
        print(f"\nBest parameters: p={self.best_params[0]}, d={self.best_params[1]}, q={self.best_params[2]}")
        
        return self.best_params
    
    def fit(self, order=None):
        """
        Fit ARIMA model
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) parameters (uses auto-selected if None)
        """
        if order is None:
            if not hasattr(self, 'best_params'):
                self.find_best_params()
            order = self.best_params
        
        print(f"Fitting ARIMA model with order {order}...")
        self.model = ARIMA(self.data, order=order)
        self.fitted_model = self.model.fit()
        print(self.fitted_model.summary())
        return self.fitted_model
    
    def predict(self, steps=30):
        """
        Generate predictions
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        tuple: (predictions, confidence_intervals)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before predicting")
        
        # Get forecast
        forecast = self.fitted_model.forecast(steps=steps)
        self.predictions = forecast
        
        # Get confidence intervals
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        conf_int = forecast_result.conf_int()
        
        print(f"Predictions for next {steps} days:")
        print(forecast)
        
        return forecast, conf_int
    
    def evaluate(self, test_data, plot=True):
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        test_data : pd.Series
            Actual test values
        plot : bool
            Whether to plot results
            
        Returns:
        --------
        dict: Evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Make predictions for test period
        if self.fitted_model is None:
            self.fit()
        
        # Predict as many steps as test data length
        predictions, _ = self.predict(steps=len(test_data))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        print("\n=== Model Evaluation ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        if plot:
            self.plot_predictions(test_data, predictions)
        
        return metrics
    
    def plot_predictions(self, actual, predicted, title="ARIMA Model Predictions"):
        """
        Plot actual vs predicted values
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values
        predicted : pd.Series or np.ndarray
            Predicted values
        title : str
            Plot title
        """
        plt.figure(figsize=(14, 6))
        plt.plot(actual.index, actual, label='Actual', linewidth=2)
        plt.plot(actual.index, predicted, label='Predicted', linewidth=2)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('reports/arima_predictions.png', dpi=300)
        plt.show()
    
    def save_model(self, filename="arima_model.pkl"):
        """
        Save the fitted model
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        import joblib
        joblib.dump(self.fitted_model, f"models/{filename}")
        print(f"Model saved to models/{filename}")
    
    def load_model(self, filename="arima_model.pkl"):
        """
        Load a saved model
        
        Parameters:
        -----------
        filename : str
            Input filename
        """
        import joblib
        self.fitted_model = joblib.load(f"models/{filename}")
        print(f"Model loaded from models/{filename}")


# Example usage
if __name__ == "__main__":
    from data_loader import StockDataLoader
    from data_preprocessing import DataPreprocessor
    
    # Load and prepare data
    loader = StockDataLoader(ticker="AAPL", start_date="2019-01-01")
    data = loader.get_ready_data(add_indicators=True)
    
    # Split data
    preprocessor = DataPreprocessor(data)
    train, val, test = preprocessor.split_data(test_size=0.2)
    
    # Train ARIMA
    arima = ARIMAModel(train)
    arima.fit()
    
    # Evaluate
    metrics = arima.evaluate(test['Close'])