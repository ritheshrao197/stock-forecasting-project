"""
LSTM Model Implementation - With Error Handling
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM model will not work.")

class LSTMModel:
    """LSTM model for time series forecasting"""
    
    def __init__(self, lookback=30, n_features=1):
        self.lookback = lookback
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.history = None
        
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ LSTM model initialized but TensorFlow is not available.")
            print("   LSTM will not work. Please install TensorFlow to use this model.")
    
    def prepare_data(self, data, target_column='Close', feature_columns=None, test_size=0.2):
        """Prepare data for LSTM training"""
        if feature_columns is None:
            feature_columns = [target_column]
        elif target_column not in feature_columns:
            feature_columns = [target_column] + feature_columns
        
        data = data[feature_columns].copy()
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Check if we have enough data
        if len(data) < self.lookback + 10:
            print(f"⚠️ Warning: Only {len(data)} rows available. Need at least {self.lookback + 10}.")
            print(f"Reducing lookback to {max(5, len(data) // 10)}")
            self.lookback = max(5, len(data) // 10)
        
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            raise ValueError(f"No sequences created. Data length: {len(data)}, Lookback: {self.lookback}")
        
        # Split into train and test
        split_idx = int(len(X) * (1 - test_size))
        
        # Ensure we have at least some training data
        if split_idx < 10:
            print(f"⚠️ Warning: Only {split_idx} training samples. Reducing test_size.")
            split_idx = max(10, len(X) - 10)
        
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_test = X[split_idx:]
        self.y_test = y[split_idx:]
        
        print(f"Data prepared:")
        print(f"  Total sequences: {len(X)}")
        print(f"  X_train shape: {self.X_train.shape}")
        print(f"  y_train shape: {self.y_train.shape}")
        print(f"  X_test shape: {self.X_test.shape}")
        print(f"  y_test shape: {self.y_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def build_model(self, lstm_units=[50, 25], dropout_rate=0.2, learning_rate=0.001):
        """Build LSTM architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install tensorflow-cpu or tensorflow.")
        
        print("Building LSTM model...")
        
        # Check if we have training data
        if self.X_train is None or len(self.X_train) == 0:
            raise ValueError("No training data available. Run prepare_data first with sufficient data.")
        
        # Adjust input shape
        input_shape = (self.lookback, self.n_features)
        
        model = Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='mse')
        
        print(model.summary())
        self.model = model
        return model
    
    def train(self, epochs=20, batch_size=16, validation_split=0.1, verbose=1):
        """Train the LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install tensorflow-cpu or tensorflow.")
        
        if self.model is None:
            self.build_model()
        
        if self.X_train is None or len(self.X_train) == 0:
            raise ValueError("No training data. Run prepare_data first.")
        
        # Adjust validation split if too few samples
        if len(self.X_train) < 20:
            validation_split = 0.0
            print(f"⚠️ Warning: Only {len(self.X_train)} training samples. Disabling validation split.")
        
        print(f"Training LSTM model on {len(self.X_train)} samples...")
        
        callbacks = []
        if validation_split > 0:
            callbacks.append(
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            )
        
        # Reduce batch size if too few samples
        if len(self.X_train) < batch_size:
            batch_size = max(1, len(self.X_train) // 4)
            print(f"Adjusted batch size to {batch_size}")
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks if callbacks else None,
            verbose=verbose
        )
        
        print("Training completed!")
        return self.history
    
    def predict(self, X_test=None, inverse_transform=True):
        """Generate predictions"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install tensorflow-cpu or tensorflow.")
        
        if X_test is None:
            X_test = self.X_test
        
        if self.model is None:
            raise ValueError("Model must be built and trained before predicting")
        
        if X_test is None or len(X_test) == 0:
            print("⚠️ Warning: No test data available for prediction.")
            return np.array([])
        
        print(f"Generating predictions on {len(X_test)} samples...")
        predictions = self.model.predict(X_test, verbose=0)
        
        if inverse_transform:
            dummy = np.zeros((len(predictions), self.n_features))
            dummy[:, 0] = predictions.flatten()
            predictions = self.scaler.inverse_transform(dummy)[:, 0]
        
        self.predictions = predictions
        return predictions
    
    def evaluate(self, y_true=None, y_pred=None, plot=True):
        """Evaluate model performance"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available. Cannot evaluate LSTM model.")
            return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        
        if y_true is None:
            y_true = self.y_test
        
        if y_pred is None:
            if self.predictions is None:
                self.predict(inverse_transform=False)
            y_pred = self.predictions
        
        if y_true is None or len(y_true) == 0:
            print("⚠️ No test data available for evaluation.")
            return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        
        dummy = np.zeros((len(y_true), self.n_features))
        dummy[:, 0] = y_true.flatten()
        y_true_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        rmse = np.sqrt(mean_squared_error(y_true_original, y_pred))
        mae = mean_absolute_error(y_true_original, y_pred)
        mape = np.mean(np.abs((y_true_original - y_pred) / y_true_original)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        print("\n=== LSTM Model Evaluation ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        if plot and len(y_true_original) > 0:
            self.plot_predictions(y_true_original, y_pred)
        
        return metrics
    
    def plot_predictions(self, actual, predicted, title="LSTM Model Predictions"):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(14, 6))
        plt.plot(actual, label='Actual', linewidth=2)
        plt.plot(predicted, label='Predicted', linewidth=2)
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        import os
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/lstm_predictions.png', dpi=300)
        plt.show()
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if 'mae' in self.history.history:
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['mae'], label='Training MAE')
            if 'val_mae' in self.history.history:
                plt.plot(self.history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        import os
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/lstm_training_history.png', dpi=300)
        plt.show()
    
    def save_model(self, filename="lstm_model.h5"):
        """Save the trained model"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available. Cannot save model.")
            return
        
        import os
        os.makedirs('models', exist_ok=True)
        self.model.save(f"models/{filename}")
        print(f"Model saved to models/{filename}")


if __name__ == "__main__":
    from data_loader import StockDataLoader
    
    loader = StockDataLoader(ticker="AAPL", start_date="2020-01-01")
    data = loader.get_ready_data(add_indicators=True)
    
    if TENSORFLOW_AVAILABLE:
        lstm = LSTMModel(lookback=30, n_features=5)
        feature_cols = ['Close', 'SMA_20', 'RSI', 'Volume', 'Volatility']
        lstm.prepare_data(data, target_column='Close', feature_columns=feature_cols, test_size=0.2)
        lstm.build_model(lstm_units=[50, 25], dropout_rate=0.2)
        lstm.train(epochs=10, batch_size=16)
        lstm.evaluate()
        lstm.plot_training_history()
    else:
        print("❌ TensorFlow not available. Skipping LSTM demo.")