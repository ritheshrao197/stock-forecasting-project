"""
Model Comparison Module
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelComparison:
    """Compare multiple forecasting models"""
    
    def __init__(self, models_dict):
        self.models = models_dict
        self.results = None
        self.predictions = {}
        self.y_true = None
    
    def evaluate_all(self, test_data, target_column='Close'):
        """Evaluate all models on test data"""
        if isinstance(test_data, pd.DataFrame):
            y_true = test_data[target_column].values
        else:
            y_true = test_data.values
        
        self.y_true = y_true
        results = []
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {name}...")
            print('='*50)
            
            try:
                if name.lower() == 'arima':
                    pred, _ = model.predict(steps=len(y_true))
                    y_pred = pred.values if hasattr(pred, 'values') else pred
                elif name.lower() == 'prophet':
                    y_pred = model.forecast['yhat'].values[-len(y_true):]
                elif name.lower() == 'lstm':
                    y_pred = model.predictions
                    if len(y_pred) > len(y_true):
                        y_pred = y_pred[-len(y_true):]
                    elif len(y_pred) < len(y_true):
                        y_pred = np.pad(y_pred, (len(y_true)-len(y_pred), 0), mode='edge')
                else:
                    y_pred = model.predict(len(y_true))
                
                if len(y_pred) > len(y_true):
                    y_pred = y_pred[:len(y_true)]
                
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                results.append({
                    'Model': name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'R²': r2
                })
                
                self.predictions[name] = y_pred
                
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"MAPE: {mape:.2f}%")
                print(f"R²: {r2:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results.append({
                    'Model': name,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'MAPE': np.nan,
                    'R²': np.nan
                })
                self.predictions[name] = np.array([np.nan] * len(y_true))
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def plot_comparison(self, save_path='reports/model_comparison.png'):
        """Create comparison visualizations"""
        if self.results is None or self.results.empty:
            print("No results to plot. Run evaluate_all first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.barplot(data=self.results, x='Model', y='RMSE', ax=axes[0, 0])
        axes[0, 0].set_title('RMSE Comparison', fontsize=14)
        axes[0, 0].set_ylabel('RMSE')
        
        sns.barplot(data=self.results, x='Model', y='MAPE', ax=axes[0, 1])
        axes[0, 1].set_title('MAPE Comparison', fontsize=14)
        axes[0, 1].set_ylabel('MAPE (%)')
        
        sns.barplot(data=self.results, x='Model', y='R²', ax=axes[1, 0])
        axes[1, 0].set_title('R² Comparison', fontsize=14)
        axes[1, 0].set_ylabel('R²')
        
        ax = axes[1, 1]
        ax.plot(self.y_true, label='Actual', linewidth=2, color='black')
        for name, pred in self.predictions.items():
            if not np.all(np.isnan(pred)):
                ax.plot(pred, label=name, alpha=0.7, linewidth=1.5)
        ax.set_title('Predictions vs Actual', fontsize=14)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        import os
        os.makedirs('reports', exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.show()
        return fig
    
    def generate_report(self, save_path='reports/model_comparison_report.txt'):
        """Generate a text report"""
        if self.results is None or self.results.empty:
            print("No results to report. Run evaluate_all first.")
            return
        
        report = []
        report.append("="*70)
        report.append("STOCK MARKET FORECASTING MODEL COMPARISON REPORT")
        report.append("="*70)
        report.append("")
        report.append("OVERALL PERFORMANCE SUMMARY")
        report.append("-"*50)
        report.append(self.results.to_string(index=False))
        report.append("")
        
        best_rmse = self.results.loc[self.results['RMSE'].idxmin()]
        report.append(f"BEST MODEL (by RMSE): {best_rmse['Model']}")
        report.append(f"  RMSE: {best_rmse['RMSE']:.4f}")
        report.append(f"  MAE: {best_rmse['MAE']:.4f}")
        report.append(f"  MAPE: {best_rmse['MAPE']:.2f}%")
        report.append(f"  R²: {best_rmse['R²']:.4f}")
        report.append("")
        
        report.append("="*70)
        
        import os
        os.makedirs('reports', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {save_path}")
        return report

    def export_results_csv(self, save_path='reports/model_comparison_results.csv'):
        """Export comparison results to CSV
        
        Parameters:
        -----------
        save_path : str
            Path to save the CSV file
        """
        if self.results is None or self.results.empty:
            logger.warning("No results to export")
            return
        
        import os
        os.makedirs('reports', exist_ok=True)
        self.results.to_csv(save_path, index=False)
        logger.info(f"Results exported to {save_path}")
        print(f"Results exported to {save_path}")


if __name__ == "__main__":
    print("Model comparison module loaded successfully")