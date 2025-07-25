
"""
Model evaluation utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Model evaluation utilities for forecasting models"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, actual, predicted):
        """Calculate comprehensive evaluation metrics"""
        try:
            # Ensure arrays are numpy arrays
            actual = np.array(actual)
            predicted = np.array(predicted)
            
            # Remove any NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual = actual[mask]
            predicted = predicted[mask]
            
            if len(actual) == 0:
                return {'error': 'No valid data points for evaluation'}
            
            # Basic metrics
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual, predicted)
            
            # MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            non_zero_mask = actual != 0
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
            else:
                mape = np.inf
            
            # sMAPE (Symmetric Mean Absolute Percentage Error)
            smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
            
            # WAPE (Weighted Absolute Percentage Error)
            wape = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
            
            # Directional accuracy
            actual_direction = np.diff(actual) > 0
            predicted_direction = np.diff(predicted) > 0
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            # Bias
            bias = np.mean(predicted - actual)
            
            # Normalized metrics
            normalized_mae = mae / np.mean(actual)
            normalized_rmse = rmse / np.mean(actual)
            
            metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape),
                'smape': float(smape),
                'wape': float(wape),
                'bias': float(bias),
                'directional_accuracy': float(directional_accuracy),
                'normalized_mae': float(normalized_mae),
                'normalized_rmse': float(normalized_rmse),
                'n_samples': len(actual)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {'error': str(e)}
    
    def plot_predictions(self, actual, predicted, dates=None, title="Predictions vs Actual", save_path=None):
        """Plot predictions against actual values"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Time series plot
            if dates is not None:
                ax1.plot(dates, actual, label='Actual', alpha=0.7)
                ax1.plot(dates, predicted, label='Predicted', alpha=0.7)
            else:
                ax1.plot(actual, label='Actual', alpha=0.7)
                ax1.plot(predicted, label='Predicted', alpha=0.7)
            
            ax1.set_title(title)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Energy Demand (MW)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Scatter plot
            ax2.scatter(actual, predicted, alpha=0.5)
            
            # Perfect prediction line
            min_val = min(np.min(actual), np.min(predicted))
            max_val = max(np.max(actual), np.max(predicted))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Predicted')
            ax2.set_title('Predicted vs Actual Scatter Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            return None
    
    def plot_residuals(self, actual, predicted, dates=None, title="Residual Analysis", save_path=None):
        """Plot residual analysis"""
        try:
            residuals = actual - predicted
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Residuals over time
            if dates is not None:
                ax1.plot(dates, residuals)
            else:
                ax1.plot(residuals)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_title('Residuals Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Residuals')
            ax1.grid(True, alpha=0.3)
            
            # Residuals vs predicted
            ax2.scatter(predicted, residuals, alpha=0.5)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('Residuals vs Predicted')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.grid(True, alpha=0.3)
            
            # Histogram of residuals
            ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax3.set_title('Distribution of Residuals')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot of Residuals')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting residuals: {str(e)}")
            return None
    
    def plot_error_distribution(self, actual, predicted, title="Error Distribution", save_path=None):
        """Plot error distribution analysis"""
        try:
            errors = predicted - actual
            percentage_errors = (errors / actual) * 100
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Absolute errors
            ax1.hist(np.abs(errors), bins=30, alpha=0.7, edgecolor='black')
            ax1.set_title('Distribution of Absolute Errors')
            ax1.set_xlabel('Absolute Error')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Percentage errors
            ax2.hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
            ax2.set_title('Distribution of Percentage Errors')
            ax2.set_xlabel('Percentage Error (%)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Box plot of errors by hour (if applicable)
            if len(actual) > 24:
                hours = np.tile(np.arange(24), len(actual) // 24 + 1)[:len(actual)]
                error_by_hour = pd.DataFrame({'hour': hours, 'error': errors})
                error_by_hour.boxplot(column='error', by='hour', ax=ax3)
                ax3.set_title('Errors by Hour of Day')
                ax3.set_xlabel('Hour')
                ax3.set_ylabel('Error')
            else:
                ax3.text(0.5, 0.5, 'Not enough data\nfor hourly analysis', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Errors by Hour of Day')
            
            # Cumulative error
            cumulative_error = np.cumsum(np.abs(errors))
            ax4.plot(cumulative_error)
            ax4.set_title('Cumulative Absolute Error')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Cumulative Absolute Error')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting error distribution: {str(e)}")
            return None
    
    def compare_models(self, actual, predictions_dict, dates=None, save_path=None):
        """Compare multiple models"""
        try:
            n_models = len(predictions_dict)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # Time series comparison
            if dates is not None:
                axes[0].plot(dates, actual, label='Actual', linewidth=2, alpha=0.8)
                for model_name, predictions in predictions_dict.items():
                    axes[0].plot(dates, predictions, label=model_name, alpha=0.7)
            else:
                axes[0].plot(actual, label='Actual', linewidth=2, alpha=0.8)
                for model_name, predictions in predictions_dict.items():
                    axes[0].plot(predictions, label=model_name, alpha=0.7)
            
            axes[0].set_title('Model Predictions Comparison')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Energy Demand (MW)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Metrics comparison
            metrics_data = []
            for model_name, predictions in predictions_dict.items():
                metrics = self.calculate_metrics(actual, predictions)
                metrics['model'] = model_name
                metrics_data.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # MAPE comparison
            axes[1].bar(metrics_df['model'], metrics_df['mape'])
            axes[1].set_title('MAPE Comparison')
            axes[1].set_ylabel('MAPE (%)')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
            
            # RMSE comparison
            axes[2].bar(metrics_df['model'], metrics_df['rmse'])
            axes[2].set_title('RMSE Comparison')
            axes[2].set_ylabel('RMSE')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3)
            
            # R² comparison
            axes[3].bar(metrics_df['model'], metrics_df['r2'])
            axes[3].set_title('R² Comparison')
            axes[3].set_ylabel('R²')
            axes[3].tick_params(axis='x', rotation=45)
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig, metrics_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return None, None
    
    def forecast_accuracy_by_horizon(self, actual, predicted, horizons=[1, 6, 12, 24, 48]):
        """Calculate forecast accuracy by prediction horizon"""
        try:
            accuracy_results = {}
            
            for horizon in horizons:
                if horizon >= len(actual):
                    continue
                
                # Calculate metrics for each horizon
                horizon_actual = actual[horizon:]
                horizon_predicted = predicted[:-horizon] if horizon > 0 else predicted
                
                if len(horizon_actual) != len(horizon_predicted):
                    min_len = min(len(horizon_actual), len(horizon_predicted))
                    horizon_actual = horizon_actual[:min_len]
                    horizon_predicted = horizon_predicted[:min_len]
                
                metrics = self.calculate_metrics(horizon_actual, horizon_predicted)
                accuracy_results[f'horizon_{horizon}'] = metrics
            
            return accuracy_results
            
        except Exception as e:
            logger.error(f"Error calculating accuracy by horizon: {str(e)}")
            return {}
    
    def seasonal_accuracy_analysis(self, actual, predicted, dates):
        """Analyze forecast accuracy by season"""
        try:
            if dates is None:
                return {}
            
            # Create DataFrame
            df = pd.DataFrame({
                'actual': actual,
                'predicted': predicted,
                'date': dates
            })
            
            # Add seasonal information
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            # Calculate metrics by season
            seasonal_results = {}
            for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
                season_data = df[df['season'] == season]
                if len(season_data) > 0:
                    metrics = self.calculate_metrics(
                        season_data['actual'].values,
                        season_data['predicted'].values
                    )
                    seasonal_results[season] = metrics
            
            return seasonal_results
            
        except Exception as e:
            logger.error(f"Error in seasonal accuracy analysis: {str(e)}")
            return {}
    
    def generate_evaluation_report(self, actual, predicted, dates=None, model_name="Model"):
        """Generate comprehensive evaluation report"""
        try:
            report = {
                'model_name': model_name,
                'evaluation_date': datetime.now().isoformat(),
                'data_period': {
                    'start': dates[0].isoformat() if dates is not None else 'N/A',
                    'end': dates[-1].isoformat() if dates is not None else 'N/A',
                    'n_samples': len(actual)
                }
            }
            
            # Basic metrics
            report['metrics'] = self.calculate_metrics(actual, predicted)
            
            # Accuracy by horizon
            report['horizon_accuracy'] = self.forecast_accuracy_by_horizon(actual, predicted)
            
            # Seasonal accuracy
            if dates is not None:
                report['seasonal_accuracy'] = self.seasonal_accuracy_analysis(actual, predicted, dates)
            
            # Summary statistics
            report['summary'] = {
                'best_metric': 'R²',
                'best_value': report['metrics']['r2'],
                'worst_metric': 'MAPE',
                'worst_value': report['metrics']['mape'],
                'overall_performance': 'Good' if report['metrics']['mape'] < 10 else 'Fair' if report['metrics']['mape'] < 20 else 'Poor'
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            return {'error': str(e)}

