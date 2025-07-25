
"""
Prophet-based energy demand forecasting model
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProphetForecaster:
    """Prophet-based forecasting model for energy demand"""
    
    def __init__(self, 
                 seasonality_mode='multiplicative',
                 yearly_seasonality=True,
                 weekly_seasonality=True,
                 daily_seasonality=True,
                 holidays=None):
        
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.model = None
        self.is_fitted = False
        
    def fit(self, data):
        """Fit the Prophet model"""
        try:
            logger.info("Fitting Prophet model...")
            
            # Initialize Prophet model
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                holidays=self.holidays
            )
            
            # Add custom seasonalities
            self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            self.model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            
            # Add regressors if available
            if 'temperature' in data.columns:
                self.model.add_regressor('temperature')
            if 'humidity' in data.columns:
                self.model.add_regressor('humidity')
            if 'wind_speed' in data.columns:
                self.model.add_regressor('wind_speed')
            if 'is_holiday' in data.columns:
                self.model.add_regressor('is_holiday')
            
            # Fit the model
            self.model.fit(data)
            self.is_fitted = True
            
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            raise
    
    def predict(self, future_data):
        """Make predictions using the fitted model"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            # Make predictions
            forecast = self.model.predict(future_data)
            
            # Return predictions (yhat column)
            return forecast['yhat'].values
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_with_uncertainty(self, future_data):
        """Make predictions with uncertainty intervals"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            forecast = self.model.predict(future_data)
            
            return {
                'predictions': forecast['yhat'].values,
                'lower_bound': forecast['yhat_lower'].values,
                'upper_bound': forecast['yhat_upper'].values,
                'trend': forecast['trend'].values,
                'seasonal': forecast.get('seasonal', np.zeros(len(forecast))).values
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with uncertainty: {str(e)}")
            raise
    
    def forecast_future(self, periods, freq='H', include_history=False):
        """Forecast future periods"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before forecasting")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods, 
                freq=freq, 
                include_history=include_history
            )
            
            # Add regressor values for future periods (using simple forward fill)
            if hasattr(self.model, 'extra_regressors'):
                for regressor in self.model.extra_regressors:
                    if regressor not in future.columns:
                        # Use mean value for missing regressors
                        future[regressor] = 0  # Would use historical mean in production
            
            # Make forecast
            forecast = self.model.predict(future)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting future: {str(e)}")
            raise
    
    def get_components(self, forecast_data):
        """Get forecast components (trend, seasonality, etc.)"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting components")
            
            forecast = self.model.predict(forecast_data)
            
            components = {
                'trend': forecast['trend'].values,
                'yearly': forecast.get('yearly', np.zeros(len(forecast))).values,
                'weekly': forecast.get('weekly', np.zeros(len(forecast))).values,
                'daily': forecast.get('daily', np.zeros(len(forecast))).values
            }
            
            # Add custom seasonalities if they exist
            if 'monthly' in forecast.columns:
                components['monthly'] = forecast['monthly'].values
            if 'quarterly' in forecast.columns:
                components['quarterly'] = forecast['quarterly'].values
            
            return components
            
        except Exception as e:
            logger.error(f"Error getting components: {str(e)}")
            raise
    
    def cross_validate(self, data, initial='730 days', period='180 days', horizon='30 days'):
        """Perform cross-validation"""
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            
            if not self.is_fitted:
                self.fit(data)
            
            # Perform cross-validation
            cv_results = cross_validation(
                self.model, 
                initial=initial, 
                period=period, 
                horizon=horizon
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            return cv_results, metrics
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def plot_forecast(self, forecast_data, save_path=None):
        """Plot forecast results"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before plotting")
            
            forecast = self.model.predict(forecast_data)
            
            # Create plot
            fig = self.model.plot(forecast)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {str(e)}")
            raise
    
    def plot_components(self, forecast_data, save_path=None):
        """Plot forecast components"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before plotting")
            
            forecast = self.model.predict(forecast_data)
            
            # Create components plot
            fig = self.model.plot_components(forecast)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting components: {str(e)}")
            raise
    
    def save(self, filepath):
        """Save the fitted model"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before saving")
            
            model_data = {
                'model': self.model,
                'seasonality_mode': self.seasonality_mode,
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality,
                'holidays': self.holidays,
                'is_fitted': self.is_fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, filepath):
        """Load a fitted model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.seasonality_mode = model_data['seasonality_mode']
            self.yearly_seasonality = model_data['yearly_seasonality']
            self.weekly_seasonality = model_data['weekly_seasonality']
            self.daily_seasonality = model_data['daily_seasonality']
            self.holidays = model_data['holidays']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': 'Prophet',
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'is_fitted': self.is_fitted,
            'has_regressors': hasattr(self.model, 'extra_regressors') and len(self.model.extra_regressors) > 0 if self.model else False
        }
    
    def tune_hyperparameters(self, data, param_grid=None):
        """Tune hyperparameters using cross-validation"""
        try:
            if param_grid is None:
                param_grid = {
                    'seasonality_mode': ['additive', 'multiplicative'],
                    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
                }
            
            best_params = None
            best_score = float('inf')
            
            # Simple grid search (in production, use more sophisticated methods)
            for seasonality_mode in param_grid.get('seasonality_mode', [self.seasonality_mode]):
                for changepoint_prior in param_grid.get('changepoint_prior_scale', [0.05]):
                    for seasonality_prior in param_grid.get('seasonality_prior_scale', [10.0]):
                        
                        # Create model with current parameters
                        model = Prophet(
                            seasonality_mode=seasonality_mode,
                            changepoint_prior_scale=changepoint_prior,
                            seasonality_prior_scale=seasonality_prior,
                            yearly_seasonality=self.yearly_seasonality,
                            weekly_seasonality=self.weekly_seasonality,
                            daily_seasonality=self.daily_seasonality
                        )
                        
                        try:
                            # Fit and evaluate
                            model.fit(data)
                            
                            # Simple validation split
                            train_size = int(0.8 * len(data))
                            train_data = data[:train_size]
                            val_data = data[train_size:]
                            
                            model.fit(train_data)
                            forecast = model.predict(val_data[['ds']])
                            
                            # Calculate MAPE
                            actual = val_data['y'].values
                            predicted = forecast['yhat'].values
                            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                            
                            if mape < best_score:
                                best_score = mape
                                best_params = {
                                    'seasonality_mode': seasonality_mode,
                                    'changepoint_prior_scale': changepoint_prior,
                                    'seasonality_prior_scale': seasonality_prior
                                }
                        
                        except Exception as e:
                            logger.warning(f"Failed to evaluate parameters: {str(e)}")
                            continue
            
            if best_params:
                logger.info(f"Best parameters found: {best_params} with MAPE: {best_score:.2f}%")
                
                # Update model with best parameters
                self.seasonality_mode = best_params['seasonality_mode']
                
                return best_params, best_score
            else:
                logger.warning("No valid parameters found during tuning")
                return None, None
                
        except Exception as e:
            logger.error(f"Error tuning hyperparameters: {str(e)}")
            raise

