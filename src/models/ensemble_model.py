
"""
Ensemble model combining multiple forecasting approaches
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import logging

logger = logging.getLogger(__name__)

class EnsembleForecaster:
    """Ensemble model combining multiple forecasting models"""
    
    def __init__(self, models=None, combination_method='weighted_average'):
        self.models = models or []
        self.combination_method = combination_method
        self.weights = None
        self.meta_model = None
        self.is_fitted = False
        
    def add_model(self, model):
        """Add a model to the ensemble"""
        self.models.append(model)
    
    def fit_weights(self, predictions_list, actual_values, method='optimal'):
        """Fit ensemble weights based on validation performance"""
        try:
            logger.info(f"Fitting ensemble weights using {method} method...")
            
            if len(predictions_list) != len(self.models):
                raise ValueError("Number of prediction arrays must match number of models")
            
            predictions_array = np.column_stack(predictions_list)
            
            if method == 'equal':
                # Equal weights
                self.weights = np.ones(len(self.models)) / len(self.models)
                
            elif method == 'inverse_error':
                # Weights inversely proportional to individual model errors
                errors = []
                for pred in predictions_list:
                    mse = mean_squared_error(actual_values, pred)
                    errors.append(mse)
                
                # Inverse of errors (add small epsilon to avoid division by zero)
                inv_errors = 1.0 / (np.array(errors) + 1e-8)
                self.weights = inv_errors / np.sum(inv_errors)
                
            elif method == 'optimal':
                # Optimal weights using linear regression
                reg = LinearRegression(fit_intercept=False, positive=True)
                reg.fit(predictions_array, actual_values)
                
                # Normalize weights to sum to 1
                raw_weights = reg.coef_
                self.weights = raw_weights / np.sum(raw_weights)
                
            elif method == 'stacking':
                # Use a meta-model for stacking
                self.meta_model = LinearRegression()
                self.meta_model.fit(predictions_array, actual_values)
                self.weights = None  # Not used in stacking
                
            else:
                raise ValueError(f"Unknown combination method: {method}")
            
            self.is_fitted = True
            
            if self.weights is not None:
                logger.info(f"Ensemble weights: {self.weights}")
            else:
                logger.info("Using meta-model for ensemble combination")
                
        except Exception as e:
            logger.error(f"Error fitting ensemble weights: {str(e)}")
            raise
    
    def predict(self, predictions_list):
        """Make ensemble predictions"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble must be fitted before making predictions")
            
            if len(predictions_list) != len(self.models):
                raise ValueError("Number of prediction arrays must match number of models")
            
            predictions_array = np.column_stack(predictions_list)
            
            if self.combination_method == 'stacking' and self.meta_model is not None:
                # Use meta-model
                ensemble_pred = self.meta_model.predict(predictions_array)
            else:
                # Use weighted average
                ensemble_pred = np.dot(predictions_array, self.weights)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {str(e)}")
            raise
    
    def predict_with_individual(self, predictions_list):
        """Make predictions and return both ensemble and individual model predictions"""
        try:
            ensemble_pred = self.predict(predictions_list)
            
            return {
                'ensemble': ensemble_pred,
                'individual': {f'model_{i}': pred for i, pred in enumerate(predictions_list)},
                'weights': self.weights.tolist() if self.weights is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with individual results: {str(e)}")
            raise
    
    def evaluate_ensemble(self, predictions_list, actual_values):
        """Evaluate ensemble performance"""
        try:
            ensemble_pred = self.predict(predictions_list)
            
            # Calculate ensemble metrics
            mse = mean_squared_error(actual_values, ensemble_pred)
            mae = np.mean(np.abs(actual_values - ensemble_pred))
            rmse = np.sqrt(mse)
            
            # Calculate MAPE
            mask = actual_values != 0
            mape = np.mean(np.abs((actual_values[mask] - ensemble_pred[mask]) / actual_values[mask])) * 100
            
            ensemble_metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            }
            
            # Calculate individual model metrics for comparison
            individual_metrics = {}
            for i, pred in enumerate(predictions_list):
                mse_ind = mean_squared_error(actual_values, pred)
                mae_ind = np.mean(np.abs(actual_values - pred))
                rmse_ind = np.sqrt(mse_ind)
                mape_ind = np.mean(np.abs((actual_values[mask] - pred[mask]) / actual_values[mask])) * 100
                
                individual_metrics[f'model_{i}'] = {
                    'mse': float(mse_ind),
                    'mae': float(mae_ind),
                    'rmse': float(rmse_ind),
                    'mape': float(mape_ind)
                }
            
            return {
                'ensemble': ensemble_metrics,
                'individual': individual_metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            raise
    
    def optimize_weights(self, predictions_list, actual_values, methods=['equal', 'inverse_error', 'optimal']):
        """Try different weight optimization methods and select the best"""
        try:
            logger.info("Optimizing ensemble weights...")
            
            best_method = None
            best_score = float('inf')
            best_weights = None
            best_meta_model = None
            
            for method in methods:
                try:
                    # Fit weights with current method
                    self.fit_weights(predictions_list, actual_values, method)
                    
                    # Evaluate performance
                    ensemble_pred = self.predict(predictions_list)
                    mse = mean_squared_error(actual_values, ensemble_pred)
                    
                    logger.info(f"Method {method}: MSE = {mse:.4f}")
                    
                    if mse < best_score:
                        best_score = mse
                        best_method = method
                        best_weights = self.weights.copy() if self.weights is not None else None
                        best_meta_model = self.meta_model
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate method {method}: {str(e)}")
                    continue
            
            # Set best configuration
            if best_method:
                self.combination_method = best_method
                self.weights = best_weights
                self.meta_model = best_meta_model
                
                logger.info(f"Best ensemble method: {best_method} with MSE: {best_score:.4f}")
                
                return best_method, best_score
            else:
                logger.warning("No valid ensemble method found")
                return None, None
                
        except Exception as e:
            logger.error(f"Error optimizing weights: {str(e)}")
            raise
    
    def get_feature_importance(self, predictions_list, actual_values):
        """Get feature importance for ensemble models"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble must be fitted before getting feature importance")
            
            if self.weights is not None:
                # For weighted average, weights represent importance
                importance = {
                    f'model_{i}': float(weight) 
                    for i, weight in enumerate(self.weights)
                }
            elif self.meta_model is not None:
                # For stacking, use meta-model coefficients
                importance = {
                    f'model_{i}': float(coef) 
                    for i, coef in enumerate(self.meta_model.coef_)
                }
            else:
                importance = {}
            
            return importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def save(self, filepath):
        """Save the ensemble model"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble must be fitted before saving")
            
            ensemble_data = {
                'combination_method': self.combination_method,
                'weights': self.weights,
                'meta_model': self.meta_model,
                'is_fitted': self.is_fitted,
                'num_models': len(self.models)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            logger.info(f"Ensemble model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble model: {str(e)}")
            raise
    
    def load(self, filepath):
        """Load the ensemble model"""
        try:
            with open(filepath, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.combination_method = ensemble_data['combination_method']
            self.weights = ensemble_data['weights']
            self.meta_model = ensemble_data['meta_model']
            self.is_fitted = ensemble_data['is_fitted']
            
            logger.info(f"Ensemble model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get ensemble model information"""
        return {
            'model_type': 'Ensemble',
            'combination_method': self.combination_method,
            'num_models': len(self.models),
            'weights': self.weights.tolist() if self.weights is not None else None,
            'has_meta_model': self.meta_model is not None,
            'is_fitted': self.is_fitted
        }
    
    def cross_validate_ensemble(self, predictions_list, actual_values, cv_folds=5):
        """Cross-validate ensemble performance"""
        try:
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            predictions_array = np.column_stack(predictions_list)
            
            for train_idx, val_idx in kf.split(predictions_array):
                # Split data
                train_preds = [pred[train_idx] for pred in predictions_list]
                val_preds = [pred[val_idx] for pred in predictions_list]
                train_actual = actual_values[train_idx]
                val_actual = actual_values[val_idx]
                
                # Fit ensemble on training fold
                temp_ensemble = EnsembleForecaster(combination_method=self.combination_method)
                temp_ensemble.fit_weights(train_preds, train_actual)
                
                # Predict on validation fold
                val_ensemble_pred = temp_ensemble.predict(val_preds)
                
                # Calculate score
                mse = mean_squared_error(val_actual, val_ensemble_pred)
                cv_scores.append(mse)
            
            cv_results = {
                'mean_mse': np.mean(cv_scores),
                'std_mse': np.std(cv_scores),
                'cv_scores': cv_scores
            }
            
            logger.info(f"Cross-validation results: Mean MSE = {cv_results['mean_mse']:.4f} ± {cv_results['std_mse']:.4f}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

