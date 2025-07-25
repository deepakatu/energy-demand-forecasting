
"""
Model training pipeline for energy demand forecasting
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.prophet_model import ProphetForecaster
from models.lstm_model import LSTMForecaster
from models.ensemble_model import EnsembleForecaster
from features.feature_engineering import FeatureEngineer
from data.data_loader import EnergyDataLoader
from utils.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("energy_demand_forecasting")

def train_prophet_model(train_data, val_data, feature_engineer):
    """Train Prophet model"""
    logger.info("Training Prophet model...")
    
    with mlflow.start_run(run_name="prophet_model"):
        # Initialize model
        prophet_model = ProphetForecaster()
        
        # Prepare data
        prophet_data = feature_engineer.prepare_prophet_data(train_data)
        
        # Train model
        prophet_model.fit(prophet_data)
        
        # Make predictions
        val_prophet_data = feature_engineer.prepare_prophet_data(val_data)
        predictions = prophet_model.predict(val_prophet_data)
        
        # Calculate metrics
        actual = val_data['demand'].values
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        # Log metrics
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(prophet_model, "prophet_model")
        
        logger.info(f"Prophet - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        
        return prophet_model, {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

def train_lstm_model(train_data, val_data, feature_engineer):
    """Train LSTM model"""
    logger.info("Training LSTM model...")
    
    with mlflow.start_run(run_name="lstm_model"):
        # Initialize model
        lstm_model = LSTMForecaster(
            sequence_length=24,  # 24 hours lookback
            hidden_units=64,
            dropout_rate=0.2
        )
        
        # Prepare data
        X_train, y_train = feature_engineer.prepare_lstm_data(train_data)
        X_val, y_val = feature_engineer.prepare_lstm_data(val_data)
        
        # Train model
        history = lstm_model.fit(X_train, y_train, X_val, y_val, epochs=50)
        
        # Make predictions
        predictions = lstm_model.predict(X_val)
        
        # Calculate metrics
        actual = y_val.flatten()
        pred_flat = predictions.flatten()
        
        mape = np.mean(np.abs((actual - pred_flat) / actual)) * 100
        rmse = np.sqrt(mean_squared_error(actual, pred_flat))
        mae = mean_absolute_error(actual, pred_flat)
        r2 = r2_score(actual, pred_flat)
        
        # Log metrics
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model parameters
        mlflow.log_param("sequence_length", 24)
        mlflow.log_param("hidden_units", 64)
        mlflow.log_param("dropout_rate", 0.2)
        mlflow.log_param("epochs", 50)
        
        # Log model
        mlflow.tensorflow.log_model(lstm_model.model, "lstm_model")
        
        logger.info(f"LSTM - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        
        return lstm_model, {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

def train_ensemble_model(prophet_model, lstm_model, val_data, feature_engineer):
    """Train ensemble model"""
    logger.info("Training Ensemble model...")
    
    with mlflow.start_run(run_name="ensemble_model"):
        # Initialize ensemble
        ensemble_model = EnsembleForecaster([prophet_model, lstm_model])
        
        # Prepare validation data for both models
        val_prophet_data = feature_engineer.prepare_prophet_data(val_data)
        X_val, y_val = feature_engineer.prepare_lstm_data(val_data)
        
        # Get predictions from both models
        prophet_pred = prophet_model.predict(val_prophet_data)
        lstm_pred = lstm_model.predict(X_val).flatten()
        
        # Train ensemble weights
        ensemble_model.fit_weights(
            [prophet_pred, lstm_pred], 
            val_data['demand'].values
        )
        
        # Make ensemble predictions
        predictions = ensemble_model.predict([prophet_pred, lstm_pred])
        
        # Calculate metrics
        actual = val_data['demand'].values
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        # Log metrics
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log ensemble weights
        mlflow.log_param("prophet_weight", ensemble_model.weights[0])
        mlflow.log_param("lstm_weight", ensemble_model.weights[1])
        
        logger.info(f"Ensemble - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        logger.info(f"Ensemble weights - Prophet: {ensemble_model.weights[0]:.3f}, LSTM: {ensemble_model.weights[1]:.3f}")
        
        return ensemble_model, {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

def main():
    """Main training pipeline"""
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Load data
        logger.info("Loading data...")
        data_loader = EnergyDataLoader()
        data = data_loader.load_sample_data()
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Add features
        logger.info("Engineering features...")
        data = feature_engineer.add_temporal_features(data)
        data = feature_engineer.add_weather_features(data)
        data = feature_engineer.add_lag_features(data)
        
        # Split data
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        
        train_data = data[:train_size].copy()
        val_data = data[train_size:train_size + val_size].copy()
        test_data = data[train_size + val_size:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Train models
        prophet_model, prophet_metrics = train_prophet_model(train_data, val_data, feature_engineer)
        lstm_model, lstm_metrics = train_lstm_model(train_data, val_data, feature_engineer)
        ensemble_model, ensemble_metrics = train_ensemble_model(
            prophet_model, lstm_model, val_data, feature_engineer
        )
        
        # Save models
        logger.info("Saving models...")
        os.makedirs('models', exist_ok=True)
        
        prophet_model.save('models/prophet_model.pkl')
        lstm_model.save('models/lstm_model.h5')
        ensemble_model.save('models/ensemble_model.pkl')
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        evaluator = ModelEvaluator()
        
        # Test Prophet
        test_prophet_data = feature_engineer.prepare_prophet_data(test_data)
        prophet_test_pred = prophet_model.predict(test_prophet_data)
        prophet_test_metrics = evaluator.calculate_metrics(test_data['demand'].values, prophet_test_pred)
        
        # Test LSTM
        X_test, y_test = feature_engineer.prepare_lstm_data(test_data)
        lstm_test_pred = lstm_model.predict(X_test).flatten()
        lstm_test_metrics = evaluator.calculate_metrics(y_test.flatten(), lstm_test_pred)
        
        # Test Ensemble
        ensemble_test_pred = ensemble_model.predict([prophet_test_pred, lstm_test_pred])
        ensemble_test_metrics = evaluator.calculate_metrics(test_data['demand'].values, ensemble_test_pred)
        
        # Print final results
        logger.info("\n" + "="*50)
        logger.info("FINAL TEST SET RESULTS")
        logger.info("="*50)
        logger.info(f"Prophet  - MAPE: {prophet_test_metrics['mape']:.2f}%, RMSE: {prophet_test_metrics['rmse']:.2f}")
        logger.info(f"LSTM     - MAPE: {lstm_test_metrics['mape']:.2f}%, RMSE: {lstm_test_metrics['rmse']:.2f}")
        logger.info(f"Ensemble - MAPE: {ensemble_test_metrics['mape']:.2f}%, RMSE: {ensemble_test_metrics['rmse']:.2f}")
        logger.info("="*50)
        
        # Log final test metrics to MLflow
        with mlflow.start_run(run_name="final_test_results"):
            mlflow.log_metric("prophet_test_mape", prophet_test_metrics['mape'])
            mlflow.log_metric("lstm_test_mape", lstm_test_metrics['mape'])
            mlflow.log_metric("ensemble_test_mape", ensemble_test_metrics['mape'])
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

