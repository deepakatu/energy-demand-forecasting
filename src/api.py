
"""
FastAPI application for energy demand forecasting
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

from models.prophet_model import ProphetForecaster
from models.lstm_model import LSTMForecaster
from models.ensemble_model import EnsembleForecaster
from features.feature_engineering import FeatureEngineer
from data.data_loader import EnergyDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Energy Demand Forecasting API",
    description="API for energy demand forecasting using Prophet, LSTM, and ensemble models",
    version="1.0.0"
)

# Global variables for models
prophet_model = None
lstm_model = None
ensemble_model = None
feature_engineer = None

# Pydantic models
class ForecastRequest(BaseModel):
    horizon: int = 24  # Hours to forecast
    include_weather: bool = True
    confidence_interval: float = 0.95
    model_type: str = "ensemble"  # prophet, lstm, ensemble

class ForecastResponse(BaseModel):
    predictions: List[float]
    timestamps: List[str]
    model_used: str
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    processing_time: float

class ModelInfo(BaseModel):
    model_type: str
    is_loaded: bool
    last_trained: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global prophet_model, lstm_model, ensemble_model, feature_engineer
    
    try:
        logger.info("Loading models...")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Load models if they exist
        if os.path.exists('models/prophet_model.pkl'):
            prophet_model = ProphetForecaster()
            prophet_model.load('models/prophet_model.pkl')
            logger.info("Prophet model loaded")
        
        if os.path.exists('models/lstm_model.pkl'):
            lstm_model = LSTMForecaster()
            lstm_model.load('models/lstm_model.pkl')
            logger.info("LSTM model loaded")
        
        if os.path.exists('models/ensemble_model.pkl'):
            ensemble_model = EnsembleForecaster()
            ensemble_model.load('models/ensemble_model.pkl')
            logger.info("Ensemble model loaded")
        
        logger.info("API startup completed")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Energy Demand Forecasting API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "prophet": prophet_model is not None and prophet_model.is_fitted,
            "lstm": lstm_model is not None and lstm_model.is_fitted,
            "ensemble": ensemble_model is not None and ensemble_model.is_fitted
        }
    }

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Generate energy demand forecast"""
    try:
        start_time = datetime.now()
        
        # Validate request
        if request.horizon <= 0 or request.horizon > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Horizon must be between 1 and 168 hours")
        
        # Select model
        if request.model_type == "prophet" and prophet_model is not None:
            model = prophet_model
        elif request.model_type == "lstm" and lstm_model is not None:
            model = lstm_model
        elif request.model_type == "ensemble" and ensemble_model is not None:
            model = ensemble_model
        else:
            raise HTTPException(status_code=400, detail=f"Model {request.model_type} not available")
        
        # Generate future timestamps
        current_time = datetime.now()
        future_timestamps = [
            current_time + timedelta(hours=i) for i in range(1, request.horizon + 1)
        ]
        
        # Create future data with features
        future_data = pd.DataFrame({
            'datetime': future_timestamps
        }).set_index('datetime')
        
        # Add features
        future_data = feature_engineer.add_temporal_features(future_data)
        if request.include_weather:
            future_data = feature_engineer.add_weather_features(future_data)
        
        # Make predictions based on model type
        if request.model_type == "prophet":
            prophet_data = feature_engineer.prepare_prophet_data(future_data)
            if request.confidence_interval > 0:
                forecast_result = model.predict_with_uncertainty(prophet_data)
                predictions = forecast_result['predictions'].tolist()
                confidence_intervals = {
                    'lower': forecast_result['lower_bound'].tolist(),
                    'upper': forecast_result['upper_bound'].tolist()
                }
            else:
                predictions = model.predict(prophet_data).tolist()
                confidence_intervals = None
                
        elif request.model_type == "lstm":
            # For LSTM, we need sequence data - this is simplified
            # In practice, you'd need recent historical data
            predictions = np.random.uniform(800, 1200, request.horizon).tolist()  # Mock prediction
            confidence_intervals = None
            
        elif request.model_type == "ensemble":
            # For ensemble, combine predictions from both models
            predictions = np.random.uniform(900, 1100, request.horizon).tolist()  # Mock prediction
            confidence_intervals = None
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ForecastResponse(
            predictions=predictions,
            timestamps=[ts.isoformat() for ts in future_timestamps],
            model_used=request.model_type,
            confidence_intervals=confidence_intervals,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    try:
        models_info = {}
        
        if prophet_model is not None:
            models_info['prophet'] = ModelInfo(
                model_type="Prophet",
                is_loaded=prophet_model.is_fitted,
                performance_metrics={"mape": 3.2, "rmse": 145.7, "r2": 0.94}  # Mock metrics
            )
        
        if lstm_model is not None:
            models_info['lstm'] = ModelInfo(
                model_type="LSTM",
                is_loaded=lstm_model.is_fitted,
                performance_metrics={"mape": 2.8, "rmse": 132.4, "r2": 0.95}  # Mock metrics
            )
        
        if ensemble_model is not None:
            models_info['ensemble'] = ModelInfo(
                model_type="Ensemble",
                is_loaded=ensemble_model.is_fitted,
                performance_metrics={"mape": 2.5, "rmse": 128.9, "r2": 0.96}  # Mock metrics
            )
        
        return models_info
        
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models info: {str(e)}")

@app.post("/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    try:
        # Add retraining task to background
        background_tasks.add_task(retrain_models_task)
        
        return {
            "message": "Model retraining initiated",
            "status": "in_progress",
            "estimated_completion": "30-60 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error initiating retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate retraining: {str(e)}")

async def retrain_models_task():
    """Background task for model retraining"""
    try:
        logger.info("Starting model retraining...")
        
        # Load fresh data
        data_loader = EnergyDataLoader()
        data = data_loader.load_sample_data()
        
        # Add features
        data = feature_engineer.add_temporal_features(data)
        data = feature_engineer.add_weather_features(data)
        data = feature_engineer.add_lag_features(data)
        
        # Split data
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        # Retrain Prophet
        global prophet_model
        prophet_model = ProphetForecaster()
        prophet_data = feature_engineer.prepare_prophet_data(train_data)
        prophet_model.fit(prophet_data)
        prophet_model.save('models/prophet_model.pkl')
        
        # Retrain LSTM (simplified)
        global lstm_model
        lstm_model = LSTMForecaster()
        X_train, y_train = feature_engineer.prepare_lstm_data(train_data)
        X_val, y_val = feature_engineer.prepare_lstm_data(val_data)
        lstm_model.fit(X_train, y_train, X_val, y_val, epochs=20)
        lstm_model.save('models/lstm_model.pkl')
        
        # Retrain Ensemble
        global ensemble_model
        ensemble_model = EnsembleForecaster([prophet_model, lstm_model])
        
        # Get validation predictions
        val_prophet_data = feature_engineer.prepare_prophet_data(val_data)
        prophet_pred = prophet_model.predict(val_prophet_data)
        lstm_pred = lstm_model.predict(X_val).flatten()
        
        ensemble_model.fit_weights([prophet_pred, lstm_pred], val_data['demand'].values)
        ensemble_model.save('models/ensemble_model.pkl')
        
        logger.info("Model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model retraining: {str(e)}")

@app.get("/data/sample")
async def get_sample_data(days: int = 7):
    """Get sample energy demand data"""
    try:
        if days <= 0 or days > 30:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 30")
        
        # Generate sample data
        data_loader = EnergyDataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = data_loader.load_sample_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Convert to JSON-serializable format
        sample_data = {
            'timestamps': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'demand': data['demand'].tolist(),
            'temperature': data['temperature'].tolist(),
            'humidity': data['humidity'].tolist(),
            'wind_speed': data['wind_speed'].tolist()
        }
        
        return sample_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sample data: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get forecasting statistics"""
    try:
        # Mock statistics (would be calculated from actual data/models)
        stats = {
            "total_forecasts_generated": 15420,
            "average_mape": 2.8,
            "best_performing_model": "ensemble",
            "last_model_update": "2024-01-20T10:30:00Z",
            "data_coverage": {
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "total_hours": 35064
            },
            "model_performance": {
                "prophet": {"mape": 3.2, "rmse": 145.7, "r2": 0.94},
                "lstm": {"mape": 2.8, "rmse": 132.4, "r2": 0.95},
                "ensemble": {"mape": 2.5, "rmse": 128.9, "r2": 0.96}
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
