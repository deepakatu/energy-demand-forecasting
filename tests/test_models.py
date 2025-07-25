
"""
Unit tests for energy forecasting models
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.prophet_model import ProphetForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.ensemble_model import EnsembleForecaster
from src.data.data_loader import EnergyDataLoader
from src.features.feature_engineering import FeatureEngineer

class TestEnergyModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.data_loader = EnergyDataLoader()
        self.data = self.data_loader.load_sample_data(
            start_date='2023-01-01', 
            end_date='2023-01-07'
        )
        self.feature_engineer = FeatureEngineer()
        
    def test_prophet_model(self):
        """Test Prophet model"""
        model = ProphetForecaster()
        
        # Prepare data
        prophet_data = self.feature_engineer.prepare_prophet_data(self.data)
        
        # Test fitting
        model.fit(prophet_data)
        self.assertTrue(model.is_fitted)
        
        # Test prediction
        predictions = model.predict(prophet_data)
        self.assertEqual(len(predictions), len(prophet_data))
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
    
    def test_lstm_model(self):
        """Test LSTM model"""
        model = LSTMForecaster(sequence_length=24, hidden_units=32)
        
        # Prepare data
        X, y = self.feature_engineer.prepare_lstm_data(self.data)
        
        # Test model building
        self.assertEqual(model.sequence_length, 24)
        self.assertEqual(model.hidden_units, 32)
        
        # Test fitting (with small data)
        if len(X) > 0:
            model.fit(X[:50], y[:50], epochs=2)
            self.assertTrue(model.is_fitted)
    
    def test_ensemble_model(self):
        """Test Ensemble model"""
        # Create mock models
        prophet_model = ProphetForecaster()
        lstm_model = LSTMForecaster()
        
        ensemble = EnsembleForecaster([prophet_model, lstm_model])
        
        # Test initialization
        self.assertEqual(len(ensemble.models), 2)
        self.assertEqual(ensemble.combination_method, 'weighted_average')
        
        # Test weight fitting with mock data
        mock_pred1 = np.random.randn(100)
        mock_pred2 = np.random.randn(100)
        mock_actual = np.random.randn(100)
        
        ensemble.fit_weights([mock_pred1, mock_pred2], mock_actual)
        self.assertTrue(ensemble.is_fitted)
        self.assertIsNotNone(ensemble.weights)

if __name__ == '__main__':
    unittest.main()
