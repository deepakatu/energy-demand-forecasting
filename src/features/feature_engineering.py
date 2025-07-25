
"""
Feature engineering for energy demand forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering class for energy demand forecasting"""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
        
    def add_temporal_features(self, data):
        """Add temporal features"""
        try:
            logger.info("Adding temporal features...")
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data['datetime'] = pd.to_datetime(data['datetime'])
                data = data.set_index('datetime')
            
            # Basic temporal features
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['day_of_month'] = data.index.day
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['year'] = data.index.year
            
            # Cyclical encoding
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            # Binary features
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            data['is_business_hour'] = ((data['hour'] >= 9) & (data['hour'] <= 17)).astype(int)
            data['is_peak_hour'] = ((data['hour'] >= 17) & (data['hour'] <= 21)).astype(int)
            
            # Season features
            data['is_winter'] = data['month'].isin([12, 1, 2]).astype(int)
            data['is_spring'] = data['month'].isin([3, 4, 5]).astype(int)
            data['is_summer'] = data['month'].isin([6, 7, 8]).astype(int)
            data['is_autumn'] = data['month'].isin([9, 10, 11]).astype(int)
            
            logger.info("Temporal features added successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error adding temporal features: {str(e)}")
            raise
    
    def add_weather_features(self, data):
        """Add weather-related features"""
        try:
            logger.info("Adding weather features...")
            
            # Generate synthetic weather data if not present
            if 'temperature' not in data.columns:
                # Simulate temperature with seasonal pattern
                days_of_year = data.index.dayofyear
                base_temp = 15 + 10 * np.sin(2 * np.pi * (days_of_year - 80) / 365)
                daily_variation = 5 * np.sin(2 * np.pi * data.index.hour / 24)
                noise = np.random.normal(0, 2, len(data))
                data['temperature'] = base_temp + daily_variation + noise
            
            if 'humidity' not in data.columns:
                # Simulate humidity
                base_humidity = 60 + 20 * np.sin(2 * np.pi * data.index.dayofyear / 365)
                noise = np.random.normal(0, 5, len(data))
                data['humidity'] = np.clip(base_humidity + noise, 0, 100)
            
            if 'wind_speed' not in data.columns:
                # Simulate wind speed
                data['wind_speed'] = np.random.exponential(3, len(data))
            
            # Derived weather features
            data['temp_squared'] = data['temperature'] ** 2
            data['temp_cubed'] = data['temperature'] ** 3
            
            # Cooling/Heating Degree Days
            data['cooling_degree_days'] = np.maximum(data['temperature'] - 18, 0)
            data['heating_degree_days'] = np.maximum(18 - data['temperature'], 0)
            
            # Weather comfort index
            data['heat_index'] = data['temperature'] + 0.5 * data['humidity']
            data['wind_chill'] = data['temperature'] - 0.7 * data['wind_speed']
            
            # Weather categories
            data['is_hot'] = (data['temperature'] > 25).astype(int)
            data['is_cold'] = (data['temperature'] < 5).astype(int)
            data['is_humid'] = (data['humidity'] > 70).astype(int)
            data['is_windy'] = (data['wind_speed'] > 5).astype(int)
            
            logger.info("Weather features added successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error adding weather features: {str(e)}")
            raise
    
    def add_lag_features(self, data, target_col='demand', lags=[1, 2, 3, 6, 12, 24, 48, 168]):
        """Add lagged features"""
        try:
            logger.info("Adding lag features...")
            
            for lag in lags:
                data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
            
            # Rolling statistics
            for window in [3, 6, 12, 24]:
                data[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
                data[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window=window).std()
                data[f'{target_col}_rolling_min_{window}'] = data[target_col].rolling(window=window).min()
                data[f'{target_col}_rolling_max_{window}'] = data[target_col].rolling(window=window).max()
            
            # Exponential moving averages
            for alpha in [0.1, 0.3, 0.5]:
                data[f'{target_col}_ema_{alpha}'] = data[target_col].ewm(alpha=alpha).mean()
            
            # Differences
            data[f'{target_col}_diff_1'] = data[target_col].diff(1)
            data[f'{target_col}_diff_24'] = data[target_col].diff(24)
            data[f'{target_col}_diff_168'] = data[target_col].diff(168)
            
            logger.info("Lag features added successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error adding lag features: {str(e)}")
            raise
    
    def add_holiday_features(self, data):
        """Add holiday and special event features"""
        try:
            logger.info("Adding holiday features...")
            
            # Create basic holiday indicators
            data['is_holiday'] = 0
            
            # New Year's Day
            data.loc[(data.index.month == 1) & (data.index.day == 1), 'is_holiday'] = 1
            
            # Christmas
            data.loc[(data.index.month == 12) & (data.index.day == 25), 'is_holiday'] = 1
            
            # Independence Day (US)
            data.loc[(data.index.month == 7) & (data.index.day == 4), 'is_holiday'] = 1
            
            # Thanksgiving (4th Thursday of November)
            for year in data.index.year.unique():
                thanksgiving = pd.Timestamp(year, 11, 1)
                # Find first Thursday
                while thanksgiving.weekday() != 3:
                    thanksgiving += pd.Timedelta(days=1)
                # Add 3 weeks to get 4th Thursday
                thanksgiving += pd.Timedelta(weeks=3)
                
                data.loc[data.index.date == thanksgiving.date(), 'is_holiday'] = 1
            
            # Holiday proximity features
            data['days_to_holiday'] = 0
            data['days_from_holiday'] = 0
            
            holiday_dates = data[data['is_holiday'] == 1].index
            
            for idx in data.index:
                if len(holiday_dates) > 0:
                    # Days to next holiday
                    future_holidays = holiday_dates[holiday_dates > idx]
                    if len(future_holidays) > 0:
                        data.loc[idx, 'days_to_holiday'] = (future_holidays[0] - idx).days
                    
                    # Days from last holiday
                    past_holidays = holiday_dates[holiday_dates < idx]
                    if len(past_holidays) > 0:
                        data.loc[idx, 'days_from_holiday'] = (idx - past_holidays[-1]).days
            
            # Holiday week indicator
            data['is_holiday_week'] = 0
            for holiday_date in holiday_dates:
                week_start = holiday_date - pd.Timedelta(days=3)
                week_end = holiday_date + pd.Timedelta(days=3)
                mask = (data.index >= week_start) & (data.index <= week_end)
                data.loc[mask, 'is_holiday_week'] = 1
            
            logger.info("Holiday features added successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error adding holiday features: {str(e)}")
            raise
    
    def add_economic_features(self, data):
        """Add economic indicators (synthetic for demo)"""
        try:
            logger.info("Adding economic features...")
            
            # Simulate economic indicators
            np.random.seed(42)
            
            # Industrial activity index
            base_industrial = 100 + 10 * np.sin(2 * np.pi * data.index.dayofyear / 365)
            data['industrial_activity'] = base_industrial + np.random.normal(0, 5, len(data))
            
            # Electricity price (correlated with demand)
            base_price = 0.12 + 0.02 * np.sin(2 * np.pi * data.index.hour / 24)
            data['electricity_price'] = base_price + np.random.normal(0, 0.01, len(data))
            
            # Population growth factor
            years_from_start = (data.index - data.index.min()).days / 365.25
            data['population_factor'] = 1 + 0.02 * years_from_start
            
            logger.info("Economic features added successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error adding economic features: {str(e)}")
            raise
    
    def prepare_prophet_data(self, data, target_col='demand'):
        """Prepare data for Prophet model"""
        try:
            prophet_data = pd.DataFrame()
            prophet_data['ds'] = data.index
            prophet_data['y'] = data[target_col].values
            
            # Add regressors
            regressor_cols = ['temperature', 'humidity', 'wind_speed', 'is_holiday']
            for col in regressor_cols:
                if col in data.columns:
                    prophet_data[col] = data[col].values
            
            return prophet_data
            
        except Exception as e:
            logger.error(f"Error preparing Prophet data: {str(e)}")
            raise
    
    def prepare_lstm_data(self, data, target_col='demand', sequence_length=24):
        """Prepare data for LSTM model"""
        try:
            # Select features for LSTM
            feature_cols = [
                target_col, 'temperature', 'humidity', 'wind_speed',
                'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday'
            ]
            
            # Use available columns
            available_cols = [col for col in feature_cols if col in data.columns]
            lstm_data = data[available_cols].copy()
            
            # Handle missing values
            lstm_data = lstm_data.fillna(method='ffill').fillna(method='bfill')
            
            # Normalize data
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(lstm_data)
            
            # Create sequences
            X, y = [], []
            target_idx = lstm_data.columns.get_loc(target_col)
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, target_idx])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {str(e)}")
            raise
    
    def create_feature_matrix(self, data, target_col='demand'):
        """Create feature matrix for traditional ML models"""
        try:
            # Select relevant features
            feature_cols = []
            
            # Temporal features
            temporal_cols = [
                'hour', 'day_of_week', 'month', 'quarter',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'is_weekend', 'is_business_hour', 'is_peak_hour',
                'is_winter', 'is_spring', 'is_summer', 'is_autumn'
            ]
            feature_cols.extend([col for col in temporal_cols if col in data.columns])
            
            # Weather features
            weather_cols = [
                'temperature', 'humidity', 'wind_speed', 'temp_squared',
                'cooling_degree_days', 'heating_degree_days', 'heat_index',
                'is_hot', 'is_cold', 'is_humid', 'is_windy'
            ]
            feature_cols.extend([col for col in weather_cols if col in data.columns])
            
            # Lag features
            lag_cols = [col for col in data.columns if 'lag_' in col or 'rolling_' in col or 'ema_' in col]
            feature_cols.extend(lag_cols)
            
            # Holiday features
            holiday_cols = ['is_holiday', 'days_to_holiday', 'days_from_holiday', 'is_holiday_week']
            feature_cols.extend([col for col in holiday_cols if col in data.columns])
            
            # Economic features
            economic_cols = ['industrial_activity', 'electricity_price', 'population_factor']
            feature_cols.extend([col for col in economic_cols if col in data.columns])
            
            # Create feature matrix
            X = data[feature_cols].copy()
            y = data[target_col].copy()
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            y = y.fillna(method='ffill').fillna(method='bfill')
            
            self.feature_columns = feature_cols
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {str(e)}")
            raise
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                return None
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def select_features(self, X, y, method='correlation', k=20):
        """Select top k features using specified method"""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
            
            if method == 'correlation':
                # Correlation-based selection
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                selected_features = correlations.head(k).index.tolist()
                
            elif method == 'f_regression':
                # F-regression based selection
                selector = SelectKBest(score_func=f_regression, k=k)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                
            elif method == 'mutual_info':
                # Mutual information based selection
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
            
            logger.info(f"Selected {len(selected_features)} features using {method}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            raise

