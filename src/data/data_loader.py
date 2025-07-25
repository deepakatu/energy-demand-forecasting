
"""
Data loading and preprocessing for energy demand forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EnergyDataLoader:
    """Data loader for energy demand forecasting"""
    
    def __init__(self):
        self.data = None
        
    def load_sample_data(self, start_date='2020-01-01', end_date='2023-12-31', freq='H'):
        """Generate sample energy demand data"""
        try:
            logger.info("Generating sample energy demand data...")
            
            # Create datetime index
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Generate synthetic energy demand data
            np.random.seed(42)
            
            # Base demand with seasonal and daily patterns
            base_demand = 1000  # MW
            
            # Seasonal pattern (higher in summer and winter)
            seasonal_pattern = 200 * np.sin(2 * np.pi * date_range.dayofyear / 365.25)
            
            # Daily pattern (higher during day, peak in evening)
            daily_pattern = 300 * (
                0.5 * np.sin(2 * np.pi * (date_range.hour - 6) / 24) +
                0.3 * np.sin(2 * np.pi * (date_range.hour - 18) / 12)
            )
            
            # Weekly pattern (lower on weekends)
            weekly_pattern = -100 * (date_range.weekday >= 5).astype(int)
            
            # Holiday effect (lower demand)
            holiday_effect = np.zeros(len(date_range))
            for date in date_range:
                if (date.month == 12 and date.day == 25) or (date.month == 1 and date.day == 1):
                    holiday_effect[date_range.get_loc(date)] = -200
            
            # Weather correlation (temperature effect)
            temperature = 15 + 10 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365.25)
            temperature += 5 * np.sin(2 * np.pi * date_range.hour / 24)
            temperature += np.random.normal(0, 2, len(date_range))
            
            # Cooling and heating demand
            cooling_demand = np.maximum(temperature - 20, 0) * 20
            heating_demand = np.maximum(15 - temperature, 0) * 15
            
            # Economic growth trend
            years_from_start = (date_range - date_range[0]).days / 365.25
            growth_trend = 50 * years_from_start
            
            # Random noise
            noise = np.random.normal(0, 50, len(date_range))
            
            # Combine all components
            demand = (base_demand + seasonal_pattern + daily_pattern + weekly_pattern + 
                     holiday_effect + cooling_demand + heating_demand + growth_trend + noise)
            
            # Ensure positive demand
            demand = np.maximum(demand, 100)
            
            # Create DataFrame
            data = pd.DataFrame({
                'datetime': date_range,
                'demand': demand,
                'temperature': temperature,
                'humidity': np.random.uniform(30, 90, len(date_range)),
                'wind_speed': np.random.exponential(3, len(date_range))
            })
            
            data = data.set_index('datetime')
            
            # Add some missing values to make it realistic
            missing_indices = np.random.choice(data.index, size=int(0.001 * len(data)), replace=False)
            data.loc[missing_indices, 'demand'] = np.nan
            
            self.data = data
            
            logger.info(f"Generated {len(data)} data points from {start_date} to {end_date}")
            logger.info(f"Demand range: {data['demand'].min():.1f} - {data['demand'].max():.1f} MW")
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            raise
    
    def load_from_csv(self, filepath, datetime_col='datetime', target_col='demand'):
        """Load data from CSV file"""
        try:
            logger.info(f"Loading data from {filepath}...")
            
            data = pd.read_csv(filepath)
            
            # Convert datetime column
            data[datetime_col] = pd.to_datetime(data[datetime_col])
            data = data.set_index(datetime_col)
            
            # Ensure target column exists
            if target_col not in data.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            self.data = data
            
            logger.info(f"Loaded {len(data)} data points")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"Columns: {list(data.columns)}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from CSV: {str(e)}")
            raise
    
    def preprocess_data(self, data=None):
        """Preprocess the loaded data"""
        try:
            if data is None:
                data = self.data
            
            if data is None:
                raise ValueError("No data to preprocess")
            
            logger.info("Preprocessing data...")
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Remove outliers
            data = self._remove_outliers(data)
            
            # Ensure proper frequency
            data = self._ensure_frequency(data)
            
            self.data = data
            
            logger.info("Data preprocessing completed")
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _handle_missing_values(self, data):
        """Handle missing values in the data"""
        try:
            logger.info("Handling missing values...")
            
            # Check for missing values
            missing_counts = data.isnull().sum()
            if missing_counts.sum() > 0:
                logger.info(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
                
                # Forward fill then backward fill
                data = data.fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use interpolation
                data = data.interpolate(method='time')
                
                # Final fallback: use mean
                data = data.fillna(data.mean())
            
            return data
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def _remove_outliers(self, data, target_col='demand', method='iqr', threshold=3):
        """Remove outliers from the data"""
        try:
            logger.info("Removing outliers...")
            
            if target_col not in data.columns:
                return data
            
            original_length = len(data)
            
            if method == 'iqr':
                # Interquartile range method
                Q1 = data[target_col].quantile(0.25)
                Q3 = data[target_col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data[target_col] < lower_bound) | (data[target_col] > upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((data[target_col] - data[target_col].mean()) / data[target_col].std())
                outlier_mask = z_scores > threshold
                
            else:
                raise ValueError(f"Unknown outlier removal method: {method}")
            
            # Replace outliers with interpolated values instead of removing
            data.loc[outlier_mask, target_col] = np.nan
            data[target_col] = data[target_col].interpolate(method='time')
            
            outliers_found = outlier_mask.sum()
            logger.info(f"Handled {outliers_found} outliers ({outliers_found/original_length*100:.2f}%)")
            
            return data
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
    
    def _ensure_frequency(self, data, freq='H'):
        """Ensure data has consistent frequency"""
        try:
            logger.info("Ensuring consistent frequency...")
            
            # Resample to ensure consistent frequency
            data = data.resample(freq).mean()
            
            # Fill any gaps created by resampling
            data = data.interpolate(method='time')
            
            return data
            
        except Exception as e:
            logger.error(f"Error ensuring frequency: {str(e)}")
            raise
    
    def split_data(self, data=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split data into train, validation, and test sets"""
        try:
            if data is None:
                data = self.data
            
            if data is None:
                raise ValueError("No data to split")
            
            # Ensure ratios sum to 1
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError("Train, validation, and test ratios must sum to 1")
            
            n = len(data)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_data = data.iloc[:train_end].copy()
            val_data = data.iloc[train_end:val_end].copy()
            test_data = data.iloc[val_end:].copy()
            
            logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def get_data_info(self, data=None):
        """Get information about the data"""
        try:
            if data is None:
                data = self.data
            
            if data is None:
                return "No data loaded"
            
            info = {
                'shape': data.shape,
                'date_range': f"{data.index.min()} to {data.index.max()}",
                'frequency': pd.infer_freq(data.index),
                'columns': list(data.columns),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict()
            }
            
            if 'demand' in data.columns:
                info['demand_stats'] = {
                    'mean': float(data['demand'].mean()),
                    'std': float(data['demand'].std()),
                    'min': float(data['demand'].min()),
                    'max': float(data['demand'].max()),
                    'median': float(data['demand'].median())
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting data info: {str(e)}")
            return f"Error: {str(e)}"
    
    def save_data(self, filepath, data=None):
        """Save data to CSV file"""
        try:
            if data is None:
                data = self.data
            
            if data is None:
                raise ValueError("No data to save")
            
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def add_external_data(self, external_data, join_key='datetime'):
        """Add external data (weather, economic indicators, etc.)"""
        try:
            if self.data is None:
                raise ValueError("No base data loaded")
            
            logger.info("Adding external data...")
            
            # Ensure external data has datetime index
            if not isinstance(external_data.index, pd.DatetimeIndex):
                if join_key in external_data.columns:
                    external_data[join_key] = pd.to_datetime(external_data[join_key])
                    external_data = external_data.set_index(join_key)
                else:
                    raise ValueError(f"Join key '{join_key}' not found in external data")
            
            # Merge data
            self.data = self.data.join(external_data, how='left')
            
            # Handle missing values from join
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"External data added. New shape: {self.data.shape}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error adding external data: {str(e)}")
            raise

