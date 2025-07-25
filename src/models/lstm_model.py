
"""
LSTM-based energy demand forecasting model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging

logger = logging.getLogger(__name__)

class LSTMForecaster:
    """LSTM-based forecasting model for energy demand"""
    
    def __init__(self, 
                 sequence_length=24,
                 hidden_units=64,
                 num_layers=2,
                 dropout_rate=0.2,
                 learning_rate=0.001):
        
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.feature_columns = None
        
    def _build_model(self, input_shape):
        """Build the LSTM model architecture"""
        try:
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                self.hidden_units,
                return_sequences=True if self.num_layers > 1 else False,
                input_shape=input_shape
            ))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())
            
            # Additional LSTM layers
            for i in range(1, self.num_layers):
                return_sequences = i < self.num_layers - 1
                model.add(LSTM(
                    self.hidden_units,
                    return_sequences=return_sequences
                ))
                model.add(Dropout(self.dropout_rate))
                model.add(BatchNormalization())
            
            # Dense layers
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            model.add(Dense(1))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise
    
    def _prepare_sequences(self, data, target_col='demand'):
        """Prepare sequences for LSTM training"""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            X, y = [], []
            
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i, data.columns.get_loc(target_col)])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Fit the LSTM model"""
        try:
            logger.info("Fitting LSTM model...")
            
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._build_model(input_shape)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Prepare validation data
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_fitted = True
            logger.info("LSTM model fitted successfully")
            
            return history
            
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions using the fitted model"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_sequence(self, initial_sequence, steps_ahead):
        """Predict multiple steps ahead using recursive prediction"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            predictions = []
            current_sequence = initial_sequence.copy()
            
            for _ in range(steps_ahead):
                # Predict next step
                next_pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape))
                predictions.append(next_pred[0, 0])
                
                # Update sequence (remove first element, add prediction)
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, 0] = next_pred[0, 0]  # Assuming demand is first feature
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting sequence: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before evaluation")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - predictions.flatten()) ** 2)
            mae = np.mean(np.abs(y_test - predictions.flatten()))
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (avoiding division by zero)
            mask = y_test != 0
            mape = np.mean(np.abs((y_test[mask] - predictions.flatten()[mask]) / y_test[mask])) * 100
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            }
            
            logger.info(f"LSTM Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save(self, filepath):
        """Save the fitted model"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before saving")
            
            # Save Keras model
            model_path = filepath.replace('.pkl', '.h5')
            self.model.save(model_path)
            
            # Save other components
            model_data = {
                'sequence_length': self.sequence_length,
                'hidden_units': self.hidden_units,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'feature_columns': self.feature_columns,
                'model_path': model_path
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
            
            # Load parameters
            self.sequence_length = model_data['sequence_length']
            self.hidden_units = model_data['hidden_units']
            self.num_layers = model_data['num_layers']
            self.dropout_rate = model_data['dropout_rate']
            self.learning_rate = model_data['learning_rate']
            self.scaler = model_data['scaler']
            self.is_fitted = model_data['is_fitted']
            self.feature_columns = model_data['feature_columns']
            
            # Load Keras model
            model_path = model_data['model_path']
            self.model = tf.keras.models.load_model(model_path)
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get model information"""
        info = {
            'model_type': 'LSTM',
            'sequence_length': self.sequence_length,
            'hidden_units': self.hidden_units,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted
        }
        
        if self.model:
            info['total_params'] = self.model.count_params()
            info['trainable_params'] = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        return info
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                ax1.plot(history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot MAE
            ax2.plot(history.history['mae'], label='Training MAE')
            if 'val_mae' in history.history:
                ax2.plot(history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            return None
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, param_grid=None):
        """Tune hyperparameters using grid search"""
        try:
            if param_grid is None:
                param_grid = {
                    'hidden_units': [32, 64, 128],
                    'num_layers': [1, 2, 3],
                    'dropout_rate': [0.1, 0.2, 0.3],
                    'learning_rate': [0.001, 0.01, 0.1]
                }
            
            best_params = None
            best_score = float('inf')
            
            # Simple grid search
            for hidden_units in param_grid.get('hidden_units', [self.hidden_units]):
                for num_layers in param_grid.get('num_layers', [self.num_layers]):
                    for dropout_rate in param_grid.get('dropout_rate', [self.dropout_rate]):
                        for learning_rate in param_grid.get('learning_rate', [self.learning_rate]):
                            
                            try:
                                # Create model with current parameters
                                temp_model = LSTMForecaster(
                                    sequence_length=self.sequence_length,
                                    hidden_units=hidden_units,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate,
                                    learning_rate=learning_rate
                                )
                                
                                # Train model
                                temp_model.fit(X_train, y_train, X_val, y_val, epochs=20)
                                
                                # Evaluate
                                metrics = temp_model.evaluate(X_val, y_val)
                                score = metrics['mse']
                                
                                if score < best_score:
                                    best_score = score
                                    best_params = {
                                        'hidden_units': hidden_units,
                                        'num_layers': num_layers,
                                        'dropout_rate': dropout_rate,
                                        'learning_rate': learning_rate
                                    }
                                
                                logger.info(f"Params: {hidden_units}, {num_layers}, {dropout_rate}, {learning_rate} - MSE: {score:.4f}")
                                
                            except Exception as e:
                                logger.warning(f"Failed to evaluate parameters: {str(e)}")
                                continue
            
            if best_params:
                logger.info(f"Best parameters found: {best_params} with MSE: {best_score:.4f}")
                
                # Update model with best parameters
                self.hidden_units = best_params['hidden_units']
                self.num_layers = best_params['num_layers']
                self.dropout_rate = best_params['dropout_rate']
                self.learning_rate = best_params['learning_rate']
                
                return best_params, best_score
            else:
                logger.warning("No valid parameters found during tuning")
                return None, None
                
        except Exception as e:
            logger.error(f"Error tuning hyperparameters: {str(e)}")
            raise

