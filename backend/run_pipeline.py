import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import logging
from typing import Tuple, Optional, Union

class EnhancedCosmologyMLPipeline:
    def __init__(self, input_shape: Tuple[int, int] = (1000, 1), robust_scaling: bool = True):
        """
        Initialize the enhanced cosmology ML pipeline.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            robust_scaling: Whether to use RobustScaler (better for outliers)
        """
        self.scaler = RobustScaler() if robust_scaling else StandardScaler()
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CosmologyMLPipeline')
        
    def _build_model(self, learning_rate: float = 0.001, 
                    dropout_rate: float = 0.2, 
                    l2_lambda: float = 0.01) -> Sequential:
        """
        Build a robust 1D CNN model with:
        - Batch normalization
        - Dropout
        - L2 regularization
        - Pooling layers
        
        Args:
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate
            l2_lambda: L2 regularization factor
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Conv1D(64, 5, activation='relu', 
                  input_shape=self.input_shape,
                  kernel_regularizer=l2(l2_lambda)),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(dropout_rate),
            
            Conv1D(128, 3, activation='relu',
                  kernel_regularizer=l2(l2_lambda)),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(dropout_rate),
            
            Flatten(),
            
            Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, 
                     loss='mse',
                     metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
        
        self.logger.info("Model built successfully")
        return model
    
    def _get_callbacks(self, checkpoint_path: str = 'best_model.h5', 
                      patience: int = 10) -> list:
        """
        Get training callbacks for robust training.
        
        Args:
            checkpoint_path: Path to save best model
            patience: Patience for early stopping
            
        Returns:
            List of configured callbacks
        """
        return [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2),
            tf.keras.callbacks.TerminateOnNaN()
        ]
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                       test_size: float = 0.2, 
                       random_state: int = 42) -> Tuple:
        """
        Preprocess data with validation and scaling.
        
        Args:
            X: Input features
            y: Target values
            test_size: Size of test split
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            X = np.asarray(X)
            y = np.asarray(y)
            
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")
                
            X_scaled = self.scaler.fit_transform(X)
            return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100, 
              batch_size: int = 32, 
              k_folds: Optional[int] = None,
              learning_rate: float = 0.001) -> None:
        """
        Train the model with optional validation and cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            k_folds: Number of folds for cross-validation (None for regular training)
            learning_rate: Learning rate for training
        """
        try:
            X_train = np.asarray(X_train).reshape(-1, *self.input_shape)
            
            if X_val is not None and y_val is not None:
                X_val = np.asarray(X_val).reshape(-1, *self.input_shape)
                validation_data = (X_val, y_val)
            else:
                validation_data = None
            
            if k_folds is not None:
                self._kfold_train(X_train, y_train, k_folds, epochs, batch_size)
                return
                
            self.model = self._build_model(learning_rate=learning_rate)
            callbacks = self._get_callbacks()
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def _kfold_train(self, X: np.ndarray, y: np.ndarray, 
                    n_splits: int = 5, 
                    epochs: int = 100, 
                    batch_size: int = 32) -> None:
        """Perform K-Fold Cross Validation."""
        kf = KFold(n_splits=n_splits)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            self.logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model = self._build_model()
            self.model.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=self._get_callbacks(f'best_model_fold{fold}.h5'),
                         verbose=0)
            
            val_pred = self.model.predict(X_val)
            score = mean_squared_error(y_val, val_pred)
            fold_scores.append(score)
            self.logger.info(f"Fold {fold + 1} - MSE: {score:.4f}")
        
        self.logger.info(f"Average scores - MSE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with metrics and predictions
        """
        try:
            X_test = np.asarray(X_test).reshape(-1, *self.input_shape)
            y_test = np.asarray(y_test)
            
            metrics = self.model.evaluate(X_test, y_test, verbose=0)
            predictions = self.model.predict(X_test)
            
            results = {
                'mse': metrics[0],
                'mae': metrics[1],
                'rmse': metrics[2],
                'predictions': predictions.flatten()
            }
            
            self.logger.info(f"Evaluation results - MSE: {results['mse']:.4f}, "
                            f"MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with input validation.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Model predictions
        """
        try:
            X_scaled = self.scaler.transform(np.asarray(X))
            X_scaled = X_scaled.reshape(-1, *self.input_shape)
            return self.model.predict(X_scaled).flatten()
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def save_model(self, path: str = 'cosmology_model.h5') -> None:
        """Save model and scaler to disk."""
        self.model.save(path)
        np.savez(f'{path}_scaler.npz', 
                center=self.scaler.center_, 
                scale=self.scaler.scale_)
    
    @classmethod
    def load_model(cls, path: str = 'cosmology_model.h5', 
                  robust_scaling: bool = True) -> 'EnhancedCosmologyMLPipeline':
        """Load saved model and scaler."""
        pipeline = cls(robust_scaling=robust_scaling)
        pipeline.model = tf.keras.models.load_model(path)
        scaler_data = np.load(f'{path}_scaler.npz')
        pipeline.scaler.center_ = scaler_data['center']
        pipeline.scaler.scale_ = scaler_data['scale']
        return pipeline