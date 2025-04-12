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

class EnhancedCosmologyMLPipeline:
    def __init__(self, input_shape=(1000, 1), robust_scaling=True):
        """
        Inicializa o pipeline de ML para cosmologia
        
        Parâmetros:
        - input_shape: Formato dos dados de entrada
        - robust_scaling: Se True, usa RobustScaler (menos sensível a outliers)
        """
        self.scaler = RobustScaler() if robust_scaling else StandardScaler()
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self._setup_logging()
        
    def _setup_logging(self):
        """Configura logging detalhado"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CosmologyMLPipeline')
        
    def _build_model(self, learning_rate=0.001, dropout_rate=0.2, l2_lambda=0.01):
        """
        Constrói modelo CNN 1D robusto com:
        - Normalização de batch
        - Dropout
        - Regularização L2
        - Pooling
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
        
        return model
    
    def _get_callbacks(self, checkpoint_path='best_model.h5', patience=10):
        """Retorna callbacks para treinamento robusto"""
        return [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2),
            tf.keras.callbacks.TerminateOnNaN()
        ]
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """
        Pré-processa os dados com:
        - Validação de input
        - Escalonamento robusto
        - Divisão treino/validação
        """
        try:
            X = np.asarray(X)
            y = np.asarray(y)
            
            if X.shape[0] != y.shape[0]:
                raise ValueError("X e y devem ter o mesmo número de amostras")
                
            X_scaled = self.scaler.fit_transform(X)
            return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
            
        except Exception as e:
            self.logger.error(f"Erro no pré-processamento: {str(e)}")
            raise
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
             epochs=100, batch_size=32, k_folds=None):
        """
        Treina o modelo com opção para validação cruzada
        
        Parâmetros:
        - k_folds: Se None, treina normal, senão faz K-Fold CV
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
                
            self.model = self._build_model()
            callbacks = self._get_callbacks()
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Treinamento concluído com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro durante o treinamento: {str(e)}")
            raise
    
    def _kfold_train(self, X, y, n_splits=5, epochs=100, batch_size=32):
        """Executa K-Fold Cross Validation"""
        kf = KFold(n_splits=n_splits)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            self.logger.info(f"Treinando fold {fold + 1}/{n_splits}")
            
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
        
        self.logger.info(f"Scores médios - MSE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    def evaluate(self, X_test, y_test):
        """Avaliação detalhada do modelo"""
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
            
            self.logger.info(f"Resultados da avaliação - MSE: {results['mse']:.4f}, MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Erro na avaliação: {str(e)}")
            raise
    
    def predict(self, X):
        """Predição com verificação de input"""
        try:
            X_scaled = self.scaler.transform(np.asarray(X))
            X_scaled = X_scaled.reshape(-1, *self.input_shape)
            return self.model.predict(X_scaled).flatten()
            
        except Exception as e:
            self.logger.error(f"Erro na predição: {str(e)}")
            raise
    
    def save_model(self, path='cosmology_model.h5'):
        """Salva o modelo e o scaler"""
        self.model.save(path)
        np.savez(f'{path}_scaler.npz', 
                mean=self.scaler.center_, 
                scale=self.scaler.scale_)
    
    @classmethod
    def load_model(cls, path='cosmology_model.h5', robust_scaling=True):
        """Carrega o modelo e o scaler"""
        pipeline = cls(robust_scaling=robust_scaling)
        pipeline.model = tf.keras.models.load_model(path)
        scaler_data = np.load(f'{path}_scaler.npz')
        pipeline.scaler.center_ = scaler_data['mean']
        pipeline.scaler.scale_ = scaler_data['scale']
        return pipeline