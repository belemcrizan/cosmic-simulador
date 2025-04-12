from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

def create_robust_cnn_model(input_shape=(128, 128, 1), 
                           learning_rate=0.001, 
                           dropout_rate=0.3,
                           l2_lambda=0.001):
    """
    Cria uma CNN robusta com regularização, normalização e callbacks
    
    Parâmetros:
    - input_shape: Tupla com formato da entrada (altura, largura, canais)
    - learning_rate: Taxa de aprendizado para o otimizador
    - dropout_rate: Taxa de dropout para prevenção de overfitting
    - l2_lambda: Fator de regularização L2
    
    Retorna:
    - Modelo Keras compilado e pronto para treino
    """
    
    # Inicialização robusta
    initializer = tf.keras.initializers.HeNormal()
    
    model = Sequential([
        # Primeiro bloco convolucional
        Conv2D(32, (3, 3), 
               activation='relu', 
               kernel_initializer=initializer,
               kernel_regularizer=l2(l2_lambda),
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate/2),
        
        # Segundo bloco convolucional
        Conv2D(64, (3, 3), 
               activation='relu',
               kernel_initializer=initializer,
               kernel_regularizer=l2(l2_lambda)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Terceiro bloco convolucional (adicionado para maior profundidade)
        Conv2D(128, (3, 3), 
               activation='relu',
               kernel_initializer=initializer,
               kernel_regularizer=l2(l2_lambda)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Camadas fully connected
        Flatten(),
        Dense(128, activation='relu',
              kernel_initializer=initializer,
              kernel_regularizer=l2(l2_lambda)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu',
              kernel_initializer=initializer,
              kernel_regularizer=l2(l2_lambda)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Camada de saída
        Dense(1, activation='linear')
    ])
    
    # Otimizador configurável
    optimizer = Adam(learning_rate=learning_rate, 
                    beta_1=0.9, 
                    beta_2=0.999, 
                    epsilon=1e-07)
    
    # Compilação com métricas adicionais
    model.compile(optimizer=optimizer, 
                 loss='mse', 
                 metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def get_model_callbacks(model_path='best_model.h5', patience=10):
    """
    Retorna callbacks úteis para treinamento robusto
    
    Parâmetros:
    - model_path: Caminho para salvar o melhor modelo
    - patience: Paciência para early stopping
    
    Retorna:
    - Lista de callbacks configurados
    """
    return [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience//2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.TerminateOnNaN()
    ]