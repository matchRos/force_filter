from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add, Dense, Flatten, Input
from tensorflow.keras.models import Model

def resnet_block_1d(input_tensor, filters, kernel_size=3, strides=1):
    """Residual Block für 1D-Daten."""
    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet101_autoencoder_1d(window_size):
    """ResNet101 Autoencoder für 1D-Daten."""
    input_tensor = Input(shape=(window_size, 1))
    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual Blöcke (vereinfachtes ResNet101)
    for _ in range(3):
        x = resnet_block_1d(x, 64)
    for _ in range(4):
        x = resnet_block_1d(x, 128)
    for _ in range(23):  # Tiefere ResNet-Schichten
        x = resnet_block_1d(x, 256)
    for _ in range(3):
        x = resnet_block_1d(x, 512)
    
    x = Flatten()(x)
    x = Dense(window_size, activation='linear')(x)  # Rekonstruktion
    
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=16):
    """Trainiert den Autoencoder."""
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return history