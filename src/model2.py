from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Activation

def build_larger_autoencoder(window_size):
    model = Sequential([
        # Encoder
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(window_size, 1)),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        # Bottleneck
        Conv1D(512, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        # Decoder
        UpSampling1D(size=2),
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        UpSampling1D(size=2),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Conv1D(1, kernel_size=3, activation='linear', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=16):
    """Trainiert den Autoencoder."""
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=epochs, 
        batch_size=batch_size
    )
    return history
