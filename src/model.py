from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, UpSampling1D

def build_autoencoder(window_size):
    """Definiert den Autoencoder."""
    model = Sequential([
        # Encoder
        Conv1D(16, kernel_size=3, activation='relu', padding='same', input_shape=(window_size, 1)),
        MaxPooling1D(pool_size=2, padding='same'),
        Conv1D(8, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        
        # Decoder
        Conv1D(8, kernel_size=3, activation='relu', padding='same'),
        UpSampling1D(size=2),
        Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        UpSampling1D(size=2),
        Conv1D(1, kernel_size=3, activation='linear', padding='same')  # Ausgabe: gleiche Länge wie Eingabe
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """Trainiert den Autoencoder."""
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return history
