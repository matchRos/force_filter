import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Lädt die simulierten Daten aus einer CSV-Datei."""
    data = pd.read_csv(file_path)
    return data['noisy_signal'].values, data['ground_truth'].values

def create_sequences(data, window_size):
    """Erstellt Sequenzen für das Training."""
    sequences = []
    for i in range(len(data) - window_size + 1):  # "+1", damit die Länge korrekt ist
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

def prepare_training_data(file_path, window_size=50):
    """Erstellt die Trainings- und Testdaten."""
    noisy_signal, ground_truth = load_data(file_path)
    X = create_sequences(noisy_signal, window_size)
    y = create_sequences(ground_truth, window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train[..., np.newaxis], X_test[..., np.newaxis], y_train[..., np.newaxis], y_test[..., np.newaxis]
