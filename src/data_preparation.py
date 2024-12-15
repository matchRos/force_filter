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

def prepare_augmented_training_data(file_paths, window_size=50):
    """Bereitet Trainings- und Testdaten aus mehreren augmentierten Datensätzen vor."""
    X, y = [], []
    for file_path in file_paths:
        noisy_signal, ground_truth = load_data(file_path)
        X.append(create_sequences(noisy_signal, window_size))
        y.append(create_sequences(ground_truth, window_size))
    
    # Alle Daten kombinieren
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train[..., np.newaxis], X_test[..., np.newaxis], y_train[..., np.newaxis], y_test[..., np.newaxis]

