import numpy as np
import pandas as pd

def generate_ground_truth(t):
    """Erstellt das Ground-Truth-Signal."""
    return 50 * np.sin(2 * np.pi * 0.2 * t) + 20 * (t > 5)

def add_noise(signal, noise_level=5):
    """Fügt dem Signal Rauschen hinzu."""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def simulate_signals(save_to_file=True):
    """Simuliert das Ground Truth und verrauschte Signale."""
    t = np.linspace(0, 10, 1000)
    ground_truth = generate_ground_truth(t)
    noisy_signal = add_noise(ground_truth)
    
    if save_to_file:
        data = pd.DataFrame({'time': t, 'ground_truth': ground_truth, 'noisy_signal': noisy_signal})
        data.to_csv("data/raw_signals.csv", index=False)
    return t, ground_truth, noisy_signal
