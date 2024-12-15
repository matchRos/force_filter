import numpy as np
import pandas as pd

def generate_ground_truth(t):
    """Erstellt das Ground-Truth-Signal."""
    return 50 * np.sin(2 * np.pi * 0.2 * t) + 20 * (t > 5)

def add_noise(signal, noise_level=5):
    """Fügt dem Signal Rauschen hinzu."""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def augment_signal(signal, noise_levels=[5, 10, 15]):
    """Erzeugt mehrere Varianten eines Signals mit unterschiedlichen Rauschpegeln."""
    augmented_signals = []
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, len(signal))
        augmented_signals.append(signal + noise)
    return np.array(augmented_signals)

def simulate_augmented_signals(save_to_file=True):
    """Simuliert augmentierte Signale mit Ground Truth."""
    t = np.linspace(0, 10, 1000)
    ground_truth = generate_ground_truth(t)

    # Erstelle augmentierte Signale
    augmented_signals = augment_signal(ground_truth)

    if save_to_file:
        for i, signal in enumerate(augmented_signals):
            pd.DataFrame({'time': t, 'ground_truth': ground_truth, f'noisy_signal_{i}': signal}).to_csv(f"data/raw_signals_{i}.csv", index=False)
    return t, ground_truth, augmented_signals
