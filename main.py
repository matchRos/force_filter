from src.signal_simulation import simulate_signals
from src.data_preparation import prepare_training_data
from src.model import build_autoencoder, train_autoencoder
from src.evaluation import evaluate_model

def main():
    # 1. Daten simulieren
    t, ground_truth, noisy_signal = simulate_signals()

    # 2. Trainings- und Testdaten vorbereiten
    window_size = 50
    X_train, X_test, y_train, y_test = prepare_training_data("data/raw_signals.csv", window_size)

    # 3. Autoencoder aufbauen
    model = build_autoencoder(window_size)
    print(model.summary())

    # Simuliere eine Eingabe, um die Ausgabeform zu prüfen
    import numpy as np
    example_input = np.random.rand(1, window_size, 1)  # Batchgröße 1, Fenstergröße 50, 1 Feature
    example_output = model.predict(example_input)
    print(f"Eingabeform: {example_input.shape}, Ausgabeform: {example_output.shape}")


    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")


    # 4. Modell trainieren
    train_autoencoder(model, X_train, y_train, X_test, y_test)

    # 5. Modell evaluieren
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
