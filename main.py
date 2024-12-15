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

    # 4. Modell trainieren
    train_autoencoder(model, X_train, y_train, X_test, y_test)

    # 5. Modell evaluieren
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
