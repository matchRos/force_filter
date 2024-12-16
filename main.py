from src.signal_simulation import simulate_augmented_signals
from src.data_preparation import prepare_augmented_training_data
from src.model import build_autoencoder, train_autoencoder, build_larger_autoencoder
from src.evaluation import evaluate_model, plot_training_history, evaluate_model_predictions, plot_reconstructed_signal, compute_snr
import tensorflow as tf
from tensorflow.config import threading

def main():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Setze die Anzahl der Threads (maximale Anzahl verfügbarer Kerne)
    threading.set_inter_op_parallelism_threads(4)  # Anzahl der parallelen Threads für die Inter-Op-Operationen
    threading.set_intra_op_parallelism_threads(4)  # Anzahl der parallelen Threads für die Intra-Op-Operationen

    print(f"Inter-Op Threads: {threading.get_inter_op_parallelism_threads()}")
    print(f"Intra-Op Threads: {threading.get_intra_op_parallelism_threads()}")

    # 1. Daten simulieren
    t, ground_truth, augmented_signals = simulate_augmented_signals()

    # Bereite Trainingsdaten aus augmentierten Signalen vor
    window_size = 100
    file_paths = [f"data/raw_signals_{i}.csv" for i in range(len(augmented_signals))]
    X_train, X_test, y_train, y_test = prepare_augmented_training_data(file_paths, window_size)


    # 3. Autoencoder aufbauen
    model = build_larger_autoencoder(window_size)
    print(model.summary())

    # Simuliere eine Eingabe, um die Ausgabeform zu prüfen
    import numpy as np
    example_input = np.random.rand(1, window_size, 1)  # Batchgröße 1, Fenstergröße 50, 1 Feature
    example_output = model.predict(example_input)
    print(f"Eingabeform: {example_input.shape}, Ausgabeform: {example_output.shape}")


    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # 4. Modell trainieren
    history = train_autoencoder(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=16)

    # Visualisiere den Verlauf des Trainings
    plot_training_history(history)

    y_pred = model.predict(X_test)
    evaluate_model_predictions(y_test, y_pred)

    plot_reconstructed_signal(y_test, y_pred, index=0)

    snr_noisy = compute_snr(y_test.flatten(), y_test.flatten() - X_test.flatten())
    snr_filtered = compute_snr(y_test.flatten(), y_test.flatten() - y_pred.flatten())
    print(f"SNR (Noisy): {snr_noisy:.2f} dB")
    print(f"SNR (Filtered): {snr_filtered:.2f} dB")

    # 5. Modell evaluieren
    evaluate_model(model, X_test, y_test)

    # Speichern des Modells
    model.save("saved_models/autoencoder_model.h5")

if __name__ == "__main__":
    main()
