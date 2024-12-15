import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluiert das Modell und visualisiert die Ergebnisse."""
    predictions = model.predict(X_test)

    # Beispielplot für eine Sequenz
    plt.plot(y_test[0].flatten(), label="Ground Truth")
    plt.plot(predictions[0].flatten(), label="Predicted", linestyle='dashed')
    plt.legend()
    plt.show()

    return predictions


def plot_training_history(history):
    """Visualisiert den Verlust während des Trainings."""
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




def evaluate_model_predictions(y_true, y_pred):
    """Berechnet MAE und RMSE für die Modellvorhersagen."""
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    return mae, rmse


def plot_reconstructed_signal(y_true, y_pred, index=0):
    """Plottet Ground Truth und Vorhersagen für eine Sequenz."""
    plt.plot(y_true[index].flatten(), label="Ground Truth")
    plt.plot(y_pred[index].flatten(), label="Predicted", linestyle='dashed')
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Signal")
    plt.show()


def compute_snr(signal, noise):
    """Berechnet das Signal-zu-Rausch-Verhältnis (SNR) in dB."""
    power_signal = np.mean(np.square(signal))
    power_noise = np.mean(np.square(noise))
    return 10 * np.log10(power_signal / power_noise)
