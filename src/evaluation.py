import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Evaluiert das Modell und visualisiert die Ergebnisse."""
    predictions = model.predict(X_test)

    # Beispielplot für eine Sequenz
    plt.plot(y_test[0].flatten(), label="Ground Truth")
    plt.plot(predictions[0].flatten(), label="Predicted", linestyle='dashed')
    plt.legend()
    plt.show()

    return predictions
