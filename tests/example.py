from tensorflow.keras.models import load_model
from filter_real_time_data import filter_real_time_data
import numpy as np

def example():
    # Simulierter Datenstrom
    data_stream = np.random.normal(0, 5, 1000)

    # sägezahnartiges Signal
    data_stream += np.linspace(-100, 100, 1000)

    # sinusförmiges Signal
    data_stream += 100 * np.sin(np.linspace(0, 10, 1000))

    model = load_model("autoencoder_model.h5")
    filter_real_time_data(model, data_stream)


if __name__ == "__main__":
    example()