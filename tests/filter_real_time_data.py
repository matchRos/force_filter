import numpy as np
from matplotlib import pyplot as plt

def filter_real_time_data(model, data_stream, window_size=100):
    """Filtert Daten in Echtzeit."""
    buffer = []

    for idx , data_point in enumerate(data_stream):
        buffer.append(data_point)

        if len(buffer) == window_size:
            # Formatiere die Daten
            input_data = np.array(buffer)[np.newaxis, ..., np.newaxis]  # Hinzufügen von Batch- und Feature-Dimensionen

            # Filtere die Sequenz
            filtered_output = model.predict(input_data)[0].flatten()

            # Verarbeite das gefilterte Signal
            print("Filtered Output:", filtered_output)

            # Entferne das älteste Element
            buffer.pop(0)

            if idx % 20 == 0:

                # Visualisiere das gefilterte Signal
                plt.figure()
                plt.plot(buffer)
                plt.plot(filtered_output)
                plt.title("Filtered Output")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.show()
