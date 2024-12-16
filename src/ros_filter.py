
import rospy
from std_msgs.msg import Float32
from tensorflow.keras.models import load_model
import numpy as np

class RealTimeFilterNode:
    def __init__(self):
        self.model = load_model("saved_models/autoencoder_model.h5")
        self.buffer = []
        self.window_size = 50

        # ROS-Subscriber und Publisher
        self.sub = rospy.Subscriber("/noisy_force", Float32, self.callback)
        self.pub = rospy.Publisher("/filtered_force", Float32, queue_size=10)

    def callback(self, msg):
        # Neue Daten in den Puffer hinzufügen
        self.buffer.append(msg.data)

        if len(self.buffer) == self.window_size:
            # Formatiere die Daten
            input_data = np.array(self.buffer)[np.newaxis, ..., np.newaxis]

            # Vorhersage
            filtered_output = self.model.predict(input_data)[0].flatten()

            # Publiziere das gefilterte Signal
            self.pub.publish(filtered_output[-1])  # Nur den letzten Wert senden

            # Entferne das älteste Element
            self.buffer.pop(0)

if __name__ == "__main__":
    rospy.init_node("real_time_filter")
    node = RealTimeFilterNode()
    rospy.spin()
