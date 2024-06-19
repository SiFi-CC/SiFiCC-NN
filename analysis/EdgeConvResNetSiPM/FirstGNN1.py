import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from spektral.layers import GCNConv
import matplotlib.pyplot as plt
import numpy as np
from spektral.data import Dataset, Graph
import os

def build_graph_network(input_dim, hidden_dim, output_dim, A_input):
    # Define input layer for node features
    X_input = Input(shape=(input_dim,))

    # Graph convolutional layers
    X = GCNConv(hidden_dim, activation='relu')([X_input, A_input])
    X = GCNConv(hidden_dim, activation='relu')([X, A_input])

    # Output layer
    X_output = Dense(output_dim, activation='softmax')(X)

    # Create and compile model
    model = Model(inputs=[X_input, A_input], outputs=X_output)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

# Define your custom Dataset class to load and preprocess the data
class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.A = np.load(os.path.join(self.data_path, "A.npy"))
        print(self.A.shape)
        self.X = np.load(os.path.join(self.data_path, "node_attributes.npy"))
        self.Y = np.load(os.path.join(self.data_path, "fibre_targets.npy"))
        super().__init__()

    def read(self):
        graphs = [Graph(x=self.X[i], a=self.A[i]) for i in range(len(self.X))]
        return graphs, self.Y



# Load your dataset
dataset = CustomDataset('/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/OptimisedGeometry_CodedMaskHIT_Spot1_2e10_protons_simv5')

# Define input dimensions (number of features per node)
input_dim = 5  # Assuming x, y, z, timestamp, photon count for SiPMs
output_dim = 4  # Assuming x, z, y, energy for fibers

# Load the adjacency matrix for the first graph (assuming all graphs have the same shape)
A_input = Input(shape=(dataset.A.shape[1],), sparse=True)

# Build and compile the graph network
graph_network = build_graph_network(input_dim, hidden_dim=64, output_dim=output_dim, A_input=A_input)

# Train the model
history = graph_network.fit_generator(dataset, epochs=10, batch_size=32)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Training History')
plt.legend()
plt.show()
