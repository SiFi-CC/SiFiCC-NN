import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.models import Model
from spektral.layers import GCNConv
import numpy as np

def build_graph_network(input_dim, hidden_dim, output_dim):
    # Define input layers for node features and adjacency matrix
    X_input = Input(shape=(224, input_dim))  # Adjusted input shape
    A_input = Input(shape=(224,224))  # Adjusted input shape for dense adjacency matrix

    # Graph convolutional layers
    X = GCNConv(hidden_dim, activation='relu', sparse=False)([X_input, A_input])
    X = GCNConv(hidden_dim, activation='relu', sparse=False)([X, A_input])
    
    # Flatten the output for Dense layer
    X_flat = Flatten()(X)
    
    # Dense layer with output shape (55*7*2)
    X_dense = Dense(units=np.prod(output_dim), activation=None)(X_flat)
    
    # Reshape to desired output shape (55, 7, 2)
    X_output = Reshape(output_dim)(X_dense)

    # Create and compile model
    model = Model(inputs=[X_input, A_input], outputs=X_output)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    return model

# Example dimensions
input_dim = 5  # Number of features per node
output_dim = (55, 7, 2)  # Output shape: 55x7x2 tensor
hidden_dim = 64  # Hidden dimension for GCN layers

# Build and compile the graph network
graph_network = build_graph_network(input_dim, hidden_dim, output_dim)

# Assuming you have your data loaded as X_train and A_train
# Make sure X_train and A_train have shapes compatible with your model

# Example training loop
batch_size = 1000
epochs = 10

# Dummy data for demonstration
X_train = np.random.rand(100000, 224, input_dim)
A_train = np.random.rand(100000, 224, 224)

# Training the model
history = graph_network.fit([X_train, A_train], np.zeros((100000,) + output_dim),
                            batch_size=batch_size,
                            epochs=epochs)
