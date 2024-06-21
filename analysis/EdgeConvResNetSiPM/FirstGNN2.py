import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, AveragePooling1D, GlobalAveragePooling2D, concatenate, InputLayer
from tensorflow.keras.models import Model
from spektral.layers import GCNConv
from spektral.data import Dataset, Graph
from spektral.utils import io, sparse
import spektral
import numpy as np
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt

#graphically display graphs
import networkx as nx

def spektral_to_networkx(graph: Graph) -> nx.Graph:
    nx_graph = nx.Graph()
    adj_matrix = graph.a
    num_nodes = adj_matrix.shape[0]
    
    # Add nodes with attributes
    for i in range(num_nodes):
        nx_graph.add_node(i, features=graph.x[i])
    
    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0:
                nx_graph.add_edge(i, j)
    
    return nx_graph



""" def build_graph_network(input_dim, hidden_dim, output_dim):
    # Define input layers for node features and adjacency matrix
    X_input = Input(shape=(input_dim))  # Adjusted input shape for node features
    A_input = Input(shape=(None, None), sparse=True)  # Adjusted input shape for dense adjacency matrix

    fl = Flatten()(gc1)

    # Fully connected layers for the output
    dense1 = Dense(64, activation='relu')(fl)
    dense2 = Dense(770, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=[X_input, A_input], outputs=dense2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
 """

def build_dense_network(input_dim, hidden_dim, output_dim):
    # Define the input layer for the node features
    X_input = InputLayer(input_shape=input_dim)
    
    # Flatten the input features
    X_flattened = Flatten()(X_input)

    
    # Fully connected layers
    hidden1 = Dense(hidden_dim, activation='relu')(X_input)
    hidden2 = Dense(hidden_dim, activation='relu')(hidden1)
    
    # Output layer
    output = Dense(np.prod(output_dim), activation='softmax')(hidden2)
    output = Reshape(output_dim)(output)
    
    # Create the model
    model = Model(inputs=X_input, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model


# Define your custom Dataset class to load and preprocess the data
class CustomDataset(Dataset):
    def __init__(self, data_path):

        self.data_path = data_path

        self.node_batch_index = np.load(os.path.join(self.data_path, "graph_indicator.npy"))
        self.n__nodes = np.bincount(self.node_batch_index)
        self.n__nodes_cum = np.concatenate(([0], np.cumsum(self.n__nodes)[:-1]))

        self.Y = np.load(os.path.join(self.data_path, "fibre_tensor.npy"))



        self.A = np.load(os.path.join(self.data_path, "A.npy"))
        self.X = np.load(os.path.join(self.data_path, "node_attributes.npy"))
        self.SiPMIds = np.load(os.path.join(self.data_path, "sipm_positions.npy"))

         # Split edges into separate edge lists
        edge_batch_idx = self.node_batch_index[self.A[:, 0]]
        n_edges = np.bincount(edge_batch_idx)
        n_edges_cum = np.cumsum(n_edges[:-1])
        el_list = np.split(self.A - self.n__nodes_cum[edge_batch_idx, None],
                           n_edges_cum)
        
        # get node attributes (x_list)
        self.x_list, self.id_list = self._get_x_id_list(n_nodes_cum=self.n__nodes_cum)
        # get edge attributes (e_list), in this case edge features are disabled
        e_list = [None] * len(self.n__nodes)



        # At this point the full datasets is loaded and filtered according to the settings
        # limited to True positives only if needed
        print("Successfully loaded {}.".format(self.data_path))

            

        super().__init__()
    
    def read(self):
        return [Graph(x=x, a=self.A, idx=idx, y=self.Y[i]) for i, (x, idx) in enumerate(zip(self.x_list, self.id_list))]


    def _get_x_id_list(self, n_nodes_cum):
        """
        Grabs node features from files.
        """
        # Node features
        x_attr = np.load(self.data_path + "/" + "node_attributes.npy")
        """         if self.norm_x is None:
            self.norm_x = self._get_standardization(x_attr)
        self._standardize(x_attr, self.norm_x) """
        
        x_list = np.split(x_attr, n_nodes_cum[1:])
        id_list = np.split(self.SiPMIds, n_nodes_cum[1:])

        return x_list, id_list

    
    def _get_e_list(self, n_edges_cum):
        """
        Grabs edge features from files.
        """
        e_attr = np.load(self.data_path + "/" + "edge_attributes.npy")  # ["arr_0"]
        if self.norm_e is None:
            self.norm_e = self._get_standardization(e_attr)
        self._standardize(e_attr, self.norm_e)
        e_list = np.split(e_attr, n_edges_cum)
        return e_list


    @staticmethod
    def _get_standardization(x):
        """
        Returns array of mean and std of features along the -1 axis

        Args:
            x (numpy array): feature matrix

        Returns:
            ary_mean, ary_std
        """

        ary_norm = np.zeros(shape=(x.shape[1], 2))
        ary_norm[:, 0] = np.mean(x, axis=0)
        ary_norm[:, 1] = np.std(x, axis=0)

        return ary_norm

    @staticmethod
    def _standardize(x, ary_norm):
        for i in range(x.shape[1]):
            x[:, i] -= ary_norm[i, 0]
            x[:, i] /= ary_norm[i, 1]

    @property
    def sp(self):
        sp = np.load(self.path + "/" + "graph_sp.npy")
        return sp

    @property
    def pe(self):
        pe = np.load(self.path + "/" + "graph_pe.npy")
        return pe

    @property
    def labels(self):
        labels = np.load(self.path + "/" + "graph_labels.npy")  # ["arr_0"]
        return labels


# Load your dataset
dataset = CustomDataset('/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/OptimisedGeometry_CodedMaskHIT_Spot1_2e10_protons_simv5')

# Define input dimensions (number of features per node)
input_dim = (224,5)  # Assuming x, y, z, timestamp, photon count for SiPMs
output_dim = (55,7,2)  # Assuming x, z, y, energy for fibers
hidden_dim = 500  # Hidden dimension for GCN layers

# Build and compile the graph network
graph_network = build_dense_network(input_dim, hidden_dim, output_dim)


def are_connected(SiPM1, SiPM2):
    is_y_neighbor = SiPM1.y != SiPM2.y
    is_x_z_neighbor = abs(SiPM1.x - SiPM2.x) + abs(SiPM1.z - SiPM2.z) <=4
    return is_x_z_neighbor and is_y_neighbor

def dense_to_sparse(A):
    A = tf.convert_to_tensor(A)
    indices = tf.where(tf.not_equal(A, 0))
    values = tf.gather_nd(A, indices)
    shape = tf.shape(A, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)

""" def generator(dataset):
    target_shape = (224, 5)  # Desired shape

    while True:
        for graph in dataset:
            x_raw = graph.x
            idxs = graph.idx

            # Create a tensor of zeros with the desired shape
            padded_tensor = np.zeros(target_shape, dtype=x_raw.dtype)

            # Use idx to place x_raw into the correct positions within padded_tensor
            for i, idx in enumerate(idxs):
                padded_tensor[idx, :] = x_raw[i]

            # Convert to TensorFlow tensor
            padded_tensor_tf = tf.convert_to_tensor(padded_tensor)

            # Convert adjacency matrix to a dense tensor
            a_dense = tf.convert_to_tensor(graph.a, dtype=tf.int8)

            print(40*"=")
            print(graph.y.shape)

            #yield [padded_tensor_tf, dense_to_sparse(a_dense)], graph.y

            #yield [padded_tensor_tf, a_dense], graph.y
            yield padded_tensor_tf, graph.y """

def generator(dataset):
    target_shape = (224, 5)  # Desired shape

    while True:
        for graph in dataset:
            x_raw = graph.x
            idxs = graph.idx

            # Create a tensor of zeros with the desired shape
            padded_tensor = np.zeros(target_shape, dtype=x_raw.dtype)

            # Use idx to place x_raw into the correct positions within padded_tensor
            for i, idx in enumerate(idxs):
                padded_tensor[idx, :] = x_raw[i]

            # Convert to TensorFlow tensor
            padded_tensor_tf = tf.convert_to_tensor(padded_tensor)

            yield padded_tensor_tf, graph.y

# Load your dataset
dataset = CustomDataset('/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/OptimisedGeometry_CodedMaskHIT_Spot1_2e10_protons_simv5')


# Create generator instance
train_generator = generator(dataset.read())

# Debugging prints within training loop
for inputs, targets in train_generator:
    print("Inputs shapes in training loop:")
    print(f"X_batch shape: {inputs[0].shape}")
    #print(f"A_batch shape: {inputs[1].shape}")
    print(f"Target shape: {targets.shape}")
    break  # For demonstration, only print the first batch



""" def draw_graph(graph,i):
    # Convert Spektral graph to NetworkX graph
    nx_graph = spektral_to_networkx(dataset.read()[1])

    # Custom labels for nodes and edges
    #node_labels = {0: 'Node 1', 1: 'Node 2', 2: 'Node 3'}
    #edge_labels = {(0, 1): 'Edge 1-2', (1, 2): 'Edge 2-3'}

    # Draw the graph with custom labels
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
    nx.draw_networkx_edge_labels(nx_graph, pos)
    plt.savefig("gi/"+str(i)+".png")
    plt.clf()
    plt.close()

for i in range(100):
    draw_graph(dataset.read()[i],i) """




# Train the model
history = graph_network.fit(
    generator(dataset.read()),
    steps_per_epoch=len(dataset),
    epochs=10
)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Training History')
plt.legend()
plt.show()

