# %%
# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from spektral.data import Dataset, Graph
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt

# %%
# %%
import os


# %%
# %%
os.chdir("/home/philippe/Master/github/SiFiCC-NN/datasets/OptimisedGeometry_CodedMaskHIT_Spot1_2e10_protons_simv5")


# %%
# %%
# Load data from files
def load_data():
    adjacency_list = np.load("A.npy")  
    node_attributes = np.load("node_attributes.npy")  
    node_indicator = np.load("node_indicator.npy")
    edge_attributes = np.load("edge_attributes.npy")  
    edge_indicator = np.load("edge_indicator.npy")
    return adjacency_list, node_attributes, node_indicator, edge_attributes, edge_indicator


# %%

@njit
def create_adjacency_matrix(edges, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(edges.shape[0]):
        src, dst = edges[i]
        adj_matrix[src, dst] = 1
    return adj_matrix


def find_graph_start_indices(indicator_array):
    # Find indices where the graph index changes (i.e., where a new graph begins)
    graph_start_indices = np.where(np.diff(indicator_array, prepend=-1) != 0)[0]
    return graph_start_indices

class MyGraphDataset(Dataset):
    def __init__(self, adjacency_list, node_attributes, node_indicator, edge_attributes, edge_indicator, **kwargs):
        self.adjacency_list = adjacency_list
        self.node_attributes = node_attributes
        self.node_indicator = node_indicator
        self.edge_attributes = edge_attributes
        self.edge_indicator = edge_indicator
        
        # Compute start indices for nodes and edges
        self.node_start_indices = find_graph_start_indices(self.node_indicator)
        self.edge_start_indices = find_graph_start_indices(self.edge_indicator)
        
        self.num_graphs = len(self.node_start_indices)  # Number of graphs

    def read(self):
        graphs = []

        # Iterate over each graph index
        for i in tqdm(range(self.num_graphs), desc="Populating Graphs"):
            # Determine the start and end indices for the current graph
            node_start = self.node_start_indices[i]
            node_end = self.node_start_indices[i + 1] if i + 1 < self.num_graphs else len(self.node_indicator)
            
            edge_start = self.edge_start_indices[i]
            edge_end = self.edge_start_indices[i + 1] if i + 1 < self.num_graphs else len(self.edge_indicator)

            # Extract node and edge attributes for the current graph
            x = self.node_attributes[node_start:node_end]
            e = self.edge_attributes[edge_start:edge_end]

            # Extract edges for the current graph
            edges = self.adjacency_list[edge_start:edge_end]

            # Determine the number of nodes in the current graph
            num_nodes = node_end - node_start

            # Create adjacency matrix using Numba-optimized function
            adj_matrix = create_adjacency_matrix(edges, num_nodes)

            # Create Graph object and append to the list
            graph = Graph(x=x, a=adj_matrix, e=e)
            graphs.append(graph)

        return graphs

# %%

# Load data
adjacency_list, node_attributes, node_indicator, edge_attributes, edge_indicator = load_data()


# %%
# Create dataset
dataset = MyGraphDataset(adjacency_list, node_attributes, node_indicator, edge_attributes, edge_indicator)

graphs = dataset.read()


# %%



