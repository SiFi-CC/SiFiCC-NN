import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from spektral.data import DisjointLoader, Dataset, Graph
from spektral.datasets import OGB
from spektral.layers import ECCConv, GlobalSumPool, XENetConv

################################################################################
# Config
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10  # Number of training epochs
batch_size = 32  # Batch size
num_graphs = 100000  # Total number of graphs
max_nodes = 15  # Maximum number of nodes per graph
min_nodes = 5   # Minimum number of nodes per graph
num_node_features = 2
num_edge_features = 2

################################################################################
# Create Synthetic Dataset
################################################################################
class SyntheticGraphDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        graphs = []

        for _ in range(num_graphs):
            # Random number of nodes
            num_nodes = np.random.randint(min_nodes, max_nodes + 1)

            # Random node features
            x = np.random.rand(num_nodes, num_node_features).astype(np.float32)

            # Generate a random adjacency matrix (symmetric)
            adj = np.random.randint(0, 2, size=(num_nodes, num_nodes))
            np.fill_diagonal(adj, 0)  # No self-loops

            # Random edge features
            edge_indices = np.argwhere(adj)
            num_edges = edge_indices.shape[0]
            edge_features = np.random.rand(num_edges, num_edge_features).astype(np.float32)

            # Graph label (binary classification)
            y = np.random.randint(0, 2, size=(1,)).astype(np.float32)
            "Should be continuous edge labels: (num_edges, 2)"

            # Create a graph and add it to the list
            graphs.append(Graph(x=x, a=adj, e=edge_features, y=y))

        return graphs

# Instantiate the dataset
dataset = SyntheticGraphDataset()

# Train/test split indices (example, modify as needed)
idx_tr = np.random.choice(range(num_graphs), size=int(0.8 * num_graphs), replace=False)
idx_va = np.random.choice(list(set(range(num_graphs)) - set(idx_tr)), size=int(0.1 * num_graphs), replace=False)
idx_te = list(set(range(num_graphs)) - set(idx_tr) - set(idx_va))

# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Create train, validation, and test datasets
dataset_tr = dataset[idx_tr]
dataset_va = dataset[idx_va]
dataset_te = dataset[idx_te]

# Create loaders
loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(dataset_va, batch_size=batch_size, epochs=1)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)



################################################################################
# Build model
################################################################################
X_in = Input(shape=(F,))
A_in = Input(shape=(None,), sparse=True)
E_in = Input(shape=(S,))
I_in = Input(shape=(), dtype=tf.int64)


X_1, E1 = XENetConv(stack_channels=32, edge_channels=32, node_channels=32, activation="relu")([X_in, A_in, E_in])
X_2, E2 = XENetConv(stack_channels=32, edge_channels=32, node_channels=32, activation="relu")([X_1, A_in, E1])
E3 = GlobalSumPool()([E2])
output = Dense(n_out, activation="sigmoid")(E3)

model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
optimizer = Adam(learning_rate)
loss_fn = BinaryCrossentropy()


################################################################################
# Fit model
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


step = loss = 0
for batch in loader_tr:
    step += 1
    loss += train_step(*batch)
    if step == loader_tr.steps_per_epoch:
        step = 0
        print("Loss: {}".format(loss / loader_tr.steps_per_epoch))
        loss = 0

