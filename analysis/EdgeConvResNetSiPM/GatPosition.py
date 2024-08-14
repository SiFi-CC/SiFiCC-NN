# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# %%
import os

# %%
os.chdir("/home/philippe/Master/github/SiFiCC-NN/datasets/OptimisedGeometry_CodedMaskHIT_Spot1_2e10_protons_simv5")

# %%
# Load data from files
def load_data():
    adj_matrix = np.load("A.npy")  # Shape should be (n_graphs, 224, 224)
    node_features = np.load("node_attributes.npy")  # Shape should be (n_graphs, 224, 5)
    graph_attributes = np.load("graph_attributes.npy")[:,:,0]  # Shape should be (n_graphs, 385, 2)
    return adj_matrix, node_features, graph_attributes

# %%
def prepare_dataset(adj_matrix, node_features, graph_attributes):
    def data_generator():
        num_samples = node_features.shape[0]
        for i in range(num_samples):
            # Create a single adjacency matrix for the current sample
            single_adj_matrix = np.copy(adj_matrix)
            # Yield the tuple of (node_features, adj_matrices) and graph_attributes
            yield (node_features[i], single_adj_matrix), graph_attributes[i]
    
    # Create the dataset using the generator
    return tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            (
                tf.TensorSpec(shape=node_features.shape[1:], dtype=tf.float32),
                tf.TensorSpec(shape=(224, 224), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(graph_attributes.shape[1]), dtype=tf.float32)
        )
    )

# %%
class GATLayer(layers.Layer):
    def __init__(self, num_heads, num_out_features, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_out_features = num_out_features

    def build(self, input_shape):
        num_nodes = input_shape[0][1]
        self.attn_weights = self.add_weight(shape=(input_shape[0][-1], self.num_out_features),
                                            initializer='glorot_uniform', name='attn_weights')
        self.attn_bias = self.add_weight(shape=(self.num_out_features,), initializer='zeros', name='attn_bias')
        super(GATLayer, self).build(input_shape)

    def call(self, inputs):
        features, adj = inputs
        features_transformed = tf.matmul(features, self.attn_weights)
        attn_scores = tf.matmul(features_transformed, features_transformed, transpose_b=True)
        attn_scores += self.attn_bias
        attn_scores = tf.nn.softmax(attn_scores)
        aggregated_features = tf.matmul(attn_scores, features_transformed)
        return aggregated_features

# %%
def build_gat_model(num_node_features, num_out_features, num_graph_attributes):
    inputs = {
        'features': layers.Input(shape=(None, num_node_features)),
        'adj': layers.Input(shape=(None, None))
    }

    x = GATLayer(num_heads=1, num_out_features=224)([inputs['features'], inputs['adj']])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(num_out_features, activation='relu')(x)
    
    # Output layer for reconstructing graph attributes
    outputs = layers.Dense(num_graph_attributes, activation='linear')(x)

    model = models.Model(inputs=[inputs['features'], inputs['adj']], outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# %%
# Load data
adj_matrix, node_features, graph_attributes = load_data()

train_features, val_features, train_labels, val_labels = train_test_split(
    node_features, graph_attributes, test_size=0.2, random_state=42
)


# %%
# Prepare dataset
num_graph_attributes = graph_attributes.shape[1]  # Flattened shape
#dataset = prepare_dataset(adj_matrices, node_features, graph_attributes)

train_dataset = prepare_dataset( adj_matrix, train_features, train_labels)
val_dataset = prepare_dataset(adj_matrix, val_features,  val_labels)




# %%
# Build model
num_node_features = node_features.shape[-1]  # Update as needed
num_out_features = 10
model = build_gat_model(num_node_features, num_out_features, train_labels.shape[1])


# %%

# Train model
batch_size = 64
epochs = 3
history = model.fit(train_dataset.batch(batch_size), 
                    epochs=epochs,
                    validation_data=val_dataset.batch(batch_size))

# %%

import matplotlib.pyplot as plt

# Plotting the training and validation loss
loss = history.history['loss']
val_loss = history.history.get('val_loss', [])

plt.figure(figsize=(10, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss.png")

# %%

