import os
import numpy as np

# Define the directory containing the files
directory = "/home/home2/institut_3b/clement/Master/github/SiFiCC-NN/datasets/CMSimGraphSiPM/Photons/OptimisedGeometry_CodedMaskHIT_Spot1_1e10_protons_MK"

# Get all files in the directory
all_files = os.listdir(directory)

# Filter files for 'labels.npy' and 'graph_attributes.npy'
labels_files = sorted([f for f in all_files if f.endswith("_labels.npy")])
graph_attributes_files = sorted(
    [f for f in all_files if f.endswith("_graph_attributes.npy")])

# Check if corresponding labels and graph_attributes files have the same
# shape[0]


def verify_shapes(labels_files, graph_attributes_files, directory):
    for labels_file, graph_attributes_file in zip(
            labels_files, graph_attributes_files):
        labels_path = os.path.join(directory, labels_file)
        graph_attributes_path = os.path.join(directory, graph_attributes_file)

        # Load the arrays
        labels_array = np.load(labels_path)
        graph_attributes_array = np.load(graph_attributes_path)

        # Check if the first dimension matches
        if labels_array.shape[0] != graph_attributes_array.shape[0]:
            print(
                f"Mismatch in shapes for {labels_file} and {graph_attributes_file}:")
            print(f"  {labels_file} shape[0]: {labels_array.shape[0]}")
            print(
                f"  {graph_attributes_file} shape[0]: {graph_attributes_array.shape[0]}")
        else:
            print(
                f"Shapes match for {labels_file} and {graph_attributes_file}")


# Run the verification
verify_shapes(labels_files, graph_attributes_files, directory)
