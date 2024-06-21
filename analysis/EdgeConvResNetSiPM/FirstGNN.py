import tensorflow as tf
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore
from spektral.layers import GCNConv
import matplotlib.pyplot as plt
import numpy as np
from spektral.data import Dataset, Graph
import os

from SIFICCNN.utils import parent_directory

class DSCodedMask(Dataset):

    def __init__(self, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    @property
    def path(self):
        # Assuming you have a function parent_directory() defined elsewhere
        path = parent_directory()
        path = os.path.join(path, "datasets", "CodedMask", self.name)
        return path

    @property
    def sipm_id_to_position(self):
        sipm_id = self.sipm_ids
        outside_check = np.greater(sipm_id, 224)
        if np.any(outside_check):
            raise ValueError("SiPMID outside detector found! ID: {} ".format(sipm_id))
        y = sipm_id // 112
        sipm_id -= (y * 112)
        x = sipm_id // 28
        z = (sipm_id % 28)
        return np.array([(int(x_i), int(y_i), int(z_i)) for (x_i, y_i, z_i) in zip(x, y, z)])

    def download(self):
        print("Missing download method!")  # Implement if necessary

    def read(self):
        # Assuming you have these files in your dataset directory
        node_attributes = np.load(os.path.join(self.path, "node_attributes.npy"))
        graph_indicator = np.load(os.path.join(self.path, "graph_indicator.npy"))
        graph_labels = np.load(os.path.join(self.path, "graph_labels.npy"))

        n_nodes = np.bincount(graph_indicator)
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))

        sipm_ids = np.load(os.path.join(self.path, "sipm_ids.npy"))

        x_list = np.split(node_attributes, n_nodes_cum[1:])
        y_list = graph_labels

        # Assuming you also need to convert SiPM IDs to positions
        positions = self.sipm_id_to_position()

        # Creating graph objects
        graphs = []
        for x, label in zip(x_list, y_list):
            # You might need to modify this part based on your exact requirements
            graphs.append({
                'x': x,
                'position': positions,  # Assuming position is a required attribute
                'label': label
            })

        return graphs

    def get_classweight_dict(self):
        labels = np.load(os.path.join(self.path, "graph_labels.npy"))
        _, counts = np.unique(labels, return_counts=True)
        class_weights = {0: len(labels) / (2 * counts[0]),
                         1: len(labels) / (2 * counts[1])}
        return class_weights

    @property
    def sp(self):
        return np.load(os.path.join(self.path, "graph_sp.npy"))

    @property
    def pe(self):
        return np.load(os.path.join(self.path, "graph_pe.npy"))

    @property
    def sipm_ids(self):
        return np.load(os.path.join(self.path, "sipm_ids.npy"))

    @property
    def labels(self):
        return np.load(os.path.join(self.path, "graph_labels.npy"))