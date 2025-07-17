##########################################################################
#
# This script converts a stored version of a SiFiCC Simulation datasets to a container used for
# easier read access and direct compatibility for Tensorflow training
# The container holds each event in graph structure
#
##########################################################################

import os
import numpy as np

from SIFICCNN.utils import parent_directory

from .parameters import get_parameters

from spektral.data import Dataset, Graph
from spektral.utils import io, sparse
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import logging

logging.basicConfig(level=logging.INFO)

class DSGraphSiPM(Dataset):

    def __init__(self,
                 type,
                 mode,
                 norm_x=None,
                 norm_e=None,
                 positives=False,
                 regression=None,
                 name="SimGraphSiPM",
                 **kwargs):
        """
        Initializes the DSGraphSiPM dataset.

        Args:
            type (str): Continouous, 0mm, ...
            norm_x (array, optional): Normalization parameters for node features.
            norm_e (array, optional): Normalization parameters for edge features.
            positives (bool, optional): If True, only positive samples are used.
            regression (str, optional): Type of regression ('Energy' or 'Position').
            name (str, optional): Name of the dataset.
            **kwargs: Additional arguments for the Dataset class.
        """
        self.type = type
        self.positives = positives
        self.regression = regression

        self.norm_x = norm_x
        self.norm_e = norm_e

        self.name = name
        self.mode = mode
        self.graph_attribute_slice_edge = get_parameters(self.mode)

        super().__init__(**kwargs)

    @property
    def path(self):
        """
        Returns the path to the dataset directory.
        """
        path = parent_directory()
        path = os.path.join(path, "datasets", self.name, self.type)

        return path

    def download(self):
        """
        Download method is needed if Dataset class from Spektral library is inherited. It is
        practically not needed.

        Returns:
            None
        """
        logging.error("Missing download method!")
    
    def _check_edges(self, el_list, n_nodes):
        # Check if shape and number of nodes are correct
        mins = np.zeros(len(el_list), dtype=np.int8)
        maxs = np.zeros(len(el_list), dtype=np.int8)
        with tqdm(total=len(el_list), desc="Checking edge list nodes") as pbar:
            for i, el in enumerate(el_list):
                mins[i] = np.min(el[:, 0])
                maxs[i] = np.max(el[:, 0])
                pbar.update(1)
        number_of_nodes = maxs - mins + 1
        if not np.all(number_of_nodes == n_nodes):
            logging.error("Number of nodes in edge list does not match the number of nodes in the node list")
            idxs = np.where(number_of_nodes != n_nodes)[0]
            logging.info("Number of nodes in edge list:", number_of_nodes[idxs])
            logging.info("Number of nodes in node list:", n_nodes[idxs])
            logging.info("Index of the graphs with wrong number of nodes:", idxs)
            raise ValueError("Number of nodes in edge list does not match the number of nodes in the node list")
        else:
            logging.info("Number of nodes in edge list matches the number of nodes in the node list")

    def read(self):
        """
        Loads datasets from files and generates graph objects.

        Returns:
            list: List of Graph objects.
        """

        # Get number of nodes in each graph
        node_batch_index, n_nodes_cum, n_nodes = self._get_n_nodes()


        # Get edge lists (el_list)
        #el_list = self._get_el_list(node_batch_index, n_nodes_cum)

        # Get node attributes (x_list)
        x_list = self._get_x_list(n_nodes_cum=n_nodes_cum)
        # Get edge attributes (e_list), in this case edge features are disabled
        #e_list = np.array([None] * len(n_nodes))

        # Create sparse adjacency matrices and re-sort edge attributes in
        # lexicographic order
        a_e_list = []
        total_matrices = len(n_nodes)
        logging.info(
            f"Total number of adjacency matrices to be created: {total_matrices}")
        
        # Check if the number of nodes in the edge list matches the number of nodes in the node list
        # Only for debugging purposes
        # self._check_edges(el_list, n_nodes)
                    # for interconnected graphs
        """
        with tqdm(total=total_matrices, desc="Creating adjacency matrices") as pbar:
            
            
            Implementation for non-interconnected graphs
            for el, e, n in zip(el_list, e_list, n_nodes):
                try:
                    a = sparse.edge_index_to_matrix(edge_index=el,
                                                    edge_weight=np.ones(
                                                        el.shape[0]),
                                                    edge_features=e,
                                                    shape=(n, n))
                except:
                    logging.error("Error in creating adjacency matrix")
                    logging.info("Edge index:", el)
                    logging.info("Edge features:", e)
                    logging.info("Shape:", n)
                    a = sparse.edge_index_to_matrix(edge_index=el,
                                                    edge_weight=np.ones(
                                                        el.shape[0]),
                                                    edge_features=e,
                                                    shape=(n, n))

                a_e_list.append(a)
                pbar.update(1)

        a_list = a_e_list"""
        # If edge features are used, use this: a_list, e_list =
        # list(zip(*a_e_list))

        # Set dataset targets (classification / regression)
        if self.mode == "CMbeamtime":
            y_list = [None] * len(x_list)
            labels = [None] * len(x_list)
        else:
            y_list = self._get_y_list()
            labels = np.load(self.path + "/" + "graph_labels.npy")


        # At this point the full dataset is loaded and filtered according to the settings
        # Limited to True positives only if needed
        logging.info("Successfully loaded {}.".format(self.type))
        logging.info(f"Mode: {self.mode}, Regression: {self.regression}")
        if self.regression is None:
            return [Graph(x=x, a=np.ones(shape=(x.shape[0], x.shape[0])), y=y) for x, y in tqdm(
                zip(x_list, labels), desc="Creating graphs for classification", total=len(x_list))]
        else:
            if self.positives:
                return [
                    Graph(
                        x=x,
                        a=np.ones(shape=(x.shape[0], x.shape[0])),
                        y=y) 
                        for x, y, label in tqdm(
                        zip(
                            x_list,
                            y_list,
                            labels),
                        desc="Creating graphs for regression (positives only)", total=np.sum(labels)) if label]
            else:
                return [Graph(x=x, a=np.ones(shape=(x.shape[0], x.shape[0])), y=y) for x, y in tqdm(
                    zip(x_list, y_list), desc="Creating graphs for regression", total=len(x_list))]
            
    def _get_n_nodes(self):
        # Load the batch index for nodes
        node_batch_index = np.load(self.path + "/" + "graph_indicator.npy")
        # Count the number of nodes in each graph
        n_nodes = np.bincount(node_batch_index)
        # Cumulative sum of nodes to determine the starting index of each graph
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))
        logging.info("Graph indicator loaded succesfully.")
        return node_batch_index, n_nodes_cum, n_nodes
            
    def _get_el_list(self, node_batch_index, n_nodes_cum):
        # Load the edge list
        edges = np.load(self.path + "/" + "A.npy")

        # Split edges into separate edge lists for each graph
        edge_batch_idx = node_batch_index[edges[:, 0]]
        n_edges = np.bincount(edge_batch_idx)
        n_edges_cum = np.cumsum(n_edges[:-1])
        el_list = np.split(
            edges - n_nodes_cum[edge_batch_idx, None], n_edges_cum)
        logging.info("Edges loaded successfully.")
        return el_list

    def _get_x_list(self, n_nodes_cum):
        """
        Grabs node features from files.

        Args:
            n_nodes_cum (array): Cumulative sum of nodes to determine the starting index of each graph.

        Returns:
            list: List of node features for each graph.
        """
        # Load node features
        x_attr = np.load(self.path + "/" + "node_attributes.npy")
        # Normalize node features if normalization parameters are not provided
        if self.norm_x is None:
            self.norm_x = self._get_standardization(x_attr)
        self._standardize(x_attr, self.norm_x)
        # Split node features into separate lists for each graph
        x_list = np.split(x_attr, n_nodes_cum[1:])
        logging.info("Node attributes loaded successfully.")

        return x_list

    def _get_e_list(self, n_edges_cum, n_nodes):
        """
        Grabs edge features from files.

        Args:
            n_edges_cum (array): Cumulative sum of edges to determine the starting index of each graph.
            n_nodes (array): Number of nodes in each graph.

        Returns:
            list: List of edge features for each graph.
        """
        # Load edge features
        e_attr = np.load(self.path + "/" + "edge_attributes.npy")
        # Normalize edge features if normalization parameters are not provided
        if self.norm_e is None:
            self.norm_e = self._get_standardization(e_attr)
        self._standardize(e_attr, self.norm_e)
        # Split edge features into separate lists for each graph
        e_list = np.split(e_attr, n_edges_cum)

        return e_list

    def _get_y_list(self):
        """
        Grabs targets from files. Type of targets are set during the initialization of the dataset.

        Returns:
            array: Array of targets.
        """
        logging.info("Loading targets...")
        if self.regression is not None and self.mode != "CMbeamtime":
            # Load graph attributes for regression tasks
            graph_attributes = np.load(
                self.path + "/" + "graph_attributes.npy")
            logging.info("Graph attributes loaded successfully.")
            if self.mode == "CC-4to1":
                if self.regression == "Energy":
                    y_list = graph_attributes[:, :self.graph_attribute_slice_edge]
                elif self.regression == "Position":
                    y_list = graph_attributes[:, self.graph_attribute_slice_edge:]
                else:
                    logging.error("Regression type not set correctly")
                    return None
            else:
                if self.regression == "Energy":
                    y_list = graph_attributes[:, :self.graph_attribute_slice_edge]
                elif self.regression == "PositionY":
                    y_list = graph_attributes[:, self.graph_attribute_slice_edge+1] # Regression for y-coordinate
                elif self.regression == "PositionXZ":
                    flat_y_list = graph_attributes[:, -1]
                    y_list = np.eye(385, dtype=np.bool_)[flat_y_list.astype(int)]
                    logging.info(f"Shape of y_list: {y_list.shape}")
                else:
                    raise ValueError(f"Unknown regression type: {self.regression}")
        else:
            # Return class labels for classification tasks
            y_list = np.load(self.path + "/" + "graph_labels.npy")
        return y_list

    def get_classweight_dict(self):
        """
        Computes class weights for imbalanced datasets.

        Returns:
            dict: Dictionary with class weights.
        """
        if self.regression is None and self.positives==False:
            labels = np.load(self.path + "/" + "graph_labels.npy")

            _, counts = np.unique(labels, return_counts=True)
            class_weights = {0: len(labels) / (2 * counts[0]),
                             1: len(labels) / (2 * counts[1])}
        else:
            # Multi-class classification case
            graph_attributes = np.load(self.path + "/" + "graph_attributes.npy")
            classes = graph_attributes[:, -1]
            unique_classes = np.unique(classes)
            
            # Compute class weights using sklearn's method
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=classes)
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            
        return class_weight_dict

    @staticmethod
    def _get_standardization(x):
        """
        Returns array of mean and std of features along the -1 axis.

        Args:
            x (numpy array): Feature matrix.

        Returns:
            array: Array of mean and std for each feature.
        """
        ary_norm = np.zeros(shape=(x.shape[1], 2))
        ary_norm[:, 0] = np.mean(x, axis=0)
        ary_norm[:, 1] = np.std(x, axis=0)

        return ary_norm

    @staticmethod
    def _standardize(x, ary_norm):
        """
        Standardizes the features using the provided normalization parameters.

        Args:
            x (numpy array): Feature matrix.
            ary_norm (array): Array of mean and std for each feature.
        """
        for i in range(x.shape[1]):
            x[:, i] -= ary_norm[i, 0]
            x[:, i] /= ary_norm[i, 1]

    @property
    def sp(self):
        """
        Loads and returns the shortest path matrix.

        Returns:
            array: Shortest path matrix.
        """
        sp = np.load(self.path + "/" + "graph_sp.npy")
        return sp

    @property
    def pe(self):
        """
        Loads and returns the potential energy matrix.

        Returns:
            array: Potential energy matrix.
        """
        pe = np.load(self.path + "/" + "graph_pe.npy")
        return pe

    @property
    def labels(self):
        """
        Loads and returns the graph labels.

        Returns:
            array: Graph labels.
        """
        labels = np.load(self.path + "/" + "graph_labels.npy")
        return labels
