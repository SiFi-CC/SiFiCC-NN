import logging
import numpy as np
import os
from tqdm import tqdm
from .Event import Event
from .Cluster import Cluster

logging.basicConfig(level=logging.INFO)



class DatasetReader:
    def __init__(self, dataset_path, show_adjacency_matrix=False, mode=None, **kwargs):
        self.name = dataset_path.split("/")[-1]
        self.dataset_path = dataset_path
        self.show_adjacency_matrix = show_adjacency_matrix
        self.mode = mode
        self._set_mode()
        super().__init__(**kwargs)

    @property
    def path(self):
        return self.dataset_path

    def read(self, block_size=100, start_index=0, initial=False):
        """
        Reads and processes event data in blocks. As data is structure for clusters, the events must be reassambled.

        Parameters:
        block_size (int): The number of events to process in each block. Default is 100.
        start_index (int): The starting index of the events to process. Default is 0.
        initial (bool): If True, initializes and loads the dataset. Default is False.

        Yields:
        list: A list of Event objects for each block of events.

        Raises:
        FileNotFoundError: If any of the required files are not found during initialization.

        Notes:
        - When `initial` is True, the method loads necessary data files and initializes internal structures.
        - The method processes events in blocks, yielding a list of Event objects for each block.
        - The `tqdm` library is used to display a progress bar for assembling events.
        """
        if initial:
            # Load dataset only once
            logging.info(f"Loading dataset: {self.name}")
            try:
                self.node_batch_index = np.load(
                    os.path.join(self.path, "graph_indicator.npy")
                )
                self.n_nodes = np.bincount(self.node_batch_index)
                self.n_nodes_cum = np.concatenate(([0], np.cumsum(self.n_nodes)[:-1]))

                self.x_list = self._get_x_list(self.n_nodes_cum)
                self.y_list = self._get_y_list()
                self.labels = np.load(os.path.join(self.path, "graph_labels.npy"))
                self._get_fibre_data()
                self._set_mode()
                logging.info(f"Successfully loaded dataset: {self.name}")
                logging.info("Finished loading block.")
                return self._build_block(block_size, start_index)

            except FileNotFoundError as e:
                logging.error(f"Required file not found: {e}")
                raise e
        else:
            logging.info("Loading block.")
            return self._build_block(block_size, start_index)


    def _get_x_list(self, n_nodes_cum):
        sipm_attributes = np.load(os.path.join(self.path, "node_attributes.npy"))
        return np.split(sipm_attributes, n_nodes_cum[1:])

    def _get_y_list(self):
        return np.load(os.path.join(self.path, "graph_attributes.npy"))

    @property
    def sp(self):
        return np.load(os.path.join(self.path, "graph_sp.npy"))

    @property
    def pe(self):
        return np.load(os.path.join(self.path, "graph_pe.npy"))
    
    def _get_event_indicator(self):
        logging.info("Trying to load event indicator at: {}".format(self.path))
        event_indicator = np.load(
        os.path.join(self.path, "event_indicator.npy")
        )
        clusters_per_event = np.bincount(event_indicator)
        return event_indicator, clusters_per_event
    
    def _get_fibre_data(self):
        try:
            self.fibre_indicator = np.load(
                os.path.join(self.path, "fibre_indicator.npy")
            )
            self.fibre_positions = np.load(
                os.path.join(self.path, "fibre_positions.npy")
            )
            self.has_fibre_data = True
            logging.info("Fibre positions and indicators loaded.")
        except FileNotFoundError as e:
            logging.warning(f"Required file not found: {e}")
            logging.warning("Fibre positions and indicators will not be loaded.")
            self.fibre_indicator = None
            self.fibre_positions = None
            self.has_fibre_data = False
    
    def _set_mode(self):
        if self.mode == None:
            logging.warning("Mode not set. Trying to detect mode...")
            try:
                self.event_indicator, self.clusters_per_event = self._get_event_indicator()
                self.mode = "CM-4to1"
            except:
                self.mode = "CC-4to1"
            logging.info(f"Detected mode: {self.mode}")
        else:
            logging.info(f"Set mode: {self.mode}")
            if self.mode == "CM-4to1":
                self.event_indicator, self.clusters_per_event = self._get_event_indicator()
    
    def _build_block(self, block_size, start_index):
        logging.info("Starting to build event block: {}".format(self.mode))
        if self.mode == "CC-4to1":
            block_events = []
            for i in np.arange(start_index, start_index + block_size, 1):

                sipm_attributes = self.x_list.pop(0)
                graph_label = self.labels[i]
                graph_attributes = self.y_list[i]
                block_events.append(
                    Event([Cluster(sipm_attributes, graph_label, graph_attributes)], mode=self.mode)
                )

                yield block_events

        elif self.mode == "CM-4to1":
            # Create events from clusters for particular block
            logging.info("Counting clusters in block")
            cluster_counter = sum(self.clusters_per_event[:start_index])
            logging.info(f"There are {len(self.clusters_per_event)} clusters in total")
            logging.info("Starting to assemble events from clusters")
            for start_idx in range(start_index, len(self.clusters_per_event), block_size):
                end_idx = min(start_idx + block_size, len(self.clusters_per_event))
                block_events = []

                for cluster_count in tqdm(
                    self.clusters_per_event[start_idx:end_idx],
                    desc="Assembling events",
                    total=end_idx - start_idx,
                ):
                    event_list = []
                    for _ in range(cluster_count):
                        sipm_attributes = self.x_list.pop(0)
                        cluster_label = self.labels[cluster_counter]
                        cluster_attributes = self.y_list[cluster_counter]
                        event_list.append(
                            Cluster(sipm_attributes, cluster_label, cluster_attributes)
                        )
                        cluster_counter += 1
                    block_events.append(Event(event_list, self.mode))

                yield block_events
    
    def get_mode(self):
        return self.mode

    
    """@property
    def labels(self):
        return np.load(os.path.join(self.path, "graph_labels.npy"))"""
