import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DatasetReader:
    def __init__(self, dataset_path, show_adjacency_matrix=False, **kwargs):
        self.name = dataset_path.split("/")[-1]
        self.dataset_path = dataset_path
        self.show_adjacency_matrix = show_adjacency_matrix
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
                    os.path.join(self.path, "graph_indicator.npy"))
                self.n_nodes = np.bincount(self.node_batch_index)
                self.n_nodes_cum = np.concatenate(
                    ([0], np.cumsum(self.n_nodes)[:-1]))

                self.x_list = self._get_x_list(self.n_nodes_cum)
                self.y_list = self._get_y_list()
                self.labels = np.load(os.path.join(
                    self.path, "graph_labels.npy"))
                self.event_indicator = np.load(
                    os.path.join(self.path, "event_indicator.npy"))
                self.clusters_per_event = np.bincount(self.event_indicator)
            except FileNotFoundError as e:
                logging.error(f"Required file not found: {e}")
                raise

            logging.info(f"Successfully loaded dataset: {self.name}")

        # Create events from clusters for particular block
        cluster_counter = sum(self.clusters_per_event[:start_index])

        for start_idx in range(start_index, len(self.clusters_per_event), block_size):
            end_idx = min(start_idx + block_size, len(self.clusters_per_event))
            block_events = []

            for cluster_count in tqdm(self.clusters_per_event[start_idx:end_idx], desc="Assembling events", total=end_idx - start_idx):
                event_list = []
                for _ in range(cluster_count):
                    sipm_attributes = self.x_list.pop(0)
                    cluster_label = self.labels[cluster_counter]
                    cluster_attributes = self.y_list[cluster_counter]
                    event_list.append(
                        Cluster(sipm_attributes, cluster_label, cluster_attributes))
                    cluster_counter += 1
                block_events.append(Event(event_list))

            yield block_events

    def _get_x_list(self, n_nodes_cum):
        sipm_attributes = np.load(os.path.join(
            self.path, "node_attributes.npy"))
        return np.split(sipm_attributes, n_nodes_cum[1:])

    def _get_y_list(self):
        return np.load(os.path.join(self.path, "graph_attributes.npy"))

    @property
    def sp(self):
        return np.load(os.path.join(self.path, "graph_sp.npy"))

    @property
    def pe(self):
        return np.load(os.path.join(self.path, "graph_pe.npy"))

    """@property
    def labels(self):
        return np.load(os.path.join(self.path, "graph_labels.npy"))"""


class Detector:
    """
    A class used to represent a Detector with SiPM (Silicon Photomultiplier) positions.
    Attributes
    ----------
    sipm_size : int
        The size of each SiPM.
    sipm_bins0_bottom : numpy.ndarray
        The x-coordinates of the bottom SiPMs.
    sipm_bins1_bottom : int
        The y-coordinate of the bottom SiPMs.
    sipm_bins2_bottom : numpy.ndarray
        The z-coordinates of the bottom SiPMs.
    sipm_bins0_top : numpy.ndarray
        The x-coordinates of the top SiPMs.
    sipm_bins1_top : int
        The y-coordinate of the top SiPMs.
    sipm_bins2_top : numpy.ndarray
        The z-coordinates of the top SiPMs.
    sipm_positions : numpy.ndarray
        The positions of all SiPMs.
    Methods
    -------
    __init__():
        Initializes the Detector and generates SiPM positions.
    _initialize_sipm_bins():
        Initializes the SiPM bin coordinates.
    _generate_sipm_positions():
        Generates the positions of the SiPMs based on the initialized bins.
    """

    def __init__(self):
        self._initialize_sipm_bins()
        self.sipm_positions = self._generate_sipm_positions()

    def _initialize_sipm_bins(self):
        self.sipm_size = 4
        self.sipm_bins0_bottom = np.arange(-55,
                                           53 + self.sipm_size, self.sipm_size)
        self.sipm_bins1_bottom = -51
        self.sipm_bins2_bottom = np.arange(
            226, 238 + self.sipm_size, self.sipm_size)
        self.sipm_bins0_top = np.arange(-53,
                                        55 + self.sipm_size, self.sipm_size)
        self.sipm_bins1_top = 51
        self.sipm_bins2_top = np.arange(
            228, 240 + self.sipm_size, self.sipm_size)

    def _generate_sipm_positions(self):
        bottom_positions = np.array(
            [[x, self.sipm_bins1_bottom, z]
                for x in self.sipm_bins0_bottom for z in self.sipm_bins2_bottom]
        )
        top_positions = np.array(
            [[x, self.sipm_bins1_top, z]
                for x in self.sipm_bins0_top for z in self.sipm_bins2_top]
        )
        return np.vstack((bottom_positions, top_positions))


class SiPM:
    """
    A class to represent a Silicon Photomultiplier (SiPM) sensor.

    Attributes
    ----------
    position : list
        The position of the SiPM sensor in 3D space.
    timestamp : float
        The timestamp associated with the SiPM sensor event.
    photoncount : int
        The number of photons detected by the SiPM sensor.

    Methods
    -------
    __repr__():
        Returns a string representation of the SiPM object.

    Constructs all the necessary attributes for the SiPM object.

    Parameters
    ----------
    node : list
        A list containing the SiPM sensor data, where the first three elements
        represent the position, the fifth element represents the photon count,
        and the sixth element represents the timestamp.
    """

    def __init__(self, node):
        self.position = node[:3]
        self.timestamp = node[5]
        self.photoncount = node[4]

    def __repr__(self):
        return f"SiPM(position={self.position})"


class Cluster:
    """
    A class used to represent a Cluster of SiPMs (Silicon Photomultipliers).

    Attributes
    ----------
    sipms : list
        A list of nodes representing the SiPMs in the cluster.
    label : any
        A label associated with the cluster.
    cluster_hit : any
        Information about the cluster hit.
    nSipms : int
        The number of SiPMs in the cluster.

    Methods
    -------
    get_bounding_box():
        Returns the minimum and maximum coordinates of the SiPM positions in the cluster.
    """

    def __init__(self, nodes, label, cluster_hit):
        self.sipms = nodes
        self.label = label
        self.cluster_hit = cluster_hit
        self.nSipms = len(self.sipms)

    def get_bounding_box(self):
        positions = np.array([sipm[:3] for sipm in self.sipms])
        return np.min(positions, axis=0), np.max(positions, axis=0)

    def __repr__(self):
        return f"Cluster(num_sipms={len(self.sipms)})"


class Event:
    """
    A class to represent an event consisting of multiple clusters.

    Attributes
    ----------
    clusters : list
        A list of clusters in the event.
    nClusters : int
        The number of clusters in the event.
    contains_coupling_hit : bool
        Indicates if any cluster contains a coupling hit.

    Methods
    -------
    plot(detector, event_idx, show_sipms=False, show_cluster_area=False, show_photon_hits=False, ax=None):
        Plots a 3D visualization of the event.
    """

    def __init__(self, clusters):
        self.clusters = clusters
        self.nClusters = len(self.clusters)
        self.contains_coupling_hit = 0 in [
            cluster.label for cluster in self.clusters]

    def plot(self, detector, event_idx, show_sipms=False, show_cluster_area=False, show_photon_hits=False, ax=None):
        """
        Plots a 3D visualization of the event with various options for displaying SiPMs, cluster areas, and photon hits.

        Parameters:
        detector (object): The detector object containing SiPM positions.
        event_idx (int): The index of the event to be visualized.
        show_sipms (bool, optional): If True, displays the SiPM positions. Default is False.
        show_cluster_area (bool, optional): If True, displays the bounding box of the clusters. Default is False.
        show_photon_hits (bool, optional): If True, displays the assumed photon hit positions. Default is False.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): The 3D axis to plot on. If None, a new figure and axis are created. Default is None.

        Returns:
        None
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

        cluster_colors = plt.cm.get_cmap('plasma', len(self.clusters))

        for idx, cluster in enumerate(self.clusters):
            cluster_positions = np.array([sipm[:3] for sipm in cluster.sipms])
            cluster_color = cluster_colors(idx)
            activated_positions = set()
            all_positions = detector.sipm_positions
            if show_sipms:
                ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], cluster_positions[:, 2],
                           label=f'Cluster {idx}', color=cluster_color, s=100)
                activated_positions.update(map(tuple, cluster_positions))
                print("SiPM positions", cluster_positions)

            if show_cluster_area:
                min_vals, max_vals = cluster.get_bounding_box()

                vertices = np.array([[min_vals[0], min_vals[1], min_vals[2]],
                                     [min_vals[0], min_vals[1], max_vals[2]],
                                     [min_vals[0], max_vals[1], max_vals[2]],
                                     [min_vals[0], max_vals[1], min_vals[2]],
                                     [max_vals[0], min_vals[1], min_vals[2]],
                                     [max_vals[0], min_vals[1], max_vals[2]],
                                     [max_vals[0], max_vals[1], max_vals[2]],
                                     [max_vals[0], max_vals[1], min_vals[2]]])

                faces = [[vertices[0], vertices[1], vertices[2], vertices[3]],
                         [vertices[4], vertices[5], vertices[6], vertices[7]],
                         [vertices[0], vertices[1], vertices[5], vertices[4]],
                         [vertices[2], vertices[3], vertices[7], vertices[6]],
                         [vertices[0], vertices[3], vertices[7], vertices[4]],
                         [vertices[1], vertices[2], vertices[6], vertices[5]]]

                poly3d = Poly3DCollection(
                    faces, color=cluster_color, alpha=0.3)
                ax.add_collection3d(poly3d)

            if show_photon_hits:
                cluster_hit_position = cluster.cluster_hit[1:]
                print("cluster_hit_position", cluster_hit_position)
                ax.scatter(cluster_hit_position[0], cluster_hit_position[1], cluster_hit_position[2],
                           color='red', marker='*', s=200, label=f'assumed photon Hit {idx}')

            inactive_positions = [pos for pos in all_positions if tuple(
                pos) not in activated_positions]
            inactive_positions = np.array(inactive_positions)

            ax.scatter(inactive_positions[:, 0], inactive_positions[:, 1], inactive_positions[:, 2],
                       color='gray', alpha=0.3, label='Inactive SiPMs')

        ax.set_title(f"3D Event Visualization of Event {event_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        mid_x, mid_y, mid_z = map(lambda lim: (
            lim[0] + lim[1]) / 2.0, [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        max_range = max(
            ax.get_xlim()[1] - ax.get_xlim()[0],
            ax.get_ylim()[1] - ax.get_ylim()[0],
            ax.get_zlim()[1] - ax.get_zlim()[0]
        ) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()

    def __repr__(self):
        return f"Event(num_clusters={len(self.clusters)})"


def main():
    name = 'OptimisedGeometry_CodedMaskHIT_Spot1_1e10_protons_MK'
    reader = DatasetReader(name)

    # Initialize detector
    detector = Detector()

    for event_idx, block in enumerate(reader.read()):
        for idx, event in enumerate(block):
            event.plot(detector, event_idx * 100 + idx, show=True)


if __name__ == "__main__":
    main()
