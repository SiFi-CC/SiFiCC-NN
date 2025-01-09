
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                self.node_batch_index = np.load(os.path.join(self.path, "graph_indicator.npy"))
                self.n_nodes = np.bincount(self.node_batch_index)
                self.n_nodes_cum = np.concatenate(([0], np.cumsum(self.n_nodes)[:-1]))

                self.x_list = self._get_x_list(self.n_nodes_cum)
                self.y_list = self._get_y_list()
                self.labels = np.load(os.path.join(self.path, "graph_labels.npy"))
            except FileNotFoundError as e:
                logging.error(f"Required file not found: {e}")
                raise

            logging.info(f"Successfully loaded dataset: {self.name}")

        block_events = []
        for i in np.arange(start_index, start_index+block_size, 1):
            
            sipm_attributes = self.x_list.pop(0)
            graph_label = self.labels[i]
            graph_attributes = self.y_list[i]
            block_events.append(Event([Cluster(sipm_attributes, graph_label, graph_attributes)]))

        yield block_events

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

        self.sipm_bins0_bottom_scatterer = np.arange(-55, 53 + self.sipm_size, self.sipm_size)
        self.sipm_bins1_bottom_scatterer = -51
        self.sipm_bins2_bottom_scatterer = np.arange(143, 155 + self.sipm_size, self.sipm_size)
        self.sipm_bins0_top_scatterer = np.arange(-53, 55 + self.sipm_size, self.sipm_size)
        self.sipm_bins1_top_scatterer = 51
        self.sipm_bins2_top_scatterer = np.arange(145, 157 + self.sipm_size, self.sipm_size)

        self.sipm_bins0_bottom_absorber = np.arange(-63, 61 + self.sipm_size, self.sipm_size)
        self.sipm_bins1_bottom_absorber = -51
        self.sipm_bins2_bottom_absorber = np.arange(255, 283 + self.sipm_size, self.sipm_size)
        self.sipm_bins0_top_absorber = np.arange(-61, 63 + self.sipm_size, self.sipm_size)
        self.sipm_bins1_top_absorber = 51
        self.sipm_bins2_top_absorber = np.arange(257, 285 + self.sipm_size, self.sipm_size)



    def _generate_sipm_positions(self):
        bottom_positions_scatterer = np.array(
            [[x, self.sipm_bins1_bottom_scatterer, z] for x in self.sipm_bins0_bottom_scatterer for z in self.sipm_bins2_bottom_scatterer]
        )
        top_positions_scatterer = np.array(
            [[x, self.sipm_bins1_top_scatterer, z] for x in self.sipm_bins0_top_scatterer for z in self.sipm_bins2_top_scatterer]
        )
        bottom_positions_absorber = np.array(
            [[x, self.sipm_bins1_bottom_absorber, z] for x in self.sipm_bins0_bottom_absorber for z in self.sipm_bins2_bottom_absorber]
        )   
        top_positions_absorber = np.array(
            [[x, self.sipm_bins1_top_absorber, z] for x in self.sipm_bins0_top_absorber for z in self.sipm_bins2_top_absorber]
        )

        bottom_positions = np.vstack((bottom_positions_scatterer, bottom_positions_absorber))
        top_positions = np.vstack((top_positions_scatterer, top_positions_absorber))

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
        self.contains_non_compton_hit = 0 in [cluster.label for cluster in self.clusters]

    def plot(self, detector, event_idx, show_sipms=False, show_cluster_area=False, show_compton_hits=False, ax=None):
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

                poly3d = Poly3DCollection(faces, color=cluster_color, alpha=0.3)
                ax.add_collection3d(poly3d)


            if show_compton_hits:
                if self.contains_non_compton_hit:
                    print("No Compton Hit found!")
                else:
                    cluster_hit_position = cluster.cluster_hit[2:5] ##################################################
                    print("cluster_hit_position", cluster_hit_position)
                    ax.scatter(cluster_hit_position[0], cluster_hit_position[1], cluster_hit_position[2],
                           color='red', marker='*', s=200, label=f'Compton e Position')


            inactive_positions = [pos for pos in all_positions if tuple(pos) not in activated_positions]
            inactive_positions = np.array(inactive_positions)

            ax.scatter(inactive_positions[:, 0], inactive_positions[:, 1], inactive_positions[:, 2],
                       color='gray', alpha=0.3, label='Inactive SiPMs')

        ax.set_title(f"3D Event Visualization of Event {event_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        mid_x, mid_y, mid_z = map(lambda lim: (lim[0] + lim[1]) / 2.0, [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
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
    name = 'datasets/SimGraphSiPM/OptimisedGeometry_4to1_0mm_3.9e9protons_simv4'
    reader = DatasetReader(name)

    # Initialize detector
    detector = Detector()

    for event_idx, block in enumerate(reader.read()):
        for idx, event in enumerate(block):
            event.plot(detector, event_idx * 100 + idx)

if __name__ == "__main__":
    main()


"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_edges(x, y, z, xdim, ydim, zdim):
    """"""
    Calculates a list of all edges of a cuboid. Used to feed matplotlib methods.

    Args:
        x (float): x-coordinate of center
        y (float): y-coordinate of center
        z (float): z-coordinate of center
        xdim (float): length of cuboid in x-dimension
        ydim (float): length of cuboid in y-dimension
        zdim (float): length of cuboid in z-dimension

    Return:
        list of edges (list of 3-length lists)""""""
    list_edges = [
        [[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z - zdim / 2]],
        [[x - xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z - zdim / 2]],
        [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y - ydim / 2], [z - zdim / 2, z - zdim / 2]],
        [[x + xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z - zdim / 2]],
        [[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y + ydim / 2], [z + zdim / 2, z + zdim / 2]],
        [[x - xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z + zdim / 2, z + zdim / 2]],
        [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y - ydim / 2], [z + zdim / 2, z + zdim / 2]],
        [[x + xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z + zdim / 2, z + zdim / 2]],
        [[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z + zdim / 2]],
        [[x - xdim / 2, x - xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]],
        [[x + xdim / 2, x + xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z + zdim / 2]],
        [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]]]

    return list_edges


def get_surface(x, y, z, xdim, ydim, zdim):
    """"""
    Calculates a list of all surfaces of a cuboid. Used to feed matplotlib methods.

    Args:
        x (float): x-coordinate of center
        y (float): y-coordinate of center
        z (float): z-coordinate of center
        xdim (float): length of cuboid in x-dimension
        ydim (float): length of cuboid in y-dimension
        zdim (float): length of cuboid in z-dimension

    Return:
        list of edges (list of 3-length lists)
    """"""

    one = np.ones(4).reshape(2, 2)
    list_surface = [
        [[x - xdim / 2, x + xdim / 2], [y - ydim / 2, y + ydim / 2], (z - zdim / 2) * one],
        [[x - xdim / 2, x + xdim / 2], [y - ydim / 2, y + ydim / 2], (z + zdim / 2) * one],
        [[x - xdim / 2, x + xdim / 2], (y - ydim / 2) * one, [z - zdim / 2, z + zdim / 2]],
        [[x - xdim / 2, x + xdim / 2], (y + ydim / 2) * one, [z - zdim / 2, z + zdim / 2]],
        [(x - xdim / 2) * one, [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]],
        [(x + xdim / 2) * one, [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]]]
    return list_surface


def unit_vec(vec):
    """"""
    Returns the unit vector of a given vector.

    Args:
        vec (TVector3): vector

    Returns:
        unit vector (TVector3)

    """"""
    return vec / np.sqrt(np.dot(vec, vec))


def vec_angle(vec1, vec2):
    """"""
    Calculates the vector between two given angles.

    Args:
        vec1 (TVector3): vector 1
        vec2 (TVector3): vector 2

    Return:
         angle between vectors
    """"""
    return np.arccos(np.clip(np.dot(unit_vec(vec1), unit_vec(vec2)), -1.0, 1.0))


def get_compton_cone_cracow(vec_apex, vec_axis, vec_origin, theta, sr=8):
    """"""
    Computes the compton cone of a reconstructed event.

    Args:
        vec_apex    (TVector3): origin vector of true scatterer interaction
        vec_axis    (TVector3): vector pointing from true absorber interaction to true scatterer
                                interaction
        vec_origin  (TVector3): origin vector of true source position
        theta:
        sr:

    Returns:

    """"""
    # TODO: All vector rotations are done via scipy transformation library
    # TODO: Uproot vector rotations should be better suited
    # Correct angle theta (stems from the definition of the axis vector as it is flipped)
    theta = np.pi - theta

    # Rotate reference vector around scattering angle theta
    ref_vec = np.array([1, 0, 0])
    rotation_y = R.from_rotvec((vec_axis.theta - np.pi / 2 - theta) * np.array([0, 1, 0]))
    rotation_z = R.from_rotvec(vec_axis.phi * np.array([0, 0, 1]))
    ref_vec = rotation_y.apply(ref_vec)
    ref_vec = rotation_z.apply(ref_vec)

    # Rotate reference vector around axis vector to sample cone edges
    list_cone_vec = []
    rot_axis_ary = np.array([vec_axis.x, vec_axis.y, vec_axis.z])
    # Phi angle sampling (not the same as scattering angle theta!)
    list_phi = np.linspace(0, 360, sr)
    for angle in list_phi:
        vec_temp = ref_vec
        rot_vec = np.radians(angle) * rot_axis_ary / np.sqrt(np.dot(rot_axis_ary, rot_axis_ary))
        rot_M = R.from_rotvec(rot_vec)
        vec_temp = rot_M.apply(vec_temp)
        list_cone_vec.append(vec_temp)

    # scale each cone vector to hit the final canvas
    # shift them to correct final position

    for i in range(len(list_cone_vec)):
        a = -(vec_apex.x - vec_origin.x) / list_cone_vec[i][0]
        list_cone_vec[i] *= a
        list_cone_vec[i] = np.array([list_cone_vec[i][0] + vec_apex.x,
                                     list_cone_vec[i][1] + vec_apex.y,
                                     list_cone_vec[i][2] + vec_apex.z])

    return list_cone_vec


def get_compton_cone_aachen(vec_apex, vec_axis, vec_origin, theta, sr=8):
    """"""
    Computes the compton cone of a reconstructed event.

    Args:
        vec_apex    (TVector3): origin vector of true scatterer interaction
        vec_axis    (TVector3): vector pointing from true absorber interaction to true scatterer
                                interaction
        vec_origin  (TVector3): origin vector of true source position
        theta:
        sr:

    Returns:

    """"""
    # TODO: All vector rotations are done via scipy transformation library
    # TODO: Uproot vector rotations should be better suited
    # Correct angle theta (stems from the definition of the axis vector as it is flipped)
    theta = np.pi - theta

    # Rotate reference vector around scattering angle theta
    ref_vec = np.array([0, 0, 1])
    rotation_y = R.from_rotvec((np.pi - vec_axis.theta - theta) * np.array([0, 1, 0]))
    rotation_z = R.from_rotvec((vec_axis.phi + np.pi) * np.array([0, 0, 1]))
    ref_vec = rotation_y.apply(ref_vec)
    ref_vec = rotation_z.apply(ref_vec)

    # Rotate reference vector around axis vector to sample cone edges
    list_cone_vec = []
    rot_axis_ary = np.array([vec_axis.x, vec_axis.y, vec_axis.z])
    # Phi angle sampling (not the same as scattering angle theta!)
    list_phi = np.linspace(0, 360, sr)
    for angle in list_phi:
        vec_temp = ref_vec
        rot_vec = np.radians(angle) * rot_axis_ary / np.sqrt(np.dot(rot_axis_ary, rot_axis_ary))
        rot_M = R.from_rotvec(rot_vec)
        vec_temp = rot_M.apply(vec_temp)
        list_cone_vec.append(vec_temp)

    # scale each cone vector to hit the final canvas
    # shift them to correct final position
    for i in range(len(list_cone_vec)):
        a = -(vec_apex.z - vec_origin.z) / list_cone_vec[i][2]
        list_cone_vec[i] *= a
        list_cone_vec[i] = np.array([list_cone_vec[i][0] + vec_apex.x,
                                     list_cone_vec[i][1] + vec_apex.y,
                                     list_cone_vec[i][2] + vec_apex.z])

    return list_cone_vec"""