import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Detector:
    """
    Class representing a Detector made up of SiPMs.
    """

    def __init__(self):
        self.sipm_size = 4
        self.sipm_bins0_bottom = np.arange(-55, 53 + self.sipm_size, self.sipm_size)
        self.sipm_bins1_bottom = -51
        self.sipm_bins2_bottom = np.arange(226, 238 + self.sipm_size, self.sipm_size)
        self.sipm_bins0_top = np.arange(-53, 55 + self.sipm_size, self.sipm_size)
        self.sipm_bins1_top = 51
        self.sipm_bins2_top = np.arange(228, 240 + self.sipm_size, self.sipm_size)
        self.sipm_positions = self.generate_sipm_positions()

    def generate_sipm_positions(self):
        """
        Generate all possible SiPM positions in the detector.
        :return: A numpy array of all possible positions.
        """
        sipm_positions = []
        for i in range(len(self.sipm_bins0_bottom)):
            for j in range(len(self.sipm_bins2_bottom)):
                sipm_positions.append(
                    [
                        self.sipm_bins0_bottom[i],
                        self.sipm_bins1_bottom,
                        self.sipm_bins2_bottom[j],
                    ]
                )
        for i in range(len(self.sipm_bins0_top)):
            for j in range(len(self.sipm_bins2_top)):
                sipm_positions.append(
                    [
                        self.sipm_bins0_top[i],
                        self.sipm_bins1_top,
                        self.sipm_bins2_top[j],
                    ]
                )
        return np.array(sipm_positions)


class SiPM:
    """
    Class representing a SiPM.
    """

    def __init__(self, position):
        """
        Initialize a SiPM with its 3D position.
        :param position: A list or array containing [x, y, z] coordinates.
        """
        self.position = position

    def __repr__(self):
        return f"SiPM(position={self.position})"


class Cluster:
    """
    Class representing a Cluster containing multiple SiPMs.
    """

    def __init__(self, sipms):
        """
        Initialize a cluster with a list of SiPMs.
        :param sipms: List of SiPM objects belonging to the cluster.
        """
        self.sipms = sipms

    def __repr__(self):
        return f"Cluster(num_sipms={len(self.sipms)})"


class Event:
    """
    Class representing an Event containing multiple Clusters.
    """

    def __init__(self, clusters):
        """
        Initialize an event with a list of Clusters.
        :param clusters: List of Cluster objects in the event.
        """
        self.clusters = clusters

    def __repr__(self):
        return f"Event(num_clusters={len(self.clusters)})"

    def plot(self, detector, event_idx):
        """
        Plot a 3D representation of the event with all possible SiPMs and activated SiPMs.
        :param detector: Detector object containing all possible SiPM positions.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Get all positions and activated positions
        all_positions = detector.sipm_positions
        activated_positions = set()

        # Define colors for clusters (stronger colors)
        cluster_colors = plt.cm.get_cmap(
            "plasma", len(self.clusters)
        )  # Using a brighter color map

        # Plot activated SiPMs by cluster and keep track of activated positions
        for idx, cluster in enumerate(self.clusters):
            cluster_positions = np.array([sipm.position for sipm in cluster.sipms])
            # Color for the current cluster
            cluster_color = cluster_colors(idx)

            # Plot SiPMs in the current cluster
            ax.scatter(
                cluster_positions[:, 0],
                cluster_positions[:, 1],
                cluster_positions[:, 2],
                label=f"Cluster {idx}",
                color=cluster_color,
                s=100,
            )  # Larger size for activated SiPMs

            # Store activated positions in a set
            activated_positions.update(map(tuple, cluster_positions))

            # Calculate the bounding box (cuboid) around the cluster SiPMs
            min_vals = np.min(cluster_positions, axis=0)
            max_vals = np.max(cluster_positions, axis=0)

            # Create the vertices for the cuboid
            x = [min_vals[0], max_vals[0]]
            y = [min_vals[1], max_vals[1]]
            z = [min_vals[2], max_vals[2]]

            # Create the 8 vertices of the cuboid
            vertices = np.array(
                [
                    [x[0], y[0], z[0]],
                    [x[0], y[0], z[1]],
                    [x[0], y[1], z[1]],
                    [x[0], y[1], z[0]],
                    [x[1], y[0], z[0]],
                    [x[1], y[0], z[1]],
                    [x[1], y[1], z[1]],
                    [x[1], y[1], z[0]],
                ]
            )

            # Define the 12 faces of the cuboid
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom face
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top face
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front face
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back face
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left face
                [vertices[1], vertices[2], vertices[6], vertices[5]],
            ]  # right face

            # Create a Poly3DCollection for the cuboid with translucent fill
            poly3d = Poly3DCollection(faces, color=cluster_color, alpha=0.3)
            ax.add_collection3d(poly3d)

        # Plot gray inactive SiPMs, but avoid positions that are activated
        inactive_positions = [
            pos for pos in all_positions if tuple(pos) not in activated_positions
        ]
        inactive_positions = np.array(inactive_positions)

        ax.scatter(
            inactive_positions[:, 0],
            inactive_positions[:, 1],
            inactive_positions[:, 2],
            color="gray",
            alpha=0.3,
            label="Inactive SiPMs",
        )

        # Set plot labels and title
        ax.set_title("3D Event Visualization of Event " + str(event_idx))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Ensure equal scaling for all axes
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()
        max_range = (
            max(
                x_limits[1] - x_limits[0],
                y_limits[1] - y_limits[0],
                z_limits[1] - z_limits[0],
            )
            / 2.0
        )

        mid_x = (x_limits[0] + x_limits[1]) / 2.0
        mid_y = (y_limits[0] + y_limits[1]) / 2.0
        mid_z = (z_limits[0] + z_limits[1]) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        plt.savefig(f"event_{event_idx}.png")


class EventLoader:
    """
    Lazy loader for events.
    """

    def __init__(self, clusters_file, sipms_per_cluster_file, sipm_positions_file):
        self.clusters_file = clusters_file
        self.sipms_per_cluster_file = sipms_per_cluster_file
        self.sipm_positions_file = sipm_positions_file
        self.num_clusters_per_event = np.load(clusters_file)
        self.num_sipms_per_cluster = np.load(sipms_per_cluster_file)
        self.sipm_positions = np.load(sipm_positions_file)
        self.cumulative_sipms_per_cluster = np.cumsum(self.num_sipms_per_cluster)
        self.cumulative_clusters_per_event = np.cumsum(self.num_clusters_per_event)

    def load_event(self, event_idx):
        """
        Load a specific event by index.
        :param event_idx: Index of the event to load.
        :return: Event object.
        """

        clusters = []
        n_clusters = self.num_clusters_per_event[event_idx]
        right_bound = self.cumulative_clusters_per_event[event_idx]
        left_bound = right_bound - n_clusters
        for cluster_idx in range(left_bound, right_bound):
            n_sipms = self.num_sipms_per_cluster[cluster_idx]
            right_bound_sipm = self.cumulative_sipms_per_cluster[cluster_idx]
            left_bound_sipm = right_bound_sipm - n_sipms
            sipm_positions = self.sipm_positions[left_bound_sipm:right_bound_sipm]
            sipms = [SiPM(position=pos) for pos in sipm_positions]
            clusters.append(Cluster(sipms=sipms))

        return Event(clusters=clusters)


def main():
    # File paths
    clusters_file = "number_of_clusters.npy"
    sipms_per_cluster_file = "sipms_per_cluster.npy"
    sipm_positions_file = "sipm_positions.npy"

    cluster_multiplicity = np.load(clusters_file)

    # Initialize detector
    detector = Detector()

    # Initialize event loader
    event_loader = EventLoader(
        clusters_file, sipms_per_cluster_file, sipm_positions_file
    )

    # Load and visualize specific events
    for number_of_clusters in np.unique(cluster_multiplicity):
        event_ids = np.where(cluster_multiplicity == number_of_clusters)[0]
        for event_idx in event_ids[:3]:
            event = event_loader.load_event(event_idx)
            print(f"Visualizing Event {event_idx}: {event}")
            event.plot(detector, event_idx)


if __name__ == "__main__":
    main()
