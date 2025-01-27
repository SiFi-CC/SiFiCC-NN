import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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

    def __init__(self, clusters, mode):
        self.clusters = clusters
        self.nClusters = len(self.clusters)
        self.mode = mode
        if self.mode == "CC-4to1":
            self.contains_non_compton_hit = 0 in [
                cluster.label for cluster in self.clusters
            ]
        elif self.mode == "CM-4to1":
            self.contains_coupling_hit = 0 in [cluster.label for cluster in self.clusters]


    def plot(
        self,
        detector,
        event_idx,
        show_sipms=False,
        show_cluster_area=False,
        show_compton_hits=False,
        show_CMphoton_hits=False,
        show_fibre_hits=False,
        ax=None,
    ):
        """
        Plots a 3D visualization of the event with various options for displaying SiPMs, cluster areas, and photon hits.

        Parameters:
        detector (object): The detector object containing SiPM positions.
        event_idx (int): The index of the event to be visualized.
        show_sipms (bool, optional): If True, displays the SiPM positions. Default is False.
        show_cluster_area (bool, optional): If True, displays the bounding box of the clusters. Default is False.
        show_CMphoton_hits (bool, optional): If True, displays the assumed photon hit positions. Default is False.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): The 3D axis to plot on. If None, a new figure and axis are created. Default is None.

        Returns:
        None
        """
        logging.info(f"Plotting event {event_idx}")
        logging.info(f"The following arguments are set: show_sipms={show_sipms}, show_cluster_area={show_cluster_area}, show_photon_hits={show_CMphoton_hits}")
        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")

        cluster_colors = plt.cm.get_cmap("plasma", len(self.clusters))

        for idx, cluster in enumerate(self.clusters):
            cluster_positions = np.array([sipm[:3] for sipm in cluster.sipms])
            cluster_color = cluster_colors(idx)
            activated_positions = set()
            all_positions = detector.sipm_positions
            if show_sipms:
                ax.scatter(
                    cluster_positions[:, 0],
                    cluster_positions[:, 1],
                    cluster_positions[:, 2],
                    label=f"Cluster {idx}",
                    color=cluster_color,
                    s=100,
                )
                activated_positions.update(map(tuple, cluster_positions))
                logging.info("SiPM positions", cluster_positions)

            if show_cluster_area:
                min_vals, max_vals = cluster.get_bounding_box()

                vertices = np.array(
                    [
                        [min_vals[0], min_vals[1], min_vals[2]],
                        [min_vals[0], min_vals[1], max_vals[2]],
                        [min_vals[0], max_vals[1], max_vals[2]],
                        [min_vals[0], max_vals[1], min_vals[2]],
                        [max_vals[0], min_vals[1], min_vals[2]],
                        [max_vals[0], min_vals[1], max_vals[2]],
                        [max_vals[0], max_vals[1], max_vals[2]],
                        [max_vals[0], max_vals[1], min_vals[2]],
                    ]
                )

                edges = [
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]],
                    [vertices[4], vertices[5]],
                    [vertices[5], vertices[6]],
                    [vertices[6], vertices[7]],
                    [vertices[7], vertices[4]],
                    [vertices[0], vertices[4]],
                    [vertices[1], vertices[5]],
                    [vertices[2], vertices[6]],
                    [vertices[3], vertices[7]],
                ]

                edge_collection = Line3DCollection(edges, colors=[cluster_color], linestyles='dotted', linewidths=1)
                ax.add_collection3d(edge_collection)


            if show_compton_hits:
                if self.contains_non_compton_hit:
                    logging.info("No Compton Hit found!")
                else:
                    cluster_hit_position = cluster.cluster_hit[2:5]
                    logging.info("cluster_hit_position", cluster_hit_position)
                    ax.scatter(
                        cluster_hit_position[0],
                        cluster_hit_position[1],
                        cluster_hit_position[2],
                        color="red",
                        marker="*",
                        s=200,
                        label=f"Compton e Position",
                    )
            elif show_CMphoton_hits:
                logging.info(f"Plotting photon hits at positions: {cluster.cluster_hit}")
                cluster_hit_position = cluster.cluster_hit[1:]
                logging.info("cluster_hit_position", cluster_hit_position)
                ax.scatter(
                    cluster_hit_position[0],
                    cluster_hit_position[1],
                    cluster_hit_position[2],
                    color="red",
                    marker="*",
                    s=200,
                    label=f"assumed photon Hit {idx}",
                )

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

        ax.set_title(f"3D Event Visualization of Event {event_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        mid_x, mid_y, mid_z = map(
            lambda lim: (lim[0] + lim[1]) / 2.0,
            [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()],
        )
        max_range = (
            max(
                ax.get_xlim()[1] - ax.get_xlim()[0],
                ax.get_ylim()[1] - ax.get_ylim()[0],
                ax.get_zlim()[1] - ax.get_zlim()[0],
            )
            / 2.0
        )

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()

    def __repr__(self):
        return f"Event(num_clusters={len(self.clusters)})"