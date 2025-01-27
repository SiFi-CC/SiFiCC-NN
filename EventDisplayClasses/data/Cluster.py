import numpy as np

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