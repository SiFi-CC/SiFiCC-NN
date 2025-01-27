import numpy as np

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
