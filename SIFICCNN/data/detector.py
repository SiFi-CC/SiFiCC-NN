import numpy as np


class Detector:
    """
    A container that represents one module of the SiFi-CC detector. The data represents the
    dimensions and positions of the module

    Attributes:
        pos: ndarray (3,); vector pointing towards the middle of the detector module
        dimx: float; detector thickness in x-dimension
        dimy: float; detector thickness in y-dimension
        dimz: float; detector thickness in z-dimension

    """

    def __init__(self, pos, dimx, dimy, dimz):
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        # position is defined by the Detector-Module middle
        self.posx = pos["fX"]
        self.posy = pos["fY"]
        self.posz = pos["fZ"]
        self.pos = np.array([self.posx, self.posy, self.posy])
