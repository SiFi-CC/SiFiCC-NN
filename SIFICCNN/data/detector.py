import awkward as ak
import numpy as np
from numba import njit

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

    def __init__(self, posx, posy, posz, dimx, dimy, dimz):
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        # position is defined by the Detector-Module middle
        self.posx = posx
        self.posy = posy
        self.posz = posz

    @classmethod
    def from_root(cls, pos, dimx, dimy, dimz):
        return cls(
            posx=pos["fX"],
            posy=pos["fY"],
            posz=pos["fZ"],
            dimx=dimx,
            dimy=dimy,
            dimz=dimz,
        )
    
    def get_detector_dimensions(self):
        """
        Returns the dimensions of the detector module.

        Returns:
            tuple: Dimensions of the detector module (dimx, dimy, dimz, posx, posy, posz).
        """
        return np.array([self.dimx, self.dimy, self.dimz, self.posx, self.posy, self.posz])

    def is_vec_in_module_ak(self, input_vec, a=0.001):
        """
        Vectorized version of `is_vec_in_module()` using Awkward Arrays.

        Args:
            input_vec (ak.Array of dicts {"x", "y", "z"}): Input vector(s) to check.
            a (float): Buffer to compensate for float uncertainties.

        Returns:
            ak.Array[bool]: Boolean mask indicating which vectors are inside the module.
        """

        # Ensure input is an Awkward Array (convert if necessary)
        if not isinstance(input_vec, ak.Array):
            input_vec = ak.Array(input_vec)

        # Compute masks for each coordinate
        mask_x = abs(self.posx - input_vec["x"]) <= self.dimx / 2 + a
        mask_y = abs(self.posy - input_vec["y"]) <= self.dimy / 2 + a
        mask_z = abs(self.posz - input_vec["z"]) <= self.dimz / 2 + a

        # Final mask (vector must be inside all three dimensions)
        inside_mask = mask_x & mask_y & mask_z

        return inside_mask
    