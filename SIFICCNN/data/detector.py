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

    def is_vec_in_module(self, input_vec, a=0.001):
        """
        Checks if a vector points inside the module.

        Args:
            input_vec (TVector3) or (list<TVector3>): If a list type of vectors is given, True will
                                                      be returned. If at least one of the vectors is
                                                      inside the detector.
            a (float):                                some events are right on the border of the
                                                      detector. The factor "a" adds a small buffer
                                                      to compensate for float uncertainties

        Return:
            True if vector points inside module, False otherwise

        """
        # check type of parameter vec
        # TODO: do proper type check

        try:
            for vec in input_vec:
                # check if vector is inside detector boundaries
                if (
                    abs(self.posx - vec.x) <= self.dimx / 2 + a
                    and abs(self.posy - vec.y) <= self.dimy / 2 + a
                    and abs(self.posz - vec.z) <= self.dimz / 2 + a
                ):
                    return True
            return False
        # if input_vec is not iterable
        except TypeError:
            if (
                abs(self.posx - input_vec.x) <= self.dimx / 2 + a
                and abs(self.posy - input_vec.y) <= self.dimy / 2 + a
                and abs(self.posz - input_vec.z) <= self.dimz / 2 + a
            ):
                return True
            else:
                return False
