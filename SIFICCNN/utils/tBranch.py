import numpy as np
import awkward as ak

def convert_tvector3_to_arrays(batch, mode="ak"):
    """Convert TVector3 into separate x, y, z components"""
    if mode == "ak":
        return ak.zip({
            "x": batch["fX"],
            "y": batch["fY"],
            "z": batch["fZ"]
        })
    elif mode == "np":
        # Convert each field to a plain numpy array,
        # and fill masked values with 0.
        # Check if fX exists
        if "fX" in batch.fields:
            fX = np.ma.filled(ak.to_numpy(batch["fX"]), 0)
            fY = np.ma.filled(ak.to_numpy(batch["fY"]), 0)
            fZ = np.ma.filled(ak.to_numpy(batch["fZ"]), 0)
        elif "x" in batch.fields:
            x = np.ma.filled(ak.to_numpy(batch["x"]), 0)
            y = np.ma.filled(ak.to_numpy(batch["y"]), 0)
            z = np.ma.filled(ak.to_numpy(batch["z"]), 0)
        else:
            raise ValueError("No valid fields found in provided batch")
        # Stack them columnwise to create an array of shape (n, 3)
        return np.column_stack((fX, fY, fZ))

class TVector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # operator overload
    def __str__(self):
        return "TVector3({:.3f}, {:.3f}, {:.3f})".format(self.x, self.y, self.z)

    def __truediv__(self, other):
        return TVector3(self.x / other, self.y / other, self.z / other)

    def __add__(self, other):
        return TVector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return TVector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return TVector3(self.x * other, self.y * other, self.z * other)
        # additional exception for vector multiplication to include dot
        # products
        elif isinstance(other, TVector3):
            return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    # class methods
    @classmethod
    def zeros(cls):
        return cls(x=0, y=0, z=0)

    @classmethod
    def from_numpy(cls, ary):
        return cls(x=ary[0], y=ary[1], z=ary[2])

    @classmethod
    def from_akw(cls, akw):
        return cls(x=akw["fX"], y=akw["fY"], z=akw["fZ"])

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    # vector properties
    @property
    def mag(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def M(self):
        return self.mag()

    @property
    def perp2(self):
        return self.x**2 + self.y**2

    @property
    def perp(self):
        return np.sqrt(self.perp2)

    @property
    def phi(self):
        return np.arctan2(self.y, self.x)

    @property
    def theta(self):
        return np.arctan2(self.perp, self.z)


def tVector_list(ary_akw):
    vec_list = []
    for i in range(len(ary_akw)):
        vec_list.append(TVector3.from_akw(ary_akw[i]))
    return vec_list
