import numpy as np


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
        return TVector3(self.x * other, self.y * other, self.z * other)

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

    # vector properties
    @property
    def mag(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


def tVector_list(ary_akw):
    vec_list = []
    for i in range(len(ary_akw)):
        vec_list.append(TVector3.from_akw(ary_akw[i]))
    return vec_list
