import numpy as np


def sigma_ee(ee, a, b, c):
    """
    FILL THIS ONCE MORE DETAILS ARE GIVEN!
    """
    return ee * (a + b / np.sqrt(ee) + c / ee)


def sigma_ep(ep, a, b, c):
    """
    FILL THIS ONCE MORE DETAILS ARE GIVEN!
    """
    return sigma_ee(ep, a, b, c)


def sigma_ey(ee, a, b, c):
    """
    FILL THIS ONCE MORE DETAILS ARE GIVEN!
    """
    return a + b / np.sqrt(ee) + c / ee


def sigma_py(ep, a, b, c):
    """
    FILL THIS ONCE MORE DETAILS ARE GIVEN!
    """
    return sigma_ey(ep, a, b, c)


def sigma_ex(ee, p0, p1, p2, p3, p4):
    """
    FILL THIS ONCE MORE DETAILS ARE GIVEN!
    """

    return p0 + p1 * ee + p2 * ee ** 2 + p3 * ee ** 3 + p4 * ee ** 4


def sigma_ez(ee, p0, p1, p2, p3, p4):
    """
    FILL THIS ONCE MORE DETAILS ARE GIVEN!
    """
    return sigma_ex(ee, p0, p1, p2, p3, p4)


def sigma_px(ep, p0, p1, p2, p3, p4):
    """
    FILL THIS ONCE MORE DETAILS ARE GIVEN!
    """
    return sigma_ex(ep, p0, p1, p2, p3, p4)


def sigma_pz(ep, p0, p1, p2, p3, p4):
    """
    FILL THIS ONCE MORE DETAILS ARE GIVEN!
    """
    return sigma_ex(ep, p0, p1, p2, p3, p4)
