import numpy as np
import uproot


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


####################################################################################################

def read_resolution_file(root_file):
    """
    This method read a root file containing parameter fits for resolutions
    """
    # open root file
    root_file = uproot.open(root_file)

    # read fitting parameter from root files:
    # Trees containing each fit are denoted as "XXX-resolution_func", a root TF1-object
    # Each TF1-object do not have any behaviour but the content is readable, the property
    # ".all_members" calls the content. The entry "fFormula" contains a TFormula-object with all
    # information regarding the parameter fit.
    param_dict = {"ee": root_file["ee_vs_ee_resolution_func"].all_members["fFormula"].all_members[
                      "fClingParameters"],
                  "ex": root_file["ex_vs_ee_resolution_func"].all_members["fFormula"].all_members[
                      "fClingParameters"],
                  "ey": root_file["ey_vs_ee_resolution_func"].all_members["fFormula"].all_members[
                      "fClingParameters"],
                  "ez": root_file["ez_vs_ee_resolution_func"].all_members["fFormula"].all_members[
                      "fClingParameters"],
                  "ep": root_file["ep_vs_ep_resolution_func"].all_members["fFormula"].all_members[
                      "fClingParameters"],
                  "px": root_file["px_vs_ep_resolution_func"].all_members["fFormula"].all_members[
                      "fClingParameters"],
                  "py": root_file["py_vs_ep_resolution_func"].all_members["fFormula"].all_members[
                      "fClingParameters"],
                  "pz": root_file["pz_vs_ep_resolution_func"].all_members["fFormula"].all_members[
                      "fClingParameters"]}

    return param_dict
