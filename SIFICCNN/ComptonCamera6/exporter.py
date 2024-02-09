import os
import numpy as np
import uproot

from .veto import check_DAC, check_compton_arc, check_compton_kinematics, check_valid_prediction


def correct_input_length(entry, l):
    if isinstance(entry, float) or isinstance(entry, int):
        entry = np.ones(shape=(l,)) * entry
    return entry


def exportCC6(filename,
              ee,
              ep,
              ex,
              ey,
              ez,
              px,
              py,
              pz,
              ee_err=None,
              ep_err=None,
              ex_err=None,
              ey_err=None,
              ez_err=None,
              px_err=None,
              py_err=None,
              pz_err=None,
              veto=True,
              path="",
              verbose=0):
    # test each input of statistical errors on their type
    # If int/float, they are extended to the needed array length
    # If they are not given, each error entry is filled with zero
    l = len(ee)
    ee_err = np.zeros(shape=(l,)) if ee_err is None else correct_input_length(ee_err, l)
    ep_err = np.zeros(shape=(l,)) if ep_err is None else correct_input_length(ep_err, l)
    ex_err = np.zeros(shape=(l,)) if ex_err is None else correct_input_length(ex_err, l)
    ey_err = np.zeros(shape=(l,)) if ey_err is None else correct_input_length(ey_err, l)
    ez_err = np.zeros(shape=(l,)) if ez_err is None else correct_input_length(ez_err, l)
    px_err = np.zeros(shape=(l,)) if px_err is None else correct_input_length(px_err, l)
    py_err = np.zeros(shape=(l,)) if py_err is None else correct_input_length(py_err, l)
    pz_err = np.zeros(shape=(l,)) if pz_err is None else correct_input_length(pz_err, l)

    # Define verbose statistic on event rejection
    identified = np.ones(shape=(l,))
    reject_valid = 0
    reject_arc = 0
    reject_kinematics = 0
    reject_DAC = 0

    if veto:
        for i in range(l):
            if not check_valid_prediction(ee[i], ep[i],
                                          ex[i], ey[i], ez[i],
                                          px[i], py[i], pz[i]):
                identified[i] = 0
                reject_valid += 1
                continue

            if not check_compton_arc(ee[i], ep[i]):
                identified[i] = 0
                reject_arc += 1
                continue

            if not check_compton_kinematics(ee[i], ep[i], ee=ee_err[i], ep=ep_err[i], compton=True):
                identified[i] = 0
                reject_kinematics += 1
                continue

            if not check_DAC(ee[i], ep[i],
                             ex[i], ey[i], ez[i],
                             px[i], py[i], pz[i],
                             20, inverse=False):
                identified[i] = 0
                reject_DAC += 1
                continue

    # print MLEM export statistics
    if verbose == 1:
        print("\n# CC6 export statistics: ")
        print("Number of total events: ", l)
        print("Number of events after cuts: ", np.sum(identified))
        print("Number of cut events: ", l - np.sum(identified))
        print("    - Valid prediction: ", reject_valid)
        print("    - Compton arc: ", reject_arc)
        print("    - Compton kinematics: ", reject_kinematics)
        print("    - Beam Origin: ", reject_DAC)

    # required fields for the root file
    entries = np.sum(identified)
    zeros = np.zeros(shape=(int(entries),))
    eventnumbers = np.arange(entries)

    # apply selection of post filter events
    identified = identified == 1

    # process additional quantities
    e0 = ee + ep
    arc = np.arccos(1 - 0.511 * (1 / ep - 1 / e0))

    e0_unc = np.array([np.sqrt(ee_err[i] ** 2 + ep_err[i] ** 2) for i in range(len(ee_err))])
    p_unc_x = np.array([np.sqrt(py_err[i] ** 2 + ey_err[i] ** 2) for i in range(len(py_err))])
    p_unc_y = np.array([np.sqrt(pz_err[i] ** 2 + ez_err[i] ** 2) for i in range(len(pz_err))])
    p_unc_z = np.array([np.sqrt(px_err[i] ** 2 + ex_err[i] ** 2) for i in range(len(px_err))])
    arc_unc = np.array([0.511/np.sqrt(1-np.cos(arc[i])**2 * np.sqrt((ep_err[i]/(ep[i]**2))**2 + (ee_err[i]/(e0[i]**2))**2)) for i in range(len(ep))])

    # create root file
    if path == "":
        path=os.getcwd() + "/"
    file_name = path + filename + ".root"
    file = uproot.recreate(file_name, compression=None)

    print(np.sum(identified), "events exported")
    print("file created at: ", file_name)

    # filling the branch
    # ROOT FILES ARE FILLED IN LUEBECK COORDINATE SYSTEM
        # (x_1,y_1,z_1)   : reconstructed electron position in scatterer in mm
        # (x_2,y_2,z_2)   : reconstructed photon position in absorber in mm
        # E1              : reconstructed electron energy in scatterer in MeV
        # E2              : reconstructed photon energy in absorber in MeV
        # E0Calc          : sum of E1 and E2
        # (v_x,v_y,v_z)   : cone apex (same as (x_1,y_1,z_1))
        # (p_x,p_y,p_z)   : cone axis (calculated as difference of (x_2,y_2,z_2) and x_1,y_1,z_1))
        # arc             : cone angle in rad (calulated from E1 and E2)
    file['ConeList'] = {'GlobalEventNumber': eventnumbers,
                        'ClassID': zeros,
                        'EventType': zeros,
                        'EnergyBinID': zeros,
                        'x_1': ey[identified],
                        'y_1': -ez[identified],
                        'z_1': -ex[identified],
                        'x_1_unc': ey_err[identified],
                        'y_1_unc': ez_err[identified],
                        'z_1_unc': ex_err[identified],
                        'x_2': py[identified],
                        'y_2': -pz[identified],
                        'z_2': -px[identified],
                        'x_2_unc': py_err[identified],
                        'y_2_unc': pz_err[identified],
                        'z_2_unc': px_err[identified],
                        'x_3': zeros,
                        'y_3': zeros,
                        'z_3': zeros,
                        'x_3_unc': zeros,
                        'y_3_unc': zeros,
                        'z_3_unc': zeros,
                        'E1': ee[identified],
                        'E1_unc': ee_err[identified],
                        'E2': ep[identified],
                        'E2_unc': ep_err[identified],
                        'E3': zeros,
                        'E3_unc': zeros,
                        'E0Calc': e0[identified],
                        'E0Calc_unc': e0_unc[identified],
                        'v_x': ey[identified],
                        'v_y': -ez[identified],
                        'v_z': -ex[identified],
                        'v_unc_x': ey_err[identified],
                        'v_unc_y': ez_err[identified],
                        'v_unc_z': ex_err[identified],
                        'p_x': py[identified] - ey[identified],
                        'p_y': -pz[identified] + ez[identified],
                        'p_z': -px[identified] + ex[identified],
                        'p_unc_x': p_unc_x[identified],
                        'p_unc_y': p_unc_y[identified],
                        'p_unc_z': p_unc_z[identified],
                        'arc': arc[identified],
                        'arc_unc': arc_unc[identified]}

    # filling the branch
    file['TreeStat'] = {'StartEvent': [0],
                        'StopEvent': [entries],
                        'TotalSimNev': [0]}

    # closing the root file
    file.close()
