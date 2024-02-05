import numpy as np
import os
import json
import tensorflow as tf

from SIFICCNN.utils import parent_directory
from SIFICCNN.ComptonCamera6 import exportCC6

from SIFICCNN.utils.plotter import plot_position_error_vs_energy

from SIFICCNN.analysis import sigma_ee, \
    sigma_ep, sigma_ex, sigma_ey, sigma_ez, sigma_px, sigma_py, sigma_pz


def main(run_name,
         exp_name,
         threshold,
         fponly=False):
    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "1to1_Cluster_CONT_2e10protons_simV3"
    DATASET_0MM = "1to1_Cluster_BP0mm_2e10protons_simV3"
    DATASET_5MM = "1to1_Cluster_BP5mm_4e9protons_simV3"
    DATASET_m5MM = "1to1_Cluster_BPm5mm_4e9protons_simV3"

    # go backwards in directory tree until the main repo directory is matched
    path = parent_directory()
    path_main = path
    path_results = path_main + "/results/" + run_name + "/"
    path_datasets = path_main + "/datasets/"

    for file in [DATASET_0MM, DATASET_5MM, DATASET_m5MM]:
        os.chdir(path_results + file + "/")
        # gather all network predictions
        y_score_pred = np.loadtxt(file + "_clas_pred.txt", delimiter=",")
        y_regE_pred = np.loadtxt(file + "_regE_pred.txt", delimiter=",")
        y_regP_pred = np.loadtxt(file + "_regP_pred.txt", delimiter=",")

        # define positive classified events
        idx_pos = np.zeros(shape=(len(y_score_pred, )))
        for i in range(len(idx_pos)):
            if y_score_pred[i] > threshold:
                idx_pos[i] = 1
        idx_pos = idx_pos == 1

        # generate additional plots
        # Some plots are generated here as they need both energy and position
        y_score_true = np.loadtxt(file + "_clas_true.txt", delimiter=",")
        y_regE_true = np.loadtxt(file + "_regE_true.txt", delimiter=",")
        y_regP_true = np.loadtxt(file + "_regP_true.txt", delimiter=",")

        # plotting of more convoluted histograms
        # TODO: THIS NEEDS TO BE PLACED SOMEWHERE DIFFERENT
        plot_position_error_vs_energy(y_regP_pred[y_score_true == 1],
                                      y_regP_true[y_score_true == 1],
                                      np.sum(y_regE_true[y_score_true == 1], axis=1),
                                      "error_regression")

        # calculate uncertainties of all quantities
        # TODO: THIS IS RIGHT NOW HARDCODED FOR BP0MM; CHANGES COMMING SOONISH
        ee_err = sigma_ee(y_regE_pred[idx_pos, 0], 9.087e-2, -6.904e-2, 6.41e-2)
        ep_err = sigma_ep(y_regE_pred[idx_pos, 1], 0.05636, -0.1248, 0.4622)
        ey_err = sigma_ey(y_regE_pred[idx_pos, 0], 3.089e0, 5.959e0, -7.568e-2)
        py_err = sigma_py(y_regE_pred[idx_pos, 1], -3.181e-1, 1.519e1, -3.651e0)
        ex_err = sigma_ex(y_regE_pred[idx_pos, 0], 0.3620, -0.0663, 0.1115, -0.0178, 0.00083)
        ez_err = sigma_ez(y_regE_pred[idx_pos, 0], 0.3637, 0.00376, 0.0693, -0.0139, 0.000803)
        px_err = sigma_px(y_regE_pred[idx_pos, 1], 0.4312, -0.1020, 0.0588, -0.00675, 0.000267)
        pz_err = sigma_pz(y_regE_pred[idx_pos, 1], 0.4352, -0.05364, 0.0420, -0.004795, 0.00194)

        # export to root file compatible with CC6 image reconstruction
        exportCC6(ee=y_regE_pred[idx_pos, 0],
                  ep=y_regE_pred[idx_pos, 1],
                  ex=y_regP_pred[idx_pos, 0],
                  ey=y_regP_pred[idx_pos, 1],
                  ez=y_regP_pred[idx_pos, 2],
                  px=y_regP_pred[idx_pos, 3],
                  py=y_regP_pred[idx_pos, 4],
                  pz=y_regP_pred[idx_pos, 5],
                  ee_err=ee_err,
                  ep_err=ep_err,
                  ex_err=ex_err,
                  ey_err=ey_err,
                  ez_err=ez_err,
                  px_err=px_err,
                  py_err=py_err,
                  pz_err=pz_err,
                  filename="CC6_{}_{}".format(exp_name, file),
                  verbose=1,
                  veto=True)

    if fponly:
        os.chdir(path_results + DATASET_0MM + "/")
        # gather all network predictions
        y_score_pred = np.loadtxt(DATASET_0MM + "_clas_pred.txt", delimiter=",")
        y_score_true = np.loadtxt(DATASET_0MM + "_clas_true.txt", delimiter=",")
        y_regE_pred = np.loadtxt(DATASET_0MM + "_regE_pred.txt", delimiter=",")
        y_regP_pred = np.loadtxt(DATASET_0MM + "_regP_pred.txt", delimiter=",")

        # define positive classified events
        idx_pos = np.zeros(shape=(len(y_score_pred, )))
        for i in range(len(idx_pos)):
            if y_score_true[i] == 0 and y_score_pred[i] > threshold:
                idx_pos[i] = 1
        idx_pos = idx_pos == 1

        # export to root file compatible with CC6 image reconstruction
        exportCC6(ee=y_regE_pred[idx_pos, 0],
                  ep=y_regE_pred[idx_pos, 1],
                  ex=y_regP_pred[idx_pos, 0],
                  ey=y_regP_pred[idx_pos, 1],
                  ez=y_regP_pred[idx_pos, 2],
                  px=y_regP_pred[idx_pos, 3],
                  py=y_regP_pred[idx_pos, 4],
                  pz=y_regP_pred[idx_pos, 5],
                  filename="CC6_FPONLY_{}_{}".format(exp_name, file),
                  verbose=1,
                  veto=True)


if __name__ == "__main__":
    run_name = "ECRNCluster_PostTraining"
    exp_name = "ECRNCluster_PostTraining"
    threshold = 0.5

    main(run_name=run_name,
         exp_name=exp_name,
         threshold=threshold,
         fponly=False)
