import numpy as np
import os
import json
import tensorflow as tf

from SIFICCNN.utils import parent_directory
from SIFICCNN.ComptonCamera6 import exportCC6


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

    for file in [DATASET_0MM]:
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

        # export to root file compatible with CC6 image reconstruction
        exportCC6(ary_e=y_regE_pred[idx_pos, 0],
                  ary_p=y_regE_pred[idx_pos, 1],
                  ary_ex=y_regP_pred[idx_pos, 0],
                  ary_ey=y_regP_pred[idx_pos, 1],
                  ary_ez=y_regP_pred[idx_pos, 2],
                  ary_px=y_regP_pred[idx_pos, 3],
                  ary_py=y_regP_pred[idx_pos, 4],
                  ary_pz=y_regP_pred[idx_pos, 5],
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
        exportCC6(ary_e=y_regE_pred[idx_pos, 0],
                  ary_p=y_regE_pred[idx_pos, 1],
                  ary_ex=y_regP_pred[idx_pos, 0],
                  ary_ey=y_regP_pred[idx_pos, 1],
                  ary_ez=y_regP_pred[idx_pos, 2],
                  ary_px=y_regP_pred[idx_pos, 3],
                  ary_py=y_regP_pred[idx_pos, 4],
                  ary_pz=y_regP_pred[idx_pos, 5],
                  filename="CC6_FPONLY_{}_{}".format(exp_name, file),
                  verbose=1,
                  veto=True)


if __name__ == "__main__":
    run_name = "ECRNCluster_unnamed"
    exp_name = "ECRNCluster_unnamed"
    threshold = 0.5

    main(run_name=run_name,
         exp_name=exp_name,
         threshold=threshold,
         fponly=False)
