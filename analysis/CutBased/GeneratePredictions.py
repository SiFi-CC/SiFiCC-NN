import numpy as np
import os

from SIFICCNN.data.roots import RootSimulation
from SIFICCNN.utils import parent_directory


def main(run_name):
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
    path_root = path + "/root_files/"
    path_results = path_main + "/results/" + run_name + "/"

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM, DATASET_m5MM]:
        if not os.path.isdir(path_results + "/" + file + "/"):
            os.mkdir(path_results + "/" + file + "/")

    gen_results(path_results + DATASET_0MM + "/" + DATASET_0MM,
                path_root + DATASET_0MM + ".root")


def gen_results(target,
                file_name):
    # open root file
    root_simulation = RootSimulation(file_name)

    # generate empty arrays for prediction and classification
    y_clas_pred = np.zeros(shape=(root_simulation.events_entries,), dtype=int)
    y_clas_true = np.zeros(shape=(root_simulation.events_entries,), dtype=int)
    y_regE_pred = np.zeros(shape=(root_simulation.events_entries, 2))
    y_regE_true = np.zeros(shape=(root_simulation.events_entries, 2))
    y_regP_pred = np.zeros(shape=(root_simulation.events_entries, 6))
    y_regP_true = np.zeros(shape=(root_simulation.events_entries, 6))

    for i, event in enumerate(root_simulation.iterate_events(n=None)):
        # get
        ee, ep = event.RecoCluster.get_reco_energy()
        tv3e, tv3p = event.RecoCluster.get_reco_position()
        target_ee, target_ep = event.get_target_energy()
        target_tv3e, target_tv3p = event.get_target_position()

        # fill
        y_clas_pred[i] = int((event.RecoCluster.Identified != 0) * 1)
        y_clas_true[i] = event.get_distcompton_tag() * 1

        y_regE_pred[i, :] = [ee, ep]
        y_regE_true[i, :] = [target_ee, target_ep]
        y_regP_pred[i, :] = [tv3e.z, -tv3e.y, tv3e.x, tv3p.z, -tv3p.y, tv3p.x]
        y_regP_true[i, :] = [target_tv3e.z, -target_tv3e.y, target_tv3e.x,
                             target_tv3p.z, -target_tv3p.y, target_tv3p.x]

    np.savetxt(fname=target + "_clas_pred.txt",
               X=y_clas_pred,
               delimiter=",",
               newline="\n")
    np.savetxt(fname=target + "_clas_true.txt",
               X=y_clas_true,
               delimiter=",",
               newline="\n")

    np.savetxt(fname=target + "_regE_pred.txt",
               X=y_regE_pred,
               delimiter=",",
               newline="\n")
    np.savetxt(fname=target + "_regE_true.txt",
               X=y_regE_true,
               delimiter=",",
               newline="\n")

    np.savetxt(fname=target + "_regP_pred.txt",
               X=y_regP_pred,
               delimiter=",",
               newline="\n")
    np.savetxt(fname=target + "_regP_true.txt",
               X=y_regP_true,
               delimiter=",",
               newline="\n")


if __name__ == "__main__":
    main(run_name="CutBased")
