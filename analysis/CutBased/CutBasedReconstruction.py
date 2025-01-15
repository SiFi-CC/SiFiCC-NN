import numpy as np
import os
import pickle as pkl
import json
import argparse

from SIFICCNN.utils import parent_directory
from SIFICCNN.analysis import fastROCAUC, print_classifier_summary, write_classifier_summary

from SIFICCNN.plot import plot_1dhist_energy_residual, \
    plot_1dhist_energy_residual_relative, \
    plot_1dhist_position_residual, \
    plot_2dhist_energy_residual_vs_true, \
    plot_2dhist_energy_residual_relative_vs_true, \
    plot_2dhist_position_residual_vs_true


def main(run_name="CutBased"):
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

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM, DATASET_m5MM]:
        if not os.path.isdir(path_results + "/" + file + "/"):
            os.mkdir(path_results + "/" + file + "/")

    for file in [DATASET_0MM, DATASET_5MM, DATASET_m5MM]:
        # eval_classifier(run_name, file, path_results)
        # eval_regressionEnergy(run_name, file, path_results)
        eval_regressionPosition(run_name, file, path_results)


def eval_classifier(run_name, dataset_name, path):
    # change working directory
    os.chdir(path + dataset_name + "/")

    # load classification results
    y_pred = np.loadtxt(dataset_name + "_clas_pred.txt", delimiter=",")
    y_true = np.loadtxt(dataset_name + "_clas_true.txt", delimiter=",")

    # evaluation and plot
    # print_classifier_summary(y_pred, y_true, run_name=run_name)
    write_classifier_summary(y_pred, y_true, run_name=run_name)


def eval_regressionEnergy(run_name, dataset_name, path):
    # change working directory
    os.chdir(path + dataset_name + "/")

    y_label = np.loadtxt(dataset_name + "_clas_true.txt", delimiter=",") == 1
    y_pred = np.loadtxt(dataset_name + "_regE_pred.txt", delimiter=",")
    y_true = np.loadtxt(dataset_name + "_regE_true.txt", delimiter=",")

    plot_1dhist_energy_residual(y_pred=y_pred[y_label, 0],
                                y_true=y_true[y_label, 0],
                                particle="e",
                                file_name="1dhist_energy_electron_residual.png")
    plot_1dhist_energy_residual_relative(y_pred=y_pred[y_label, 0],
                                         y_true=y_true[y_label, 0],
                                         particle="e",
                                         file_name="1dhist_energy_electron_residual_relative.png")
    plot_2dhist_energy_residual_vs_true(y_pred=y_pred[y_label, 0],
                                        y_true=y_true[y_label, 0],
                                        particle="e",
                                        file_name="2dhist_energy_electron_residual_vs_true.png")
    plot_2dhist_energy_residual_relative_vs_true(y_pred=y_pred[y_label, 0],
                                                 y_true=y_true[y_label, 0],
                                                 particle="e",
                                                 file_name="2dhist_energy_electron_residual_relative_vs_true.png")

    plot_1dhist_energy_residual(y_pred=y_pred[y_label, 1],
                                y_true=y_true[y_label, 1],
                                particle="\gamma",
                                f="gaussian_gaussian",
                                file_name="1dhist_energy_gamma_residual.png")
    plot_1dhist_energy_residual_relative(y_pred=y_pred[y_label, 1],
                                         y_true=y_true[y_label, 1],
                                         particle="\gamma",
                                         f="gaussian_gaussian",
                                         file_name="1dhist_energy_gamma_residual_relative.png")


def eval_regressionPosition(run_name, dataset_name, path):
    # change working directory
    os.chdir(path + dataset_name + "/")

    labels = np.loadtxt(dataset_name + "_clas_true.txt", delimiter=",") == 1
    y_pred = np.loadtxt(dataset_name + "_regP_pred.txt", delimiter=",")
    y_true = np.loadtxt(dataset_name + "_regP_true.txt", delimiter=",")

    plot_1dhist_position_residual(y_pred=y_pred[labels, 0],
                                  y_true=y_true[labels, 0],
                                  particle="e",
                                  coordinate="x",
                                  file_name="1dhist_electron_position_{}_residual.png".format("x"))
    plot_1dhist_position_residual(y_pred=y_pred[labels, 3],
                                  y_true=y_true[labels, 3],
                                  particle="\gamma",
                                  coordinate="x",
                                  file_name="1dhist_gamma_position_{}_residual.png".format("x"))

    plot_1dhist_position_residual(y_pred=y_pred[labels, 1],
                                  y_true=y_true[labels, 1],
                                  particle="e",
                                  coordinate="y",
                                  f="lorentzian",
                                  file_name="1dhist_electron_position_{}_residual.png".format("y"))
    plot_1dhist_position_residual(y_pred=y_pred[labels, 4],
                                  y_true=y_true[labels, 4],
                                  particle="\gamma",
                                  coordinate="y",
                                  f="lorentzian",
                                  file_name="1dhist_gamma_position_{}_residual.png".format("y"))

    plot_1dhist_position_residual(y_pred=y_pred[labels, 2],
                                  y_true=y_true[labels, 2],
                                  particle="e",
                                  coordinate="z",
                                  file_name="1dhist_electron_position_{}_residual.png".format("z"))
    plot_1dhist_position_residual(y_pred=y_pred[labels, 5],
                                  y_true=y_true[labels, 5],
                                  particle="\gamma",
                                  coordinate="z",
                                  file_name="1dhist_gamma_position_{}_residual.png".format("z"))

    for i, r in enumerate(["x", "y", "z"]):
        plot_2dhist_position_residual_vs_true(y_pred=y_pred[labels, i],
                                              y_true=y_true[labels, i],
                                              particle="e",
                                              coordinate=r,
                                              file_name="2dhist_position_electron_{}_residual_vs_true.png".format(
                                                  r))
        plot_2dhist_position_residual_vs_true(y_pred=y_pred[labels, i + 3],
                                              y_true=y_true[labels, i + 3],
                                              particle="\gamma",
                                              coordinate=r,
                                              file_name="2dhist_position_gamma_{}_residual_vs_true.png".format(
                                                  r))


if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(
        description='Trainings script ECRNCluster model')
    parser.add_argument("--name", type=str, help="Run name")
    args = parser.parse_args()
    # base settings if no parameters are given
    # can also be used to execute this script without console parameter
    base_run_name = "CutBased"
    # this bunch is to set standard configuration if argument parser is not configured
    # looks ugly but works
    run_name = args.name if args.name is not None else base_run_name
    main(run_name=run_name)
