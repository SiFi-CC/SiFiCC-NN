import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.lines import Line2D
import uproot
import pickle as pkl

from SIFICCNN.utils import parent_directory
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from SIFICCNN.analysis import sigma_ee, read_resolution_file
from SIFICCNN.plot.utils import gaussian, auto_hist_fitting

# Path are set to load the PostTraining configuration for the paper
path_repo = parent_directory()
path_results_NN = path_repo + "/results/ECRNCluster_PostTraining/"
path_results_CB = path_repo + "/results/CutBased/"
path_targets = path_repo + "/NNPaper2023/"

####################################################################################################
# NN training plots
"""
# load history of trained network
with open(path_results + "ECRNCluster_PostTraining_classifier_history" + ".hst", 'rb') as f_hist:
    history_clas = pkl.load(f_hist)
with open(path_results + "ECRNCluster_PostTraining_regressionEnergy_history" + ".hst",
          'rb') as f_hist:
    history_regE = pkl.load(f_hist)
with open(path_results + "ECRNCluster_PostTraining_regressionPosition_history" + ".hst",
          'rb') as f_hist:
    history_regP = pkl.load(f_hist)

# Classifier Training Curve Plot
loss_clas = history_clas['loss']
val_loss_clas = history_clas['val_loss']
# mse = nn_classifier.history["accuracy"]
# val_mse = nn_classifier.history["val_accuracy"]
eff = history_clas["recall"]
val_eff = history_clas["val_recall"]
pur = history_clas["precision"]
val_pur = history_clas["val_precision"]

plt.rcParams.update({'font.size': 16})
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlim(0, 100)
ax1.set_ylim(0.44, 0.59)
ax1.set_xticks(np.arange(0, 100 + 1, 1), minor=True)
ax1.set_xticks(np.arange(0, 100 + 10, 10))
ax1.set_yticks(np.arange(0.44, 0.59 + 0.03, 0.03))
ax1.set_yticks(np.arange(0.44, 0.59 + 0.03 / 10, 0.03 / 10), minor=True)
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(axis="x", which="both", direction='in')
ax1.tick_params(axis="y", which="both", direction='in')
ax1.plot(loss_clas, label="Loss", linestyle='-', color="black")
ax1.plot(val_loss_clas, label="Validation", linestyle='--', color="black")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss (Binary-Crossentropy)")
ax1.grid(which='major', color='#CCCCCC', linewidth=0.8)
ax1.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
# second axis
ax2 = ax1.twinx()
ax2.set_ylim(0.4, 0.9)
ax2.set_yticks(np.arange(0.4, 0.9 + 0.1, 0.1), ["", "0.5", "0.6", "0.7", "0.8", " 0.9"])
ax2.set_yticks(np.arange(0.4, 0.9 + 0.1 / 10, 0.1 / 10), minor=True)
ax2.tick_params(axis="y", which="both", direction='in')
ax2.plot(eff, label="Efficiency", linestyle='-', color="red")
ax2.plot(val_eff, label="Validation", linestyle='--', color="red")
ax2.plot(pur, label="Purity", linestyle="-", color="blue")
ax2.plot(val_pur, label="Validation", linestyle="--", color="blue")
ax2.set_ylabel("Efficiency, Purity")
# Custom Legend
custom_lines = [Line2D([0], [0], color="black", lw=2),
                Line2D([0], [0], color="red", lw=2),
                Line2D([0], [0], color="blue", lw=2),
                Line2D([0], [0], color="grey", lw=2, linestyle="--")]
ax1.legend(custom_lines, ["Loss", "Efficiency", "Purity", "Validation"], loc="center right")
plt.tight_layout()
plt.savefig(path_targets + "train_clas.pdf")
plt.close()

# Regression Plot
loss_regE = history_regE['loss']
val_loss_regE = history_regE['val_loss']
loss_regP = history_regP['loss']
val_loss_regP = history_regP['val_loss']
plt.rcParams.update({'font.size': 16})
fig2, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xticks(np.arange(0, 100 + 10, 10))
ax1.set_xticks(np.arange(0, 100 + 10/10, 10/10), minor=True)
ax1.set_yticks(np.arange(0.5, 0.8 + 0.05, 0.05))
ax1.set_yticks(np.arange(0.5, 0.8 + 0.05 / 10, 0.05 / 10), minor=True)
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(axis="x", which="both", direction='in')
ax1.tick_params(axis="y", which="both", direction='in')
ax1.plot(loss_regE, label="Loss", linestyle='-', color="black")
ax1.plot(val_loss_regE, label="Validation", linestyle='--', color="black")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Mean-Absolute Error (Energy)")
ax1.grid(which='major', color='#CCCCCC', linewidth=0.8)
ax1.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
ax1.set_xlim(0, 100)
ax1.set_ylim(0.5, 0.8)
# second axis
ax2 = ax1.twinx()
ax2.set_ylim(3, 9)
ax2.set_yticks(np.arange(3, 9 + 1, 1), ["", "4", "5", "6", "7", "8", "9"])
ax2.set_yticks(np.arange(3, 9 + 1 / 10, 1 / 10), minor=True)
ax2.tick_params(axis="y", which="both", direction='in')
ax2.plot(loss_regP, label="Loss", linestyle='-', color="blue")
ax2.plot(val_loss_regP, label="Validation", linestyle='--', color="blue")
ax2.set_ylabel("Mean Absolute Error (Position)")
# Custom Legend
custom_lines = [Line2D([0], [0], color="black", lw=2),
                Line2D([0], [0], color="blue", lw=2),
                Line2D([0], [0], color="grey", lw=2, linestyle="--")]
ax1.legend(custom_lines, ["Loss (Energy)", "Loss (Position)", "Validation"], loc="upper right")
plt.tight_layout()
plt.savefig(path_targets + "train_regEP.pdf")
plt.close()
"""
####################################################################################################
# Generate electron energy fitting plots

# loading all necessary files for plot
# load root file containing the data points as well as the fitting parameter
file_fit_dp = uproot.open(
    path_results_NN + "ee_vs_ee_resolution_histOnly.root")
# file_fit_params = uproot.open(path_results + "allResolutions.root")

# load neural network predictions from the 0mm dataset
# (0mm cause it functions here as a test dataset)
dataset_name = "1to1_Cluster_BP0mm_2e10protons_simV3"
path_data_NN = "{}/{}/{}".format(path_results_NN, dataset_name, dataset_name)
path_data_CB = "{}/{}/{}".format(path_results_CB, dataset_name, dataset_name)

y_NNclas_pred = np.loadtxt(path_data_NN + "_clas_pred.txt", delimiter=",")
y_NNregE_pred = np.loadtxt(path_data_NN + "_regE_pred.txt", delimiter=",")
y_CBclas_pred = np.loadtxt(path_data_CB + "_clas_pred.txt", delimiter=",")
y_CBregE_pred = np.loadtxt(path_data_CB + "_regE_pred.txt", delimiter=",")
# true results are loaded vom CutBased as an example
# it has the same initial order as the NN dataset
y_clas_true = np.loadtxt(path_data_CB + "_clas_true.txt", delimiter=",")
y_regE_true = np.loadtxt(path_data_CB + "_regE_true.txt", delimiter=",")

# identifier for event selection
# Currently: Use all True-Positive events (identified and MC true)
theta = 0.5
idx_nn = np.zeros(shape=(len(y_clas_true, )))
idx_cb = np.zeros(shape=(len(y_clas_true, )))
for i in range(len(y_clas_true)):
    if y_clas_true[i] == 1:  # and y_NNclas_pred[i] > theta
        idx_nn[i] = 1
    if y_clas_true[i] == 1:  # and y_CBclas_pred[i] > theta
        idx_cb[i] = 1
idx_nn = idx_nn == 1
idx_cb = idx_cb == 1

# 1D Counts vs (E_reco-E_mc)
# set superscript for particle type and coordinate:
xlabel = r"$(E^{{pred}}_{{{0}}} - E^{{true}}_{{{0}}})/E^{{true}}_{{{0}}}$".format("e")
width = 0.05
bins = np.arange(-1.5, 2.0, width)
bins_center = bins[:-1] + (width / 2)
# histogram for gaussian fit
histNN, _ = np.histogram(
    y_NNregE_pred[idx_nn, 0] - y_regE_true[idx_nn, 0], bins=bins)
histCB, _ = np.histogram(
    y_CBregE_pred[idx_cb, 0] - y_regE_true[idx_cb, 0], bins=bins)
# remove bins from fit with zero entries as they break the fitting
poptNN, pcovNN, xNN, fxNN = auto_hist_fitting(
    f="gaussian", bins=bins_center, hist=histNN)
poptCB, pcovCB, xCB, fxCB = auto_hist_fitting(
    f="gaussian", bins=bins_center, hist=histCB)

fig, ax = plt.subplots(figsize=(8, 5))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(axis="x", which="both", direction='in')
ax.tick_params(axis="y", which="both", direction='in')
ax.set_xticks(bins, minor=True)
ax.yaxis.set_minor_locator(AutoMinorLocator(10))
ax.set_xlabel(xlabel, loc="right")
ax.set_ylabel("Counts", loc="top")
ax.set_xlim(min(bins), max(bins))
# NN hist
ax.hist((y_NNregE_pred[idx_nn, 0] - y_regE_true[idx_nn, 0]) / y_regE_true[idx_nn, 0], bins=bins,
        alpha=0.2, color="red")
ax.hist((y_NNregE_pred[idx_nn, 0] - y_regE_true[idx_nn, 0]) / y_regE_true[idx_nn, 0], bins=bins,
        histtype=u"step", color="red", linewidth=2.0,
        label="SiFiCC-NN" + "\n" +
              r"$\mu$ = {:.2f} $\pm$ {:.2f}".format(poptNN[0], np.sqrt(pcovNN[0, 0])) + "\n" +
              r"FWHM = {:.2f} $\pm$ {:.2f}".format(poptNN[1] * 2.355,
                                                   np.sqrt(pcovNN[1, 1]) * 2.355))

# CB hist
ax.hist((y_CBregE_pred[idx_cb, 0] - y_regE_true[idx_cb, 0]) / y_regE_true[idx_cb, 0], bins=bins,
        alpha=0.2, color="black")
ax.hist((y_CBregE_pred[idx_cb, 0] - y_regE_true[idx_cb, 0]) / y_regE_true[idx_cb, 0], bins=bins,
        histtype=u"step", color="black", linewidth=2.0,
        label="Classical Approach" + "\n" +
              r"$\mu$ = {:.2f} $\pm$ {:.2f}".format(poptCB[0], np.sqrt(pcovCB[0, 0])) + "\n" +
              r"FWHM = {:.2f} $\pm$ {:.2f}".format(poptCB[1] * 2.355,
                                                   np.sqrt(pcovCB[1, 1]) * 2.355))
"""
ax.plot(xNN, fxNN, color="black", linestyle="--",
        label=r"$\mu$ = {{{}}} $\pm$ {{}}\nFWHM = {{{}}} $\pm$ {{{}}}".format(
            poptNN[0], np.sqrt(pcovNN[0, 0]), poptNN[1] * 2.355, np.sqrt(pcovNN[1, 1]) * 2.355))
ax.plot(xCB, fxCB, color="red", linestyle="--",
        label=r"$\mu$ = {{{}}} $\pm$ {{{}}}\nFWHM = {{{}}} $\pm$ {{{}}}".format(
            poptCB[0], np.sqrt(pcovCB[0, 0]), poptCB[1] * 2.355, np.sqrt(pcovCB[1, 1]) * 2.355))
"""
ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
ax.legend(loc="upper right", prop={'size': 12})
plt.tight_layout()
plt.savefig(path_targets + "1d_ee_residual.png")
plt.close()

# 2D Counts vs (E_reco-E_mc) vs E_mc
xlabel = r"$E^{{true}}_{{{0}}}$ [MeV]".format("e")
ylabel = r"$E^({{pred}}_{{{0}}} - E^{{true}}_{{{0}}})/E^{{true}}_{{{0}}}$ [MeV]".format(
    "e")
# plot settings
width = 0.1
bins_x = np.arange(0.0, 10.0, width)
bins_y = np.arange(-2.0, 2.0, width)
fig2, ax = plt.subplots(figsize=(8, 5))
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(axis="x", which="both", direction='in')
ax.tick_params(axis="y", which="both", direction='in')
ax.set_xlabel(xlabel, loc="right")
ax.set_ylabel(ylabel, loc="top")
ax.set_xlim(min(bins_x), max(bins_x))
ax.set_ylim(min(bins_y), max(bins_y))
h2d = ax.hist2d(x=y_regE_true[idx_nn, 0],
                y=(y_NNregE_pred[idx_cb, 0] -
                   y_regE_true[idx_nn, 0]) / y_regE_true[idx_nn, 0],
                bins=[bins_x, bins_y], norm=LogNorm())
ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.colorbar(h2d[3])
plt.tight_layout()
plt.savefig(path_targets + "2d_ee_residual.png")
plt.close()

# Parameter-fitting plot
# data points
x = file_fit_dp["ee_vs_ee_resolution"].axis().centers()
x_fine = np.linspace(0, 10, 1000)[1:]
y = file_fit_dp["ee_vs_ee_resolution"].values()
y_err = file_fit_dp["ee_vs_ee_resolution"].errors()
params = read_resolution_file(path_results_NN + "allResolutions.root")

fig3, ax = plt.subplots(figsize=(8, 5))
ax.set_xlabel("$E_{e}$ [MeV]")
ax.set_ylabel(r"$\sigma_{E_e}/E_{e}$")
ax.set_xticks(np.arange(0, 10 + 1, 1))
ax.set_xticks(np.arange(0, 10 + 1 / 10, 1 / 10), minor=True)
ax.set_yticks(np.arange(0.0, 0.5 + 0.1, 0.1))
ax.set_yticks(np.arange(0.0, 0.5 + 0.1 / 10, 0.1 / 10), minor=True)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(axis="x", which="both", direction='in')
ax.tick_params(axis="y", which="both", direction='in')
ax.set_xlim(0, 10)
ax.set_ylim(0, 0.5)
ax.errorbar(x, y, y_err, fmt=".", color="blue")
ax.plot(x_fine, sigma_ee(x_fine, *params["ee"]) / x_fine, color="red")
ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig(path_targets + "ee_resolution.png")
plt.close()
