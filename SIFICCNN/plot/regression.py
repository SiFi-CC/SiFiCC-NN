import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator

from .utils import auto_hist_fitting

# global settings
plt.rcParams.update({'font.size': 14})


# ##################################################################################################
# Energy regression

def plot_1dhist_energy_residual(y_pred,
                                y_true,
                                file_name,
                                f="gaussian",
                                particle="e"):
    xlabel = r"$E^{{pred}}_{{{0}}} - E^{{true}}_{{{0}}}$ [MeV]".format(particle)

    # plot settings
    width = 0.1
    bins = np.arange(-2.0, 2.0, width)
    bins_center = bins[:-1] + (width / 2)
    # histogram for gaussian fit
    hist, _ = np.histogram(y_pred - y_true, bins=bins)
    popt, pcov, x, fx = auto_hist_fitting(f=f, bins=bins_center, hist=hist)
    fit_label = ""
    for i in range(len(popt)):
        fit_label += r"$p{}$ = {:.2f} $\pm$ {:.2f}".format(i, popt[i], np.sqrt(pcov[i, i])) + "\n"
    # plot
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
    ax.hist(y_pred - y_true, bins=bins, histtype=u"step", color="black")
    ax.errorbar(bins_center, hist, np.sqrt(hist), fmt=".", color="black")
    ax.plot(x, fx, color="red", label=fit_label)
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    ax.legend(loc="upper left", prop={'size': 12})
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_1dhist_energy_residual_relative(y_pred,
                                         y_true,
                                         file_name,
                                         f="gaussian",
                                         particle="electron"):
    # set superscript for particle type:
    xlabel = r"$(E^{{pred}}_{{{0}}} - E^{{true}}_{{{0}}})/E^{{true}}_{{{0}}}$".format(particle)

    # plot settings
    width = 0.1
    bins = np.arange(-2.0, 2.0, width)
    bins_center = bins[:-1] + (width / 2)
    # histogram for gaussian fit
    hist, _ = np.histogram((y_pred - y_true) / y_true, bins=bins)
    popt, pcov, x, fx = auto_hist_fitting(f=f, bins=bins_center, hist=hist)
    x = np.linspace(min(bins), max(bins), 1000)
    fit_label = ""
    for i in range(len(popt)):
        fit_label += r"$p{}$ = {:.2f} $\pm$ {:.2f}".format(i, popt[i], np.sqrt(pcov[i, i])) + "\n"
    # plot
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
    ax.hist((y_pred - y_true) / y_true, bins=bins, histtype=u"step", color="black")
    ax.errorbar(bins_center, hist, np.sqrt(hist), fmt=".", color="black")
    ax.plot(x, fx, color="red", label=fit_label)
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    ax.legend(loc="upper left", prop={'size': 12})
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_2dhist_energy_residual_vs_true(y_pred,
                                        y_true,
                                        file_name,
                                        particle="e"):
    xlabel = r"$E^{{true}}_{{{0}}}$ [MeV]".format(particle)
    ylabel = r"$E^{{pred}}_{{{0}}} - E^{{true}}_{{{0}}}$ [MeV]".format(particle)

    # plot settings
    width = 0.1
    bins_x = np.arange(0.0, 10.0, width)
    bins_y = np.arange(-2.0, 2.0, width)

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis="x", which="both", direction='in')
    ax.tick_params(axis="y", which="both", direction='in')
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top")
    ax.set_xlim(min(bins_x), max(bins_x))
    ax.set_ylim(min(bins_y), max(bins_y))
    h2d = ax.hist2d(x=y_true, y=y_pred - y_true, bins=[bins_x, bins_y], norm=LogNorm())
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.colorbar(h2d[3])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_2dhist_energy_residual_relative_vs_true(y_pred,
                                                 y_true,
                                                 file_name,
                                                 particle="e"):
    xlabel = r"$E^{{true}}_{{{0}}}$ [MeV]".format(particle)
    ylabel = r"$(E^{{pred}}_{{{0}}} - E^{{true}}_{{{0}}})/E^{{true}}_{{{0}}}$".format(particle)

    # plot settings
    width = 0.1
    bins_x = np.arange(0.0, 10.0, width)
    bins_y = np.arange(-2.0, 2.0, width)

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis="x", which="both", direction='in')
    ax.tick_params(axis="y", which="both", direction='in')
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top")
    ax.set_xlim(min(bins_x), max(bins_x))
    ax.set_ylim(min(bins_y), max(bins_y))
    h2d = ax.hist2d(x=y_true, y=(y_pred - y_true) / y_true, bins=[bins_x, bins_y], norm=LogNorm())
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.colorbar(h2d[3])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


# ##################################################################################################
# Position regression


def plot_1dhist_position_residual(y_pred,
                                  y_true,
                                  file_name,
                                  f="gaussian",
                                  coordinate="x",
                                  particle="electron"):
    # set superscript for particle type and coordinate:
    xlabel = r"${{{0}}}^{{pred}}_{{{1}}} - {{{0}}}^{{true}}_{{{1}}}$ [mm]".format(particle,
                                                                                  coordinate)

    # plot settings
    width = 0.1
    bins_x = np.arange(-5.5, 5.5, width)
    if coordinate == "x":
        width = 0.1
        bins_x = np.arange(-5.5, 5.5, width)
    if coordinate == "y":
        width = 0.5
        bins_x = np.arange(-50.5, 50.5, width)
    if coordinate == "z":
        width = 0.1
        bins_x = np.arange(-5.5, 5.5, width)

    bins = bins_x
    bins_center = bins[:-1] + (width / 2)
    # histogram for gaussian fit
    hist, _ = np.histogram(y_pred - y_true, bins=bins)
    # remove bins from fit with zero entries as they break the fitting
    popt, pcov, x, fx = auto_hist_fitting(f=f, bins=bins_center, hist=hist)
    x = np.linspace(min(bins), max(bins), 1000)
    fit_label = ""
    for i in range(len(popt)):
        fit_label += r"$p{}$ = {:.2f} $\pm$ {:.2f}".format(i, popt[i], np.sqrt(pcov[i, i])) + "\n"
    # plot
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
    ax.hist(y_pred - y_true, bins=bins, histtype=u"step", color="black")
    ax.errorbar(bins_center, hist, np.sqrt(hist), fmt=".", color="black")
    ax.plot(x, fx, color="red", label=fit_label)
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    ax.legend(loc="upper left", prop={'size': 12})
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_2dhist_position_residual_vs_true(y_pred,
                                          y_true,
                                          file_name,
                                          coordinate="x",
                                          particle="electron"):
    xlabel = r"${{{0}}}^{{true}}_{{{1}}}$ [mm]".format(particle, coordinate)
    ylabel = r"${{{0}}}^{{pred}}_{{{1}}} - {{{0}}}^{{true}}_{{{1}}}$ [mm]".format(particle,
                                                                                  coordinate)

    # plot settings
    width = 0.1
    bins_x = np.arange(0.0, 10.0, width)
    bins_y = np.arange(-5.5, 5.5, width)
    if coordinate == "x":
        bins_x = np.arange(-98.8 / 2.0, 98.8 / 2.0, width)
        bins_y = np.arange(-5.5, 5.5, width)
    if coordinate == "y":
        bins_x = np.arange(-100.0 / 2.0, 100.0 / 2.0, width)
        bins_y = np.arange(-60.5, 60.5, width)
    if coordinate == "z":
        bins_y = np.arange(-5.5, 5.5, width)
        if particle == "e":
            bins_x = np.arange(150.0 - 20.8 / 2.0, 150.0 + 20.8 / 2.0, width)
        if particle == "\gamma":
            bins_x = np.arange(270.0 - 46.8 / 2.0, 270.0 + 46.8 / 2.0, width)

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis="x", which="both", direction='in')
    ax.tick_params(axis="y", which="both", direction='in')
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top")
    ax.set_xlim(min(bins_x), max(bins_x))
    ax.set_ylim(min(bins_y), max(bins_y))
    h2d = ax.hist2d(x=y_true, y=y_pred - y_true, bins=[bins_x, bins_y], norm=LogNorm())
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.colorbar(h2d[3])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
