import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator

from .utils import auto_hist_fitting, get_fwhm

# global settings
plt.rcParams.update({'font.size': 14})


# ##################################################################################################
# Energy regression

def plot_1dhist_energy_residual(y_pred,
                                y_true,
                                file_name,
                                f="gaussian",
                                particle="e",
                                title=""):
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
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_1dhist_energy_residual_relative(y_pred,
                                         y_true,
                                         file_name,
                                         f="gaussian",
                                         particle="electron",
                                         title=""):
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
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    return popt


def plot_2dhist_energy_residual_vs_true(y_pred,
                                        y_true,
                                        file_name,
                                        particle="e",
                                        title=""):
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
    ax.set_title(title)
    plt.minorticks_on()
    plt.colorbar(h2d[3])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_2dhist_energy_residual_relative_vs_true(y_pred,
                                                 y_true,
                                                 file_name,
                                                 particle="e",
                                                 title=""):
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
    ax.set_title(title)
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
                                  particle="electron",
                                  title=""):
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
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    return popt


def plot_2dhist_position_residual_vs_true(y_pred,
                                          y_true,
                                          file_name,
                                          coordinate="x",
                                          particle="electron",
                                          title=""):
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
        bins_y = np.arange(-8.5, 8.5, width)####################################################
        bins_x = np.arange(233 - 20 / 2.0, 233 + 20 / 2.0, 2)


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
    h2d = ax.hist2d(x=y_true, y=y_pred - y_true, bins=[bins_x, bins_y], norm=LogNorm(vmin=1, vmax=1000))
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
    ax.set_title(title)
    plt.minorticks_on()
    #plt.colorbar(h2d[3])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_position_resolution(E_prim, y_true, y_pred, max_energy=10, energy_bins=20):
    data_array = np.stack((E_prim, y_pred-y_true))
    step_size = max_energy/energy_bins
    arr_fwhm = np.zeros(shape=energy_bins, dtype=np.float32)
    for i in range(energy_bins):
        low_energy_bound = i*step_size
        high_energy_bound = (i+1)*step_size
        energy_slice = np.logical_and(np.greater_equal(data_array[0,:], low_energy_bound), np.less(data_array[0,:], high_energy_bound))
        print("y_true")
        print(y_true[energy_slice])
        print("y_pred")
        print(y_pred[energy_slice])

        data_slice = data_array[1,energy_slice]
        arr_fwhm[i] = get_fwhm(data_slice, i)

    # plot the resolution
    plt.plot(arr_fwhm)
    plt.xticks(np.linspace(0,max_energy,11))
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=max_energy)
    plt.xlabel("Primary Energy / MeV")
    plt.ylabel("FWHM-y / mm")
    plt.title("y resolution")
    plt.grid()
    plt.savefig(r"/home/home2/institut_3b/clement/Master/FWHM_OLD_DATASET.png")
